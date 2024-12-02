import itertools
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.nn import Module
from torchtune import config, generation, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, InstructTemplate, Message


# TODO HACK to import extendllama3; remove when this is a package
sys.path.append(str(Path(__file__).parent.resolve()))
from extendllama3 import setup_llama3_tokenizer  # noqa: E402; local import
from utils import info_excepthook  # noqa: E402; local import


sys.excepthook = info_excepthook
TERMINAL_WIDTH = os.get_terminal_size().columns
LOGGER = utils.get_logger("DEBUG")


class InferenceRecipe:
    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        training.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        ckpt_dict = checkpointer.load_checkpoint()
        assert cfg.model.vocab_size is None, "Do not set vocab_size explicitly. It is inferred dynamically given n_dsus"
        cfg.model.vocab_size = cfg.base_vocab_size + cfg.n_special_tokens + cfg.n_dsus
        self.model = self.setup_model(model_cfg=cfg.model, model_state_dict=ckpt_dict[training.MODEL_KEY])
        self.tokenizer, special_tokens_dynamic = setup_llama3_tokenizer(cfg.tokenizer.path, verbose=False)

    def setup_model(self, model_cfg: DictConfig, model_state_dict: dict[str, Any]) -> Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)
        model.load_state_dict(model_state_dict)
        training.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        LOGGER.info(f"Model is initialized with precision {self._dtype}.")
        return model

    def convert_prompt_to_tokens(
        self,
        prompt: DictConfig | str,
        chat_format: ChatFormat | None,
        instruct_template: InstructTemplate | None,
    ) -> list[Message]:
        """
        Either:
        (1) a raw string is passed as the prompt, in which case we call tokenizer.encode directly, or
        (2) a DictConfig is passed as the prompt. In this case there are three possibilities:
            (a) an InstructTemplate is provided. Since instruct templates output a string, we will
                call tokenizer.encode on the output of the instruct template.
            (b) a ChatFormat is provided. Since chat formats output a list of messages, we will
                call tokenizer.tokenize_messages on the output of the chat format.
            (c) neither an InstructTemplate nor a ChatFormat is provided. In this case we will
                convert the DictConfig to a list of messages and call tokenizer.tokenize_messages directly.
        """

        # Should only be chat-style prompt or instruct-style prompt
        if chat_format and instruct_template:
            raise ValueError("Cannot pass both chat format and instruct template for generation")

        # If instruct template is provided, assert that the prompt is a DictConfig
        # and apply it
        if instruct_template:
            if not isinstance(prompt, DictConfig):
                raise ValueError("Cannot apply instruct template to raw string")
            instruct_template = _get_component_from_path(instruct_template)
            prompt = instruct_template.format(prompt)

        # To hit this block, either the raw prompt is a string or an
        # instruct template has been provided to convert it to a string
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, add_bos=True, add_eos=False)

        # dict.items() will respect order for Python >= 3.7
        else:
            messages = [Message(role=k, content=v) for k, v in prompt.items()]
            messages += [Message(role="assistant", content="")]
            if chat_format:
                chat_format = _get_component_from_path(chat_format)
                messages = chat_format.format(messages)
            return self.tokenizer.tokenize_messages(messages)[0]

    @torch.inference_mode()
    def generate(self, cfg: DictConfig):
        tokens = self.convert_prompt_to_tokens(
            cfg.prompt,
            cfg.get("chat_format", None),
            cfg.get("instruct_template", None),
        )
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

        custom_generate_next_token = None

        # Ensure the cache is setup on the right device, with only as many tokens as we need
        if cfg.enable_kv_cache:
            with self._device:
                self.model.setup_caches(
                    batch_size=1,
                    dtype=self._dtype,
                    decoder_max_seq_len=prompt.numel() + cfg.max_new_tokens,
                )

        t0 = time.perf_counter()
        generated_tokens, _ = generation.generate(
            model=self.model,
            prompt=prompt,
            max_generated_tokens=cfg.max_new_tokens,
            pad_id=self.tokenizer.pad_id,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            stop_tokens=self.tokenizer.stop_tokens,
            custom_generate_next_token=custom_generate_next_token,
        )
        generated_tokens = generated_tokens.tolist()
        t = time.perf_counter() - t0

        LOGGER.info(self.tokenizer.decode(generated_tokens[0]))

        model_size = sum(
            [p.numel() * p.dtype.itemsize for p in itertools.chain(self.model.parameters(), self.model.buffers())]
        )

        tokens_generated = len(generated_tokens[0]) - prompt.size(0)
        tokens_sec = tokens_generated / t
        LOGGER.info(f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
        LOGGER.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        LOGGER.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
