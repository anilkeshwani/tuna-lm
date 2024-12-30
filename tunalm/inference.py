import itertools
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig, ListConfig
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, CROSS_ENTROPY_IGNORE_IDX, InstructTemplate, Message, padded_collate_sft, Role
from torchtune.generation import generate
from torchtune.modules import TransformerDecoder


# TODO HACK to import extendllama3; remove when this is a package
sys.path.append(str(Path(__file__).parent.resolve()))
from asr import asr_instruct_dataset  # noqa: E402; local import
from extendllama3 import setup_llama3_tokenizer  # noqa: E402; local import
from utils import get_terminal_width, info_excepthook  # noqa: E402; local import


sys.excepthook = info_excepthook
TERMINAL_WIDTH = get_terminal_width()
LOGGER = utils.get_logger("DEBUG")
CROSS_ENTROPY_IGNORE_IDX: int  # NOTE we import torchtune default x-e ignore idx value (-100) from torchtune.data


class InferenceRecipe:
    def __init__(self, cfg: DictConfig) -> None:
        self.device = utils.get_device(device=cfg.device)
        self.dtype = training.get_dtype(dtype=cfg.dtype, device=self.device)
        # self.output_dir = cfg.output_dir  # TODO where is this used? does the logger *actually* use it?
        self.seed = training.set_seed(seed=cfg.seed)  # TODO do we need to save the seed as an attribute?

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)  # no need to persist the checkpointer
        ckpt_dict = checkpointer.load_checkpoint()
        # self.compile = cfg.compile  # TODO support compilation; remove if not persisted after init
        assert cfg.model.vocab_size is None, "Do not set vocab_size explicitly. It is inferred dynamically given n_dsus"
        cfg.model.vocab_size = cfg.base_vocab_size + cfg.n_special_tokens + cfg.n_dsus
        self.model = self.setup_model(
            cfg_model=cfg.model, compile_model=cfg.compile, model_state_dict=ckpt_dict[training.MODEL_KEY]
        )
        self.tokenizer, special_tokens_dynamic = setup_llama3_tokenizer(cfg.tokenizer.path)
        # TODO Ensure the eos token is not added - this is a generation script!
        self.sampler_test, self.data_test = self.setup_test_data(
            cfg_dataset=cfg.data.test, batch_size=cfg.batch_size, collate_fn=padded_collate_sft
        )

    def setup_model(
        self, cfg_model: DictConfig, compile_model: bool, model_state_dict: dict[str, Any]
    ) -> TransformerDecoder:
        with training.set_default_dtype(self.dtype), self.device:
            model = config.instantiate(cfg_model)
        if compile_model:
            training.compile_model(model)
        model.load_state_dict(model_state_dict)
        training.validate_expected_param_dtype(model.named_parameters(), dtype=self.dtype)
        LOGGER.info(f"Model is initialized with precision {self.dtype}.")
        return model

    def setup_test_data(
        self, cfg_dataset: DictConfig, batch_size: int, collate_fn: Callable
    ) -> tuple[DistributedSampler, DataLoader]:
        if isinstance(cfg_dataset, ListConfig):
            raise NotImplementedError("ConcatDataset is not supported for inference. Please pass a single test set.")
        if cfg_dataset.get("packed") is not None:
            raise RuntimeError("Do not set packed for inference only.")
        if cfg_dataset.get("shuffle") is not None:
            raise RuntimeError("Do not set shuffle for inference only.")
        cfg_dataset.pop("shuffle")  # no such key in the datasets dataset builder config
        ds = asr_instruct_dataset(self.tokenizer, **cfg_dataset)  # custom; message_transform=ASRInputOutputToMessages
        sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=False, seed=0)
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=False,  # dropping last avoids shape issues with compile + flex attention; TODO any issues?
            collate_fn=partial(
                collate_fn,
                padding_idx=self.tokenizer.pad_id,
                ignore_idx=CROSS_ENTROPY_IGNORE_IDX,  # NOTE torchtune.data default crossentropy ignore idx
            ),
        )
        LOGGER.info(f"Test dataset and sampler initialized from {cfg_dataset.source}.")
        return sampler, dataloader

    def convert_prompt_to_tokens(
        self,
        prompt: dict[Role, str],
    ) -> list[int]:
        """
        Convert the prompt string to a user message with optional system messages
        and tokenize using the prompt template defined on the tokenizer.
        """
        messages = []
        if "system" in prompt and prompt["system"] is not None:
            messages.append(Message(role="system", content=prompt["system"]))
        messages.extend(
            [
                Message(role="user", content=prompt["user"]),
                Message(role="assistant", content=""),  # empty assistant message to kick-start generation
            ]
        )
        return self.tokenizer({"messages": messages}, inference=True)["tokens"]

    @torch.inference_mode()
    def generate(self, cfg: DictConfig):
        tokens = self.convert_prompt_to_tokens(cfg.prompt)
        prompt = torch.tensor(tokens, dtype=torch.int, device=self.device)
        custom_generate_next_token = None
        # Ensure the cache is setup on the right device, with only as many tokens as we need
        if cfg.enable_kv_cache:
            with self.device:
                self.model.setup_caches(
                    batch_size=1,
                    dtype=self.dtype,
                    decoder_max_seq_len=prompt.numel() + cfg.max_new_tokens,
                )

        t0 = time.perf_counter()
        generated_tokens, _ = generate(
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

    @torch.inference_mode()
    def inference(
        self,
        prompts: list[str],
        temperature: float = 0.0,  # greedy by default # TODO should I set this to eps e.g. 1-e-6?
        top_k: int | None = None,  # TODO implement nucleus sampling
        max_generated_tokens: int = 512,
        rng: torch.Generator | None = None,
        custom_generate_next_token: Callable | None = None,
        stop_tokens: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Generates tokens from a model conditioned on a prompt, and also returns logits for the generations.

        Args:
            prompts (list[str]): list of prompts
            max_generated_tokens (int): maximum number of tokens generated
            temperature (float): value to scale the predicted logits by; greedy decoding by default (0.0)
            top_k (int | None): If specified, we prune the sampling to only token ids within the top_k probabilities,
                default None.
            rng (torch.Generator | None): random number generator, default None.
            custom_generate_next_token (Callable | None): Custom next token generation function; generally only useful
                to specify a ``torch.compile`` version of the generate next token for performance reasons.
                If None, we use the default :func:`generate_next_token`. Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple of two tensors:
                - tokens (torch.Tensor): tensor with the generated tokens,
                    with shape ``[bsz x seq_len + num_generated_tokens]`` where ``num_generated_tokens``
                    may be less than ``max_generated_tokens`` if ``stop_tokens`` are provided.
                - logits (torch.Tensor): tensor with the logits associated with the generated tokens,
                    with shape ``[bsz x seq_len + num_generated_tokens x vocab_size]``.

        NOTE: This is a thin wrapper around torchtune.generation.generate. torch.inference_mode decorator is redundant.
        """

        # TODO enure that the EOS is not added for generation self.tokenizer.encode(prompt, add_bos=True, add_eos=False)

        # TODO Add timing for generation with t0 = time.perf_counter()
        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_id]
        prompts_ids: Tensor = torch.tensor([self.tokenizer.encode(prompt) for prompt in prompts]).to(self.device)
        self.model.eval()
        output, logits = generate(
            self.model,
            prompts_ids,
            max_generated_tokens=max_generated_tokens,
            pad_id=self.tokenizer.pad_id,  # TODO CHECK
            temperature=temperature,
            top_k=top_k,
            stop_tokens=stop_tokens,
            rng=rng,
            custom_generate_next_token=custom_generate_next_token,
        )

        model_size = sum(
            [p.numel() * p.dtype.itemsize for p in itertools.chain(self.model.parameters(), self.model.buffers())]
        )
        LOGGER.info(f"Model size: {model_size}")
        self.model.train()
        return output, logits


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
