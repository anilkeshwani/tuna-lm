import itertools
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, CROSS_ENTROPY_IGNORE_IDX, InstructTemplate, Message, padded_collate_sft, Role
from torchtune.datasets import text_completion_dataset
from torchtune.generation import generate
from torchtune.modules import TransformerDecoder
from torchtune.training import FullModelHFCheckpointer
from tqdm import tqdm

from tunalm.asr import asr_instruct_dataset, ASR_SFT_PROMPT_TEMPLATE
from tunalm.extendllama3 import setup_llama3_tokenizer
from tunalm.utils import get_terminal_width, info_excepthook


sys.excepthook = info_excepthook
TERMINAL_WIDTH = get_terminal_width()
LOGGER = utils.get_logger("DEBUG")
CROSS_ENTROPY_IGNORE_IDX: int  # NOTE we import torchtune default x-e ignore idx value (-100) from torchtune.data


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.device = utils.get_device(device=cfg.device)
        self.dtype = training.get_dtype(dtype=cfg.dtype, device=self.device)

        # NOTE Removed quantization setup; not used for now
        # self.output_dir = cfg.output_dir  # TODO where is this used? does the logger *actually* use it?

        self.seed = training.set_seed(seed=cfg.seed)  # TODO do we need to save the seed as an attribute?

    @staticmethod
    def resolve_vocab_size(cfg: DictConfig) -> DictConfig:
        # resolve extended model vocab size unless directly set - sum of base, special, and dsu tokens
        assert cfg.model.vocab_size is None, "Do not set vocab_size explicitly. It is inferred dynamically given n_dsus"
        cfg.model.vocab_size = cfg.base_vocab_size + cfg.n_special_tokens + cfg.n_dsus
        return cfg

    def setup(self, cfg: DictConfig) -> None:
        Path(cfg.checkpointer.output_dir).mkdir(parents=True, exist_ok=True)
        checkpointer: FullModelHFCheckpointer = config.instantiate(cfg.checkpointer)
        ckpt_dict = checkpointer.load_checkpoint()
        cfg = self.resolve_vocab_size(cfg)
        self.model = self.setup_model(
            cfg_model=cfg.model,
            model_state_dict=ckpt_dict[training.MODEL_KEY],
        )
        # NOTE Added for ASR SFT
        self.tokenizer, special_tokens_dynamic = setup_llama3_tokenizer(
            cfg.tokenizer.path,
            max_seq_len=cfg.tokenizer.max_seq_len,
            prompt_template=ASR_SFT_PROMPT_TEMPLATE,
            verbose=True,
        )
        # TODO Ensure the eos token is not added - this is a generation script!
        self.sampler_test, self.data_test = self.setup_test_data(
            cfg_dataset=cfg.data, batch_size=cfg.batch_size, collate_fn=padded_collate_sft
        )

        with open(Path(cfg.output_dir) / "config.yaml", "w") as f:  # TODO revert to "w" mode; defensive
            OmegaConf.save(cfg, f, resolve=True)
        LOGGER.info(f"Config saved to {cfg.output_dir}.")

    def setup_model(
        self,
        cfg_model: DictConfig,
        model_state_dict: dict[str, Any],
    ) -> TransformerDecoder:
        with training.set_default_dtype(self.dtype), self.device:
            model = config.instantiate(cfg_model)
        model.load_state_dict(model_state_dict)
        training.validate_expected_param_dtype(model.named_parameters(), dtype=self.dtype)
        LOGGER.info(f"Model is initialized with precision {self.dtype}.")
        return model

    def setup_test_data(
        self, cfg_dataset: DictConfig, batch_size: int, collate_fn: Callable
    ) -> tuple[DistributedSampler, DataLoader]:
        if isinstance(cfg_dataset, ListConfig):
            raise NotImplementedError("Dataset concatenation is not implemented. Please pass a single test set.")
        if cfg_dataset.get("packed"):
            raise NotImplementedError("Packing is not implemented / tested.")
        if "shuffle" in cfg_dataset.keys():
            raise ValueError("Do not set shuffle at inference.")

        # custom ASR dataset instantiation wrapper function with message_transform=ASRInputOutputToMessages
        ds = asr_instruct_dataset(self.tokenizer, **cfg_dataset)

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

    @torch.inference_mode()
    def generate(self, cfg: DictConfig):
        self.model.eval()
        custom_generate_next_token = None

        for i, batch in tqdm(enumerate(self.data_test), total=len(self.data_test)):
            t_start = time.perf_counter()
            utils.batch_to_device(batch, self.device)
            prompt: Tensor = batch["tokens"]
            # num_tokens += prompt.numel()  # TODO if not packed, surely this is meaningless (pad tokens)
            # num_tokens += (prmpt != self.tokenizer.pad_id).sum().item()  # TODO think it should be this
            # Ensure the cache is setup on the right device, with only as many tokens as we need
            if cfg.enable_kv_cache:
                with self.device:
                    self.model.setup_caches(
                        batch_size=cfg.batch_size,
                        dtype=self.dtype,
                        decoder_max_seq_len=prompt.numel() + cfg.max_new_tokens,
                    )
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
            t = time.perf_counter() - t_start
            LOGGER.info(f"Time for inference: {t:.02f} sec total")
            for generated_ids in generated_tokens:  # iterate over samples in the batch
                generated_str = self.tokenizer.decode(generated_ids)
                LOGGER.info(generated_str)
            t_start = time.perf_counter()

        # model_size = sum(
        #     [p.numel() * p.dtype.itemsize for p in itertools.chain(self.model.parameters(), self.model.buffers())]
        # )
        # tokens_generated = len(generated_tokens[0]) - prompt.size(0) # meaningless in batched setting
        # tokens_sec = tokens_generated / t
        # LOGGER.info(f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
        # LOGGER.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        # LOGGER.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
