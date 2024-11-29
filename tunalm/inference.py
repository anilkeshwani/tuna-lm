import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig, ListConfig
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, training, utils
from torchtune.data import padded_collate_sft
from torchtune.generation import generate
from torchtune.modules import TransformerDecoder


# TODO HACK to import extendllama3; remove when this is a package
sys.path.append(str(Path(__file__).parent.resolve()))
from extendllama3 import setup_llama3_tokenizer  # noqa: E402; local import
from utils import info_excepthook  # noqa: E402; local import


sys.excepthook = info_excepthook
TERMINAL_WIDTH = os.get_terminal_size().columns
LOGGER = utils.get_logger("DEBUG")


class InferenceRecipeSingleDevice:
    def __init__(self, cfg: DictConfig) -> None:
        self.device = utils.get_device(device=cfg.device)
        self.dtype = training.get_dtype(cfg.dtype, device=self.device)
        if self.dtype == torch.float16:
            raise NotImplementedError("Full fp16 is not supported with this recipe. Use bf16 or fp32 instead.")
        # self.output_dir = cfg.output_dir  # TODO where is this used? does the logger *actually* use it?
        self.seed = training.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        self.checkpointer = config.instantiate(cfg.checkpointer)
        ckpt_dict = self.checkpointer.load_checkpoint()
        # self.compile = cfg.compile
        assert cfg.model.vocab_size is None, "Do not set vocab_size explicitly. It is inferred dynamically given n_dsus"
        cfg.model.vocab_size = cfg.base_vocab_size + cfg.n_special_tokens + cfg.n_dsus
        self.model = self.setup_model(
            cfg_model=cfg.model, compile_model=cfg.compile, model_state_dict=ckpt_dict[training.MODEL_KEY]
        )
        self.tokenizer, special_tokens_dynamic = setup_llama3_tokenizer(cfg.tokenizer.path)
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
        if cfg_dataset.get("shuffle") is not None:
            raise RuntimeError("Do not set shuffle for inference only.")
        if cfg_dataset.get("packed") is not None:
            raise RuntimeError("Do not set packed for inference only.")
        if isinstance(cfg_dataset, ListConfig):
            raise NotImplementedError("ConcatDataset is not supported for inference. Please pass a single test set.")
        ds = config.instantiate(cfg_dataset, tokenizer=self.tokenizer)
        sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=False, seed=0)
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=partial(collate_fn, padding_idx=self.tokenizer.pad_id, ignore_idx=None),
        )
        LOGGER.info(f"Test dataset and sampler initialized from {cfg_dataset.source}.")
        return sampler, dataloader

    @torch.inference_mode()
    def generate(
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
        self.model.train()
        return output, logits

    def cleanup(self) -> None:
        self.metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="FullFinetuneRecipeSingleDevice", cfg=cfg)
    recipe = InferenceRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
