# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
import sys
from pathlib import Path
from pprint import pformat  # noqa: F401; used in debugging
from typing import Any, Dict

import torch
from omegaconf import DictConfig
from sardalign.utils import multivariate_normal_from_weights, seed_everything
from tiktoken.load import load_tiktoken_bpe
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torchtune import config, training, utils
from torchtune.models.llama3._tokenizer import LLAMA3_SPECIAL_TOKENS


TERMINAL_WIDTH = os.get_terminal_size().columns
log = utils.get_logger("DEBUG")


class Llama321BExtender:
    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        # Disable for fp16, as we haven't validated "full" fp16 with this recipe, nor
        # enabled necessary features such as gradient scaling.
        if self._dtype == torch.float16:
            raise ValueError("full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead.")
        # Training config - NOTE I've disable this loading the recipe state
        # self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._resume_from_checkpoint = False

        # Added in to make things clearer
        self.seed = training.set_seed(seed=cfg.seed)  # this should not have any effect but no detriment
        self.epochs_run = 0  # per the FullFinetuneRecipeSingleDevice.__init__ -> used on save_checkpoint

    def _setup_ckpt_output_dir(self, cfg: DictConfig) -> None:
        if cfg.checkpointer.output_dir is not None:
            return
        # Set the checkpoint directory
        stem_suffix = f"-extended-{cfg._llama3_2_1b_config.n_new_tokens}-dsus"
        ckpt_dir = Path(cfg.checkpointer.checkpoint_dir)
        ckpt_dir = ckpt_dir.with_stem(ckpt_dir.stem + stem_suffix)
        # Create the output directory if it doesn't exist
        if not ckpt_dir.is_dir():
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Created output checkpoint directory for extended model at: {ckpt_dir!s}")
        # NOTE don't think cfg.checkpointer.output_dir being Path is a problem (cast later by checkpointer) but safer
        cfg.checkpointer.output_dir = str(ckpt_dir)

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        # For us this is FullModelHFCheckpointer at the time of writing (2024-10-30)
        self._checkpointer = config.instantiate(cfg_checkpointer, resume_from_checkpoint=self._resume_from_checkpoint)

        # NOTE FullModelHFCheckpointer load_checkpoint method behaviour:
        # Iterates over the checkpoints in the checkpoints directory and "merges" the tensor weights found therein
        # by updating all possible keys in the state dictionary at each iteration
        # -> want to include only a single checkpoint in the directory if we want to load a non-final checkpoint
        # TODO Why the iteration? Does this mean not all weights are in every checkpoint? How will the above work then?
        # This assumes 4-digit checkpoint numbering -> can't get >9999 checkpoints in a single run which is fine
        # self._resume_from_checkpoint is True, the class method additionally does:
        # converted_state_dict.update(safe_torch_load(self._recipe_checkpoint, mmap=False)) to load recipe state e.g.
        # the optimizer state dict I'm assuming
        checkpoint_dict = self._checkpointer.load_checkpoint()

        # if self._resume_from_checkpoint:
        #     self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def setup(self, cfg: DictConfig) -> None:
        ############################################################################################
        # Tokenizer
        # Num new tokens + special token postions determine index of new embedding vectors
        ############################################################################################

        # NOTE OpenAI's tiktoken doesn't provide a hook to prevent loading from cache -> we'll at least check it matches
        with open(cfg.tokenizer.path, "rb") as f:
            expected_hash = hashlib.sha256(f.read()).hexdigest()
        # loads BPE merges from tokenizer.model
        mergeable_ranks = load_tiktoken_bpe(cfg.tokenizer.path, expected_hash)
        base_vocab_size = len(mergeable_ranks)
        assert base_vocab_size == max(mergeable_ranks.values()) + 1
        DYNAMIC_LLAMA3_SPECIAL_TOKENS = {
            k: v
            for k, v in zip(LLAMA3_SPECIAL_TOKENS, range(base_vocab_size, len(LLAMA3_SPECIAL_TOKENS) + base_vocab_size))
        }
        self._tokenizer = config.instantiate(cfg.tokenizer, special_tokens=DYNAMIC_LLAMA3_SPECIAL_TOKENS)

        log.info(f"Tokenizer was initialized from tokenizer model file at: {cfg.tokenizer.path}")
        pretty_special_tokens = pformat(DYNAMIC_LLAMA3_SPECIAL_TOKENS, sort_dicts=False, underscore_numbers=True)
        log.info(f"Dynamic Llama3 special tokens added to tokenizer: {pretty_special_tokens}")

        # n new tokens to add to the model
        self.base_vocab_size = cfg._llama3_2_1b_config.base_vocab_size
        self.special_tokens_size = len(LLAMA3_SPECIAL_TOKENS)
        self.n_new_tokens = self._tokenizer.vocab_size - self.base_vocab_size - self.special_tokens_size

        self._setup_ckpt_output_dir(cfg)

        # Loads model from checkpoint *always* given that these are CPT/FT scripts
        ckpt_dict = self.load_checkpoint(cfg.checkpointer)

        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._model = self._setup_model(cfg_model=cfg.model, model_state_dict=ckpt_dict[training.MODEL_KEY])

    def _setup_model(self, cfg_model: DictConfig, model_state_dict: Dict[str, Any]) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)
        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        log.info(f"Model is initialized with precision {self._dtype}.")

        return model

    def extend_model(self, cfg: DictConfig) -> None:
        # Extract the model's embedding layer - self._model.tok_embeddings.weight.data - which
        # natively has dimension torch.Size([128256, 2048]) then split it into:
        # - embedding weights for the base vocabulary: embeddings[:cfg._llama3_2_1b_config.base_vocab_size, :]
        # - embeddings for the special tokens: embeddings[cfg._llama3_2_1b_config.base_vocab_size:, :]
        # Then create a new embedding matrix for the new tokens which we initialize as a multivariate normal
        # with covariance matrix based on the empirical covariance of the base vocabulary embeddings and mean
        # based on the mean of the base vocabulary embeddings (vector mean)
        # Insert the new embeddings between the base vocabulary embeddings and the special token embeddings
        # and reconstruct the model's embedding layer with the new embeddings
        # Save the new model to the checkpoint directory

        # Reproducibility
        seed_everything(cfg.seed)

        # Retain the original embeddings
        original_embeddings = self._model.tok_embeddings.weight.data.clone()

        # Extract the model's embedding layer
        embeddings = self._model.tok_embeddings.weight.data
        base_vocab_embeddings = embeddings[: self.base_vocab_size, :]
        special_tokens_embeddings = embeddings[self.base_vocab_size :, :]
        assert self._tokenizer.vocab_size > self.base_vocab_size + self.special_tokens_size, "No new tokens to add"

        # Generate a MV Gaussian sampler based on the base vocabulary embeddings
        mvgaussian: MultivariateNormal = multivariate_normal_from_weights(
            base_vocab_embeddings,
            sigma_scaling=1e-5,  # 1e-5 is the default
        )
        new_token_embeddings = mvgaussian.sample(torch.Size((self.n_new_tokens,)))
        self._model.tok_embeddings.weight.data = torch.cat(
            (base_vocab_embeddings, new_token_embeddings, special_tokens_embeddings), dim=0
        )

        # Validate the new embeddings
        assert self._model.tok_embeddings.weight.data[: self.base_vocab_size, :].equal(
            original_embeddings[: self.base_vocab_size, :]
        ), "Base vocabulary embeddings have changed"
        assert self._model.tok_embeddings.weight.data[-self.special_tokens_size :, :].equal(
            original_embeddings[-self.special_tokens_size :, :]
        ), "Special token embeddings have changed"
        assert len(self._model.tok_embeddings.weight.data) == self._tokenizer.vocab_size
        assert (
            len(self._model.tok_embeddings.weight.data)
            == self.base_vocab_size + self.n_new_tokens + self.special_tokens_size
        )
        log.info(f"Added {self.n_new_tokens} embeddings have been added to the model.")

        breakpoint()

    def save_checkpoint(self) -> None:
        """
        Save state dict to file. The recipe save_checkpoint method is responsible for
        correctly creating the checkpoint dict and passing to the checkpointer.
        """
        ckpt_dict = {training.MODEL_KEY: self._model.state_dict()}
        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=self.epochs_run,
            intermediate_checkpoint=False,  # set to False since we're treating this as a base checkpoint
        )


@config.parse
def recipe_llama3_2_extender_main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="1B_extension", cfg=cfg)
    extender = Llama321BExtender(cfg=cfg)
    extender.setup(cfg=cfg)
    extender.extend_model(cfg=cfg)
    extender.save_checkpoint()


if __name__ == "__main__":
    sys.exit(recipe_llama3_2_extender_main())
