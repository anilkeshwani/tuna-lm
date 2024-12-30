import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.modules import TransformerDecoder


# TODO HACK to import extendllama3; remove when this is a package
sys.path.append(str(Path(__file__).parent.resolve()))
from extendllama3 import setup_llama3_tokenizer  # noqa: E402; local import
from utils import get_terminal_width, info_excepthook  # noqa: E402; local import


sys.excepthook = info_excepthook
TERMINAL_WIDTH = get_terminal_width()
LOGGER = utils.get_logger("DEBUG")


class ValidationOfflineRecipe:
    def __init__(self, cfg: DictConfig) -> None:
        # hardware & precision
        self.device = utils.get_device(device=cfg.device)
        self.dtype = training.get_dtype(cfg.dtype, device=self.device)
        if self.dtype != torch.float32:
            raise ValueError("Perform validation with float32 precision only.")  # TODO necessary? understand bf16

        # logging attributes
        self.log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self.log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # training config
        self.resume_from_checkpoint = cfg.checkpointer.resume_from_checkpoint

        # public attrs updated by the checkpoint loader when resume_from_checkpoint=True or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)

    def setup_checkpointer(self, cfg: DictConfig) -> DictConfig:
        Path(cfg.checkpointer.output_dir).mkdir(parents=True, exist_ok=True)
        # checkpoint_dir of format global-step-053862; cutoff of epoch 0 and 1: > 040492
        # HACK here we only have epochs {0,1}
        epoch = int(int(cfg.checkpointer.checkpoint_dir.split("-")[-1].strip("/")) > 40_492)
        cfg.checkpointer.checkpoint_files = [f"hf_model_0001_{epoch}.pt"]  # HACK
        # HACK (this one's horrible) we symlink a copy of the _config_json_path_hack to the checkpoint_dir on the fly
        if not Path(cfg.checkpointer.checkpoint_dir, "config.json").exists():
            Path(cfg.checkpointer.checkpoint_dir, "config.json").symlink_to(cfg._config_json_path_hack)
        self.checkpointer = config.instantiate(cfg.checkpointer)
        return cfg

    @staticmethod
    def resolve_vocab_size(cfg: DictConfig) -> DictConfig:
        # resolve extended model vocab size unless directly set - sum of base, special, and dsu tokens
        assert cfg.model.vocab_size is None, "Do not set vocab_size explicitly. It is inferred dynamically given n_dsus"
        cfg.model.vocab_size = cfg.base_vocab_size + cfg.n_special_tokens + cfg.n_dsus
        return cfg

    def setup(self, cfg: DictConfig) -> None:
        cfg = self.setup_checkpointer(cfg=cfg)  # sets up checkpointer; and cfg.checkpointer.output_dir if not set
        ckpt_dict = self.checkpointer.load_checkpoint()
        model_state_dict = ckpt_dict[training.MODEL_KEY]
        self.compile = cfg.compile
        cfg = self.resolve_vocab_size(cfg)
        self.model = self.setup_model(cfg.model, self.compile, model_state_dict)
        self.tokenizer, special_tokens_dynamic = setup_llama3_tokenizer(cfg.tokenizer.path)
        self.loss_fn = config.instantiate(cfg.loss)  # initialize loss function
        if self.compile:
            training.compile_loss(self.loss_fn)
        if self.loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            self.model.set_num_output_chunks(self.loss_fn.num_output_chunks)
        LOGGER.info("Loss is initialized.")
        self.sampler_dev, self.data_dev = self.setup_data(
            cfg_dataset=cfg.data.dev,
            batch_size=cfg.batch_size * 2,  # NOTE heuristic
            collate_fn=cfg.get("collate_fn", "torchtune.data.padded_collate_sft"),
        )
        self.steps_devset = max(1, len(self.data_dev))

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
        if self.device.type == "cuda":
            memory_stats = training.get_memory_stats(device=self.device)
            training.log_memory_stats(memory_stats)
        return model

    def setup_data(
        self,
        cfg_dataset: DictConfig,
        batch_size: int,
        collate_fn: str,
    ) -> tuple[DistributedSampler, DataLoader]:
        cfg_dataset.pop("shuffle")  # no such key in the datasets dataset builder config
        shuffle = False
        ds = config.instantiate(cfg_dataset, self.tokenizer)
        packed = cfg_dataset.get("packed", False)
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(collate_fn)
        sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=shuffle, seed=0)
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=False,  # dropping last avoids shape issues with compile + flex attention
            collate_fn=(
                partial(collate_fn, padding_idx=self.tokenizer.pad_id, ignore_idx=self.loss_fn.ignore_index)
                if not packed
                else padded_collate_packed
            ),
        )
        LOGGER.info(f"Dataset and Sampler initialized from {cfg_dataset.source}.")
        return sampler, dataloader

    def loss_step(self, batch: dict[str, Tensor]) -> Tensor:
        # Shape [b, s], needed for the loss not the model
        labels = batch.pop("labels")
        logits = self.model(**batch)
        # Shift labels to compute loss; equivalent to labels[..., 1:] and logits[..., :-1, :] w/o slicing logits
        # TODO NOTE cache this as an instance variable if slowdown detected; does this form torch.compile?
        labels = torch.hstack(
            (labels[..., 1:], torch.full((labels.size(0), 1), self.loss_fn.ignore_index, device=self.device))
        )
        if not isinstance(logits, list):
            labels = labels.reshape(-1)
            logits = logits.reshape(-1, logits.size(-1))
        loss = self.loss_fn(logits, labels)  # compute loss
        del logits  # free logits otherwise it peaks backward memory
        return loss

    @torch.inference_mode()
    def validate(self) -> None:
        if self.compile:
            LOGGER.info("torch.compile is enabled. Expect a slow first iteration; model is compiled in first forward.")
        dev_loss = 0
        self.model.eval()
        for i_dev, dev_batch in enumerate(self.data_dev):
            utils.batch_to_device(dev_batch, self.device)
            dev_loss_batch = self.loss_step(dev_batch)
            LOGGER.info(
                f"Iter {i_dev:0{len(str(self.steps_devset))}d} / {self.steps_devset} | "
                f"Dev Batch {i_dev:0{len(str(len(self.data_dev)))}d} / {len(self.data_dev)} | "
                f"Dev Loss (batch): {dev_loss_batch.item():.4f}"
            )
            dev_loss += dev_loss_batch.item()  # reduction is mean over non-ignored label elements
        LOGGER.info(f"Dev Loss (total over dev set): {dev_loss / len(self.data_dev):.4f}")
        self.model.train()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="ValidationOfflineRecipe", cfg=cfg)
    recipe = ValidationOfflineRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.validate()


if __name__ == "__main__":
    sys.exit(recipe_main())
