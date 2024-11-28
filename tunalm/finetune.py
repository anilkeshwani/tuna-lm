# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pdb
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from pprint import pformat, pp  # noqa: F401; used in debugging
from typing import Any, Optional
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchtune.training.metric_logging import WandBLogger
from tqdm import tqdm


sys.path.append(str(Path(__file__).parent.resolve()))
from extendllama3 import setup_llama3_tokenizer  # noqa: E402; local import


def info_excepthook(type, value, tb):
    if hasattr(sys, "ps1") or not sys.stderr.isatty():
        # interactive mode or we don't have a tty-like device: call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        # NOT in interactive mode: print the exception then start the debugger in post-mortem mode
        traceback.print_exception(type, value, tb)
        pdb.post_mortem(tb)


sys.excepthook = info_excepthook
TERMINAL_WIDTH = os.get_terminal_size().columns
LOGGER = utils.get_logger("DEBUG")

# Constants
# NOTE torchtune.training exports STEPS_KEY = "steps_run" # number of steps completed thus far - for PPO
GLOBAL_STEP_KEY: str = "global_step"


class FullFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    Full finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe is optimized
    for single GPU training. Training on CPU is not supported.

    Features:
        - Activation Checkpointing. This can be controlled using the ``activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * gradient accumulation steps.

            For example: with batch_size=1 and gradient_accumulation_steps=32 we get a total batch size of 32.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Optimizer in Backward. Fusing the optimizer step into the backward pass helps reduce the memory
            footprint associated with gradients. This can be especially helpful when you are memory
            constrained. Note that users can only use ONE of gradient accumulation or optimizer in backward.
            These features currently do not work together. For more details on optimizer in backward, please
            see this tutorial: https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html

        - Lower precision optimizers. This recipe supports lower-precision optimizers from the bitsandbytes
            library (https://huggingface.co/docs/bitsandbytes/main/en/index). We've tested the recipe with
            8-bit AdamW and Paged AdamW. These optimizers are especially helpful when you are memory constrained
            since they help reduce the memory footprint associated with the optimizer states.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training.

            Resuming training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/deep_dives/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

        - Gradient Clipping. Gradient clipping is supported using the ``clip_grad_norm`` flag. By default,
            ``clip_grad_norm`` is set to ``None``. If you only want to log the grad norm, you can set
            ``clip_grad_norm='inf'``.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
        RuntimeError: If ``gradient_accumulation_steps > 1`` and ``optimizer_in_bwd`` is `True`.
        RuntimeError: If ``left_pad_sequence`` is set as the data collator.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.validate_cfg(cfg=cfg)

        # hardware & precision
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        if self._dtype == torch.float16:
            raise NotImplementedError("Full fp16 training is not supported with this recipe. Use bf16 or fp32 instead.")

        # logging attributes
        self._output_dir = cfg.output_dir  # TODO where is this used? does the logger *actually* use it?
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # training config
        self._resume_from_checkpoint = cfg.checkpointer.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._optimizer_in_bwd = cfg.optimizer_in_bwd

        # public attrs updated by the checkpoint loader when resume_from_checkpoint=True or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self.save_steps = cfg.save_steps

        # training duration; resolved downstream
        self.total_epochs = cfg.epochs
        self.max_steps = cfg.max_steps

        # initialize counters for epochs run and global steps; updated downstream by the checkpoint loader
        self.epochs_run = 0
        self.global_step = 0

    # TODO Recipe:
    # - [x] max_steps_per_epoch -  check all gone
    # - [x] resolve max_steps by epoch or epoch by max_steps - resolver method
    # - [x] add hack to set checkpointer output_dir in save_checkpoint to specify output dir
    # - [x] handle the intermediate checkpoint variable on the basis of self.global_step vs max_steps
    # - [ ] Add support for max iterations
    # - [ ] Add evaluation loop every eval_steps iterations
    # - [ ] create MLS validation set (both interleaved, ASR concat style)
    # - [ ] add validation loop
    # - [ ] compute ASR on validation as well - HF evaluate with normalizations
    # - [ ] fix out dir vs ckpt dir - config JSON is written to  checkpoints/ dir
    # - [ ] maybe save cfg and an instance variable?
    # - [ ] future: do we want to save only one recipe state; maybe for now we can remove the previous one?
    #               relatively easy to incorporate into save_checkpoint method - track previous out dir and
    #               remove recipe.pt

    def validate_cfg(self, cfg: DictConfig) -> None:
        if cfg.gradient_accumulation_steps > 1 and cfg.optimizer_in_bwd:
            raise RuntimeError(
                "Gradient accumulation is not supported with optimizer in bwd."
                "Set gradient_accumulation_steps=1 or optimizer_in_bwd=False."
            )
        if bool(cfg.epochs) == bool(cfg.max_steps):
            raise ValueError("Either epochs or max_steps must be set, but not both.")

        if cfg.max_steps is not None:
            raise NotImplementedError("Implement max_steps support.")  # TODO implement max_steps support

    def setup_checkpointer(self, cfg: DictConfig) -> DictConfig:
        assert self.experiment_name is not None, "Experiment name must be defined before setting up checkpointer"
        if cfg.checkpointer.output_dir is None:
            _exp_name = self.experiment_name + (f"-id_{self._wandb_run_id}" if self._wandb_run_id else "")
            _ckptr_out_dir_parts = cfg.experiments_root_dir, cfg.model_name, _exp_name, "checkpoints"
            cfg.checkpointer.output_dir = str(Path(*_ckptr_out_dir_parts))
        # NOTE FullModelHFCheckpointer mkdir parents=False in __init__ before save_config to cfg.checkpointer.output_dir
        Path(cfg.checkpointer.output_dir).mkdir(parents=True, exist_ok=True)
        # create and save the checkpointer as an instance attribute
        self._checkpointer = config.instantiate(cfg.checkpointer)
        return cfg

    def load_checkpoint(self) -> dict[str, Any]:
        checkpoint_dict = self._checkpointer.load_checkpoint()
        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: dict[str, Any]) -> None:
        """Updates the recipe state from checkpoint."""
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]
            self.global_step = ckpt_dict[GLOBAL_STEP_KEY]

            # Prevented overrides
            if self.seed != ckpt_dict[training.SEED_KEY]:
                self.seed = ckpt_dict[training.SEED_KEY]
                warn(f"Config value for seed does not match checkpoint value. Used checkpoint value: {self.seed}")
            # Allowed overrides
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(f"Overriding checkpoint total_epochs with value specified in config: {self.total_epochs}")

            LOGGER.info(f"Successfully resumed training from recipe state in: {self._checkpointer._recipe_checkpoint}")
            LOGGER.info(f"Resuming from epoch {self.epochs_run} and global step {self.global_step}")
        except KeyError as e:
            raise KeyError("Recipe checkpoint missing required keys. Ensure recipe checkpoint path is correct.") from e

    @staticmethod
    def resolve_vocab_size(cfg: DictConfig) -> DictConfig:
        # resolve extended model vocab size unless directly set - sum of base, special, and dsu tokens
        assert cfg.model.vocab_size is None, "Do not set vocab_size explicitly. It is inferred dynamically given n_dsus"
        cfg.model.vocab_size = cfg.base_vocab_size + cfg.n_special_tokens + cfg.n_dsus
        return cfg

    def setup(self, cfg: DictConfig) -> None:
        """Sets up recipe state including recipe attributes based on ``resume_from_checkpoint`` flag"""
        self._metric_logger = config.instantiate(cfg.metric_logger)

        if cfg.experiment_name is None:
            if not isinstance(self._metric_logger, WandBLogger):
                raise ValueError("experiment_name must be defined in config if not using WandBLogger (with run name)")
            self.experiment_name = self._metric_logger._wandb.run.name

        if isinstance(self._metric_logger, WandBLogger):
            self._wandb_run_id = self._metric_logger._wandb.run.id
            self.wandb_entity = self._metric_logger._wandb.api.viewer().get("entity")
        else:
            self._wandb_run_id = None
            self.wandb_entity = None

        cfg = self.setup_checkpointer(cfg=cfg)  # sets up checkpointer; and cfg.checkpointer.output_dir if not set
        ckpt_dict = self.load_checkpoint()

        self._compile = cfg.compile

        cfg = self.resolve_vocab_size(cfg)

        # _setup_model handles initialization loads state dict; should be called before
        # _setup_optimizer since transforming the optimizer state dict requires the model
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            compile_model=self._compile,
            model_state_dict=ckpt_dict[training.MODEL_KEY],
        )

        self._tokenizer, special_tokens_dynamic = setup_llama3_tokenizer(cfg.tokenizer.path)

        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            optimizer_in_bwd=cfg.optimizer_in_bwd,
            opt_state_dict=(ckpt_dict[training.OPT_KEY] if self._resume_from_checkpoint else None),
        )

        self._loss_fn = config.instantiate(cfg.loss)  # initialize loss function

        if self._compile:
            training.compile_loss(self._loss_fn)

        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)

        LOGGER.info("Loss is initialized.")

        # sampler & dataloader depend on tokenizer & loss_fn -> set up after dependencies are initialized
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=cfg.get("collate_fn", "torchtune.data.padded_collate_sft"),
        )

        # update recipe state - can only be correctly set after all other components have been initialized and updated
        self._steps_per_epoch = max(1, len(self._dataloader) // self._gradient_accumulation_steps)
        self.max_steps = self.total_epochs * self._steps_per_epoch

        # set up lr scheduler
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.max_steps,
            last_epoch=self.global_step - 1,
        )

        # set up profiler - returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full((cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device)

        # log config with parameter override - do this last as methods resolve config items
        self._metric_logger.log_config(cfg)

    def _setup_profiler(self, cfg_profiler: DictConfig | None = None) -> torch.profiler.profile | DummyProfiler:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler

        Args:
            cfg_profiler (DictConfig | None): ``profiler`` section of the top-level ``cfg`` (the main config passed to
                `recipe.main`). Default None.

        Returns:
            profiler: Union[torch.profiler.profile, DummyProfiler] - DummyProfiler is a nullcontext with no-op methods
            for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
            that the instrumented training loop does not need to be changed profiling is disabled.

        The profiler config can be provided in configs under the `profiler` key with the following layout:

        .. code-block:: yaml
            profiler:
                enabled: bool

                #Output directory of trace artifacts
                output_dir: str

            #`torch.profiler.ProfilerActivity` types to trace
            cpu: bool
            cuda: bool

                #Trace options
                profile_memory: bool
                with_stack: bool
                record_shapes: bool
                with_flops: bool

            # `torch.profiler.schedule` options:
            # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
            wait_steps: int
            warmup_steps: int
            active_steps: int
            num_cycles: int
        """

        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_") == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        LOGGER.info(f" Profiler config after instantiation: {profiler_cfg}")

        self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
        if profiler_cfg["enabled"]:
            self.profiler_wait_steps = profiler_cfg["wait_steps"]
            self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
            self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        compile_model: bool,
        model_state_dict: dict[str, Any],
    ) -> nn.Module:
        """
        Set up the model including enabling activation checkpointing.
        """
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        if compile_model:
            training.compile_model(model)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(model, auto_wrap_policy={modules.TransformerSelfAttentionLayer})

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        LOGGER.info(f"Model is initialized with precision {self._dtype}.")

        if self._device.type == "cuda":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        """
        Set up the optimizer. This method also handles loading the optimizer state_dict, if specified.
        """
        if optimizer_in_bwd:
            # Maintain a dict of optims for every parameter.
            optim_dict = {p: config.instantiate(cfg_optimizer, [p]) for p in self._model.parameters()}
            # Register optimizer step hooks on the model to run optimizer in backward.
            training.register_optim_in_bwd_hooks(model=self._model, optim_dict=optim_dict)
            # Create a wrapper for checkpoint save/load of optimizer states when running in backward.
            self._optim_ckpt_wrapper = training.create_optim_in_bwd_wrapper(model=self._model, optim_dict=optim_dict)
            # Load optimizer states. If optimizer states are being restored in an optimizer in backward
            # run, these need to have been saved with the same setting. Cannot restore from runs that did not
            # use optimizer in backward.
            if opt_state_dict is not None:
                try:
                    self._optim_ckpt_wrapper.load_state_dict(opt_state_dict)
                except BaseException as e:
                    raise RuntimeError(
                        "Failed loading in-backward optimizer checkpoints."
                        "Please make sure run being restored from was using in-backward optimizer."
                    ) from e
            LOGGER.info("In-backward optimizers are set up.")
            return None
        else:
            optimizer = config.instantiate(cfg_optimizer, self._model.parameters())

            if opt_state_dict:
                optimizer.load_state_dict(opt_state_dict)
            LOGGER.info("Optimizer is initialized.")
            return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig | None,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optional[Optimizer]:
        """
        Set up the learning rate scheduler based on the provided configuration.
        It handles both standard optimization and optimizer-in-backward cases, and supports
        schedulers from both torchtune.modules and torch.optim.

        Args:
            cfg_lr_scheduler (DictConfig | None): The learning rate scheduler configuration.
            num_training_steps (int): The total number of training steps.
            last_epoch (int): The index of the last epoch.

        Returns:
            lr_scheduler (Optional[Optimizer]): The learning rate scheduler.
        """
        if cfg_lr_scheduler is None:
            LOGGER.info("No learning rate scheduler configured. Using constant learning rate.")
            return None

        if self._optimizer_in_bwd:
            # Use the first optimizer from the wrapper to represent the learning rate
            optimizer = next(iter(self._optim_ckpt_wrapper.optim_map.values()))
        else:
            # Standard case: use the single optimizer
            optimizer = self._optimizer

        # Instantiate the learning rate scheduler
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        if self._optimizer_in_bwd:
            # Modify the scheduler for optimizer_in_bwd case
            self._optim_ckpt_wrapper.set_lr_scheduler(lr_scheduler)

        LOGGER.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
    ) -> tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        if isinstance(cfg_dataset, ListConfig):
            datasets = [config.instantiate(single_cfg_dataset, self._tokenizer) for single_cfg_dataset in cfg_dataset]
            ds = ConcatDataset(datasets=datasets)
            packed = False
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        # Instantiate collate_fn
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(collate_fn)

        sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=shuffle, seed=0)
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,  # dropping last avoids shape issues with compile + flex attention
            collate_fn=(
                partial(collate_fn, padding_idx=self._tokenizer.pad_id, ignore_idx=self._loss_fn.ignore_index)
                if not packed
                else padded_collate_packed
            ),
        )

        LOGGER.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(self, epoch: int) -> None:
        # HACK to avoid monkey patching HF checkpointer to set output_dir for iteration-based ckpts
        _ckptr_output_dir_canonical: Path = self._checkpointer._output_dir  # save current (top-level) ckptr out dir
        stepwise_subdir = f"global-step-{self.global_step:0{len(str(self.max_steps))}d}"
        self._checkpointer._output_dir = _ckptr_output_dir_canonical / stepwise_subdir
        ckpt_dict = {training.MODEL_KEY: self._model.state_dict()}
        # if training is in-progress, checkpoint the optimizer state as well
        if self.global_step + 1 < self.max_steps:
            ckpt_dict.update(
                {
                    training.SEED_KEY: self.seed,
                    training.EPOCHS_KEY: self.epochs_run,
                    training.TOTAL_EPOCHS_KEY: self.total_epochs,
                    GLOBAL_STEP_KEY: self.global_step,
                }
            )
            if not self._optimizer_in_bwd:
                ckpt_dict[training.OPT_KEY] = self._optimizer.state_dict()
            else:
                ckpt_dict[training.OPT_KEY] = self._optim_ckpt_wrapper.state_dict()
        self._checkpointer._output_dir.mkdir(parents=False, exist_ok=False)  # NOTE canonical required; shouldn't exist
        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=epoch,
            intermediate_checkpoint=(self.global_step + 1 < self.max_steps),
        )
        self._checkpointer._output_dir = _ckptr_output_dir_canonical  # reset to top-level ckptr out dir for next ckpt

    def _loss_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Shape [b, s], needed for the loss not the model
        labels = batch.pop("labels")

        logits = self._model(**batch)

        # Shift labels to compute loss
        # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
        # But this way we dont need to slice the logits. We just add an ignore index to labels.
        labels = torch.hstack((labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]]))
        if not isinstance(logits, list):
            labels = labels.reshape(-1)
            logits = logits.reshape(-1, logits.size(-1))

        # Compute loss
        loss = self._loss_fn(logits, labels)

        del logits  # free logits otherwise it peaks backward memory

        return loss

    def train(self) -> None:
        if self._compile:
            LOGGER.info("torch.compile is enabled. Expect a slow first iteration; model is compiled in first forward.")

        # zero gradients before starting training
        if not self._optimizer_in_bwd:
            self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        self._profiler.start()

        for curr_epoch in range(self.epochs_run, self.total_epochs):  # TODO refactor to while? (break out inner)
            self._sampler.set_epoch(curr_epoch)  # correctly shuffle data across epochs if shuffle=True

            for idx, batch in tqdm(enumerate(self._dataloader), total=self._steps_per_epoch):  # TODO make function?
                # Start tracking CUDA memory for active steps for just the first epoch
                # TODO Won't this break when resuming from checkpoints with epochs_run > 0?
                if (
                    curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                ):
                    torch.cuda.memory._record_memory_history()

                utils.batch_to_device(batch, self._device)
                num_tokens += batch["tokens"].numel()  # TODO if not packed, surely this is meaningless (pad tokens)

                loss = self._loss_step(batch)
                loss = loss / self._gradient_accumulation_steps
                running_loss += loss
                loss.backward()

                # Optimizer step (on reaching effective batch size)
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    if self._clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(), max_norm=float(self._clip_grad_norm)
                        )
                    if not self._optimizer_in_bwd:
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

                    # Need to fix `lr_scheduler.step()` before `optimizer.step()` warning
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    self.global_step += 1

                    loss_to_log = running_loss.item()

                    LOGGER.info(
                        f"Epoch {curr_epoch + 1:03d} | "
                        f"Global Step {self.global_step:0{len(str(self._steps_per_epoch))}d} | "
                        f"Loss: {loss_to_log:.4f}"
                    )

                    # Log per-step metrics
                    if self.global_step % self._log_every_n_steps == 0:
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            # NOTE optimizer_in_bwd assumes all optimizers have same LR; currently can't config diff LRs
                            "lr": (
                                self._optim_ckpt_wrapper.get_optim_key("lr")
                                if self._optimizer_in_bwd
                                else self._optimizer.param_groups[0]["lr"]
                            ),
                            "tokens_per_second_per_gpu": num_tokens / time_per_step,
                        }
                        if self._device.type == "cuda" and self._log_peak_memory_stats:
                            log_dict.update(training.get_memory_stats(device=self._device))
                        if self._clip_grad_norm is not None:
                            log_dict.update({"grad_norm": grad_norm})
                        self._metric_logger.log_dict(log_dict, step=self.global_step)

                    # Save checkpoint
                    if self.global_step != 0 and self.global_step % self.save_steps == 0:
                        self.save_checkpoint(epoch=curr_epoch)
                        LOGGER.info(f"Checkpoint saved at step {self.global_step:0{len(str(self._steps_per_epoch))}d}")

                    # Reset running stats for the next step
                    running_loss = 0
                    num_tokens = 0
                    t0 = time.perf_counter()

                # Stop tracking CUDA memory now that active steps are complete
                if (
                    curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps + self.profiler_active_steps
                ):
                    torch.cuda.memory._record_memory_history(enabled=None)

                # Step the profiler
                # Note we are stepping each batch, which might not include optimizer step in the trace
                # if the schedule cycle doesn't align with gradient accumulation.
                self._profiler.step()

            self.epochs_run += 1  # TODO FIXME set on basis of ckpt global step

        self._profiler.stop()

    def cleanup(self) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="FullFinetuneRecipeSingleDevice", cfg=cfg)
    recipe = FullFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
