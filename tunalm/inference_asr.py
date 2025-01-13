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

        print(f"{self.tokenizer.stop_tokens = }")
        print(f"{[self.tokenizer.eos_id] = }")
        breakpoint()

        # tokens = <FIX ME> # self.convert_prompt_to_tokens(cfg.prompt)
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
    def inference_batched():
        for i, batch in tqdm(enumerate(self.data_train), total=self.steps_per_epoch):  # TODO make function?
            # Start tracking CUDA memory for active steps for just the first epoch
            # TODO Won't this break when resuming from checkpoints with epochs_run > 0?
            if (
                curr_epoch == 0
                and self.profiler_profile_memory
                and i == self.profiler_wait_steps + self.profiler_warmup_steps
            ):
                torch.cuda.memory._record_memory_history()

            utils.batch_to_device(batch, self.device)
            num_tokens += batch["tokens"].numel()  # TODO if not packed, surely this is meaningless (pad tokens)
            # num_tokens += (batch["tokens"] != self.tokenizer.pad_id).sum().item()  # TODO think it should be this

            loss = self.loss_step(batch)
            loss = loss / self.gradient_accumulation_steps
            running_loss += loss
            loss.backward()

            # Optimizer step (on reaching effective batch size)
            if (i + 1) % self.gradient_accumulation_steps == 0:
                if self.clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=float(self.clip_grad_norm)
                    )
                if not self.optimizer_in_bwd:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # Need to fix `lr_scheduler.step()` before `optimizer.step()` warning
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                self.global_step += 1

                loss_to_log = running_loss.item()

                LOGGER.info(
                    f"Epoch {curr_epoch + 1:03d} | "
                    f"Iter {i:0{len(str(self.steps_per_epoch))}d} / {self.steps_per_epoch} | "
                    f"Global Step {self.global_step:0{len(str(self.steps_per_epoch))}d} | "
                    f"Loss: {loss_to_log:.4f}"
                )

                # Evaluate model on dev set
                if self.global_step != 0 and self.global_step % self.eval_steps == 0:
                    dev_loss = 0
                    self.model.eval()
                    with torch.inference_mode():
                        for i_dev, dev_batch in enumerate(self.data_dev):
                            utils.batch_to_device(dev_batch, self.device)
                            dev_loss_batch = self.loss_step(dev_batch)
                            LOGGER.info(
                                f"Epoch {curr_epoch + 1:03d} | "
                                f"Iter {i:0{len(str(self.steps_per_epoch))}d} / {self.steps_per_epoch} | "
                                f"Global Step {self.global_step:0{len(str(self.steps_per_epoch))}d} | "
                                f"Dev Batch {i_dev:0{len(str(len(self.data_dev)))}d} / {len(self.data_dev)} | "
                                f"Dev Loss (batch): {dev_loss_batch.item():.4f}"
                            )
                            dev_loss += dev_loss_batch.item()  # reduction is mean over non-ignored label elements
                    self.model.train()
                else:
                    dev_loss = None  # did not evaluate at this step

                # Log per-step metrics
                if self.global_step % self.log_every_n_steps == 0:
                    time_per_step = time.perf_counter() - t0
                    log_dict = {
                        "loss": loss_to_log,
                        # NOTE optimizer_in_bwd assumes all optimizers have same LR; currently can't config diff LRs
                        "lr": (
                            self.optim_ckpt_wrapper.get_optim_key("lr")
                            if self.optimizer_in_bwd
                            else self.optimizer.param_groups[0]["lr"]
                        ),
                        "tokens_per_second_per_gpu": num_tokens / time_per_step,
                    }
                    if self.device.type == "cuda" and self.log_peak_memory_stats:
                        log_dict.update(training.get_memory_stats(device=self.device))
                    if self.clip_grad_norm is not None:
                        log_dict.update({"grad_norm": grad_norm})
                    if dev_loss is not None:
                        log_dict.update({"dev_loss": dev_loss})
                    self.metric_logger.log_dict(log_dict, step=self.global_step)

                # Save checkpoint
                if self.global_step != 0 and self.global_step % self.save_steps == 0:
                    self.save_checkpoint(epoch=curr_epoch)
                    LOGGER.info(f"Checkpoint saved at step {self.global_step:0{len(str(self.steps_per_epoch))}d}")

                # Reset running stats for the next step
                running_loss = 0
                num_tokens = 0
                t0 = time.perf_counter()

            # Stop tracking CUDA memory now that active steps are complete
            if (
                curr_epoch == 0
                and self.profiler_profile_memory
                and i == self.profiler_wait_steps + self.profiler_warmup_steps + self.profiler_active_steps
            ):
                torch.cuda.memory._record_memory_history(enabled=None)

            # Step the profiler
            # Note we are stepping each batch, which might not include optimizer step in the trace
            # if the schedule cycle doesn't align with gradient accumulation.
            self.profiler.step()

    @torch.inference_mode()
    def or_should_this_be_the_inference_method(
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
