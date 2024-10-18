# tuna-lm: Continued Pre-training & Fine-tuning with torchtune

## Setup

```bash
git submodule update --init --recursive &&
    conda create -y --name tunalm python=3.10.6 &&
    conda activate tunalm &&
    pip install -r requirements.txt &&
    pip install --no-deps -e ./submodules/speech-text-alignment
```

Note: We want to do a shallow install of sardalign so we can import constants (e.g. PUA offsets and tokens) and standard configuration.

## Download Llama 3.2 Base Model

The torchtune training recipe assumes that you've run the following command, substituting relevant variables from the configuration file values, in order to download the Llama 3.2 pre-trained (base) model:

``` bash
tune download meta-llama/${model_name} --output-dir ${base_models_dir}/${model_name} --ignore-patterns "original/consolidated.00.pth"
```

Typically:

```bash
base_models_dir=/mnt/scratch-artemis/anilkeshwani/models/base-torchtune/ &&
tune download meta-llama/Llama-3.2-1B \
    --output-dir ${base_models_dir}/Llama-3.2-1B \
    --ignore-patterns "original/consolidated.00.pth"
```

## Train

Code is based on a torchtune recipe and config for  for full model, single GPU fine-tuning of the Llama 3.2 1B model.

- **full_finetune_single_device.py**: torchtune "recipe"
- **configs/1B_full_single_device.yaml**: YAML configuration (Hydra-style)

```bash
tune run tunalm/full_finetune_single_device.py \
    --config tunalm/configs/1B_full_single_device.yaml
```

Run the above from the project root _without_ a leading `./` which induces torchtune to raise an `ImportError: Relative module names not supported`.

Overrides are via OmegaConf/Hydra syntax through the command line. For example:

```bash
tune run full_finetune_single_device \
    --config llama3_2/1B_full_single_device \
    checkpointer.checkpoint_dir="${YOUR_CHECKPOINT_DIR}"
```

> [!NOTE] This config works only for training on single device.
