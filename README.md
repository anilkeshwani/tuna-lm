# tuna-lm: Continued Pre-training & Fine-tuning with torchtune

## Setup

```bash
git submodule update --init --recursive &&
    conda create -y --name tunalm python=3.10.6 &&
    conda activate tunalm &&
    pip install -e .["dev"] &&
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
base_models_dir=/mnt/scratch-artemis/anilkeshwani/models/base/ &&
tune download meta-llama/Llama-3.2-1B \
    --output-dir ${base_models_dir}/Llama-3.2-1B \
    --ignore-patterns "original/consolidated.00.pth"
```

<details>
    <summary>Download terminal output</summary>
    ```
    Ignoring files matching the following patterns: original/consolidated.00.pth
    LICENSE.txt: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7.71k/7.71k [00:00<00:00, 2.99MB/s]
    original/params.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 220/220 [00:00<00:00, 2.06MB/s]
    USE_POLICY.md: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6.02k/6.02k [00:00<00:00, 38.1MB/s]
    README.md: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41.2k/41.2k [00:00<00:00, 13.4MB/s]
    .gitattributes: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.52k/1.52k [00:00<00:00, 14.1MB/s]
    tokenizer.model: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.18M/2.18M [00:00<00:00, 25.0MB/s]
    Fetching 12 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:02<00:00,  4.76it/s]
    Successfully downloaded model repo and wrote to the following locations:
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/.gitattributes
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/config.json
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/LICENSE.txt
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/tokenizer_config.json
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/tokenizer.json
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/original
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/.cache
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/README.md
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/generation_config.json
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/model.safetensors
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/USE_POLICY.md
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/special_tokens_map.json
    ```
</details>

## Extend Llama 3.2 Base Model

The following command will extend the base model with the specified number of tokens and save the extended model to the specified output directory:

```bash
./extendllama3.py --n_new_dsus 5000
```

## Train

Code is based on a torchtune recipe and config for  for full model, single GPU fine-tuning of the Llama 3.2 1B model.

```bash
tune run tunalm/finetune.py --config tunalm/configs/1B_finetune.yaml
```

Run the above from the project root _without_ a leading `./` which induces torchtune to raise an `ImportError: Relative module names not supported`.

Overrides are via OmegaConf/Hydra syntax through the command line. For example:

```bash
tune run full_finetune_single_device \
    --config llama3_2/1B_full_single_device \
    checkpointer.checkpoint_dir="${YOUR_CHECKPOINT_DIR}"
```

> [!NOTE]  
> This config works only for training on single device.

## Inference

Inference runs via two scripts:
1. generate_sample.py: Manual or semi-interactive generation for a single sample. At the time of writing, this is effectively a continuation
2. inference.py: Batch, non-interactive generation from a dataset. Used for systematic evaluation

### Single Sample Generation via generate_sample.py

Example calls to generate continuations from the extended Llama 3.2 1B model with continued pre-training:

```
tune run tunalm/generate_sample.py --config tunalm/configs_inference/extended_cptd.yaml checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus/playful-morning-102-id_rq5tmfca/checkpoints/global-step-000382'
```

Another checkpoint:

```
tune run tunalm/generate_sample.py --config tunalm/configs_inference/extended_cptd.yaml checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus/playful-morning-102-id_rq5tmfca/checkpoints/global-step-053862/' checkpointer.checkpoint_files='[hf_model_0001_1.pt]'
```

### Batch Inference via inference.py

Passes a dataset to the `setup_test_data` method which in turn is passed to the `inference` method, which performs batch generation. 

#### Example calls

CPT-only ckpt:

```bash
tune run tunalm/inference_asr.py \
    --config tunalm/configs_inference/inference_asr.yaml \
    checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus/playful-morning-102-id_rq5tmfca/checkpoints/global-step-000382' \
    output_dir='./outputs/Llama-3.2-1B-5000-dsus/playful-morning-102-id_rq5tmfca/checkpoints/global-step-000382'
```

SFT-only ckpt:

```bash
tune run tunalm/inference_asr.py \
    --config tunalm/configs_inference/inference_asr.yaml \
    checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/avid-wind-123-id_xoafid42/checkpoints/global-step-006500' \
    output_dir='outputs/Llama-3.2-1B-5000-dsus-sft/avid-wind-123-id_xoafid42/checkpoints/global-step-006500'
```
