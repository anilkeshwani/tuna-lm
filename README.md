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
base_models_dir=/mnt/scratch-artemis/anilkeshwani/models/base-torchtune/ &&
tune download meta-llama/Llama-3.2-1B \
    --output-dir ${base_models_dir}/Llama-3.2-1B \
    --ignore-patterns "original/consolidated.00.pth"
```

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

### Interactive or Manual Generation

Uses the `prompt` field in the inference YAML config to produce a prompt of type `dict[Role, str]` supplied to `convert_prompt_to_tokens` which is passed to the `generate` method

```yaml
prompt:
  system: null
  user: "Tell me a joke."
```

Example calls:

```bash
tune run tunalm/inference.py --config tunalm/configs_1B/inference.yaml \
  output_dir="./output_dir" \
  checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus/playful-morning-102-id_rq5tmfca/checkpoints/global-step-000382' \
  tokenizer.path='/mnt/scratch-artemis/anilkeshwani/models/extended/torchtune/Llama-3.2-1B-5000-dsus/original/tokenizer.model'
```

```bash
tune run tunalm/inference.py --config tunalm/configs_1B/inference.yaml \
  output_dir="./output_dir" \
  checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/models/base/torchtune/Llama-3.2-1B-reference' \
  tokenizer.path='/mnt/scratch-artemis/anilkeshwani/models/base/torchtune/Llama-3.2-1B-reference/original/tokenizer.model'
```

### Batch Inference

Passes a dataset to the `setup_test_data` method which in turn is passed to the `inference` method, which performs batch generation. 

Example calls:

```bash
...
```
