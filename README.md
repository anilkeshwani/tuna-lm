# tuna-lm: Continued Pre-training & Fine-tuning with torchtune

Code is based on torchtune recipe and config for full fine-tuning of Llama 3.2 1B (and other) models.

- _full_finetune_single_device.py_ is the torchtune "recipe" for full model, single GPU fine-tuning
- _configs/_ directory contains configuration YAML files (configuration is Hydra style, basically a minimal impl. inspired by Hydra)

## Setup

```bash
git submodule update --init --recursive &&
    conda create -y --name tunalm python=3.10.6 &&
    conda activate tunalm &&
    pip install -r requirements.txt
```

## Train

```bash
tune run tunalm/full_finetune_single_device.py \
    --config tunalm/configs/1B_full_single_device.yaml
```

Run the above from the project root _without_ a leading `./` which induces torchtune to raise an `ImportError: Relative module names not supported`.
