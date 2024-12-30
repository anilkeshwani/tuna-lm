# Changelog (salient changes only)

- Between generation recipe provided by torchtune as my inference.py, I removed quantization
    - see e.g. the changes in commit 23080203d6d4598291b6b0254cfc42a2533c5632


## Tokenizer Extension & Hugging Face Interoperability

### Minimal requirements for generation

- [x] config.json -> /mnt/scratch-artemis/anilkeshwani/huggingface/hub/models--meta-llama--Llama-3.2-1B/blobs/83b8b2aebb1a987b3802dae75fb9470234a3aaaf
- [x] model.safetensors -> /mnt/scratch-artemis/anilkeshwani/huggingface/hub/models--meta-llama--Llama-3.2-1B/blobs/68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a
- [ ] tokenizer_config.json -> /mnt/scratch-artemis/anilkeshwani/huggingface/hub/models--meta-llama--Llama-3.2-1B/blobs/cb9ec25536e44d86778b10509d3e5bdca459a5cf
    - need to added_tokens_decoder key in JSON and increment all entries by the number of additional tokens added
    - TODO why wasn't this a problem at training time?
- [ ] tokenizer.json -> /mnt/scratch-artemis/anilkeshwani/huggingface/hub/models--meta-llama--Llama-3.2-1B/blobs/5cc5f00a5b203e90a27a3bd60d1ec393b07971e8
    - the added_tokens_decoder field from tokenizer_config.json slots in here as the added_tokens field (top-level)

#### Not required 

Note: We don't need the generation config either - assuming I can pass those setting at runtime.

- generation_config.json -> /mnt/scratch-artemis/anilkeshwani/huggingface/hub/models--meta-llama--Llama-3.2-1B/blobs/2d73a6863086ff9d491c28e49df9fb697cd92c2b
- special_tokens_map.json -> /mnt/scratch-artemis/anilkeshwani/huggingface/hub/models--meta-llama--Llama-3.2-1B/blobs/cfabacc2620186cd3dd4b1dde9a37e057208636e

Converting the Hugging Face tokenizer files for use with vLLM was hairy; lack of documentation on which fields were used/updated when extending the model. 

### To Do - Tokenizer Extension & Hugging Face Interoperability

- Check the source code of the HF add tokens function to see which attributes are modified (requires tracking downstream serialisation of these attributes). 
- Try this implemented by directly using HF's `add_tokens` function, per our previous approach. 

## Configurations

Minimal notes on configurations, retained here for reference and often taken from YAML file comment headers. 

### 1B_full_single_device.yaml

Config for single device full finetuning in full_finetune_single_device.py using a Llama3.2 1B (Instruct) model.

The default config from torchtune uses an optimizer from bitsandbytes. Install it with: `pip install bitsandbytes`

## Tokenizer Extension

### Investigations

Removed all json and yaml config files from Llama 3 1B download and only config.json required when running `tune run`.



### Notes

Need to update:
- config.json : vocab_size
- original/params.json : vocab_size

Depending on where the new tokens go:
generation_config.json:
- "bos_token_id": 128000,
- "eos_token_id": 128001,

These positions might be "moved along" to the new indices if we insert the DSU tokens before them.

Note: original/param.json `multiple_of` parameter. What is this for?

```json
"added_tokens_decoder": {
    "128000": {
      "content": "<|begin_of_text|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    ...
```

How to treat the tokenizer.json file?

### Model embeddings size and tokenizer vocab size

- recipe._model.tok_embeddings.weight.data.size()
    - torch.Size([128256, 2048])
- recipe._tokenizer.tt_model.base_vocab_size
    - 128000
- recipe._tokenizer.tt_model.vocab_size
    - 128257