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