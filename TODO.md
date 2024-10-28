# To Do

## Tokenizer Extension

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

## General

**Llama 3.2 1B Continued Pre-Training (CPT)**
- [ ] Set optimal batch size with gradient accumulation based on Llama 3.2 technical report
