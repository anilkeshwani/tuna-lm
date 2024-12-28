# To Do

- [ ] BUG: Re-run CPT "finetune.py" recipes without using add_eos: True - this shouldn't be appended for CPT (where we want continuations)
- [ ] BUG: Save a symlink of the tokenizer, the config.json and the tokennizer_config.json at _every_ checkpoint event (in every checkpoint dir)
    - allows us to use the checkpointer exposed method interfaces without manual hacks

No textual data used for CPT. Options for CPT:

- mix text and ASR-style data
    - caveat: Llama 3.2 1B is trained from distillation - maybe next token prediction is not well suited - we should decide this empirically
- use checkpoint averaging
    - there is some literature to back up that all you need to prevent catastrophic forgetting is checkpoint averaging
    - find this literature in Obsidian vault

**Llama 3.2 1B Continued Pre-Training (CPT)**
- [x] Set optimal batch size with gradient accumulation based on Llama 3.2 technical report
    - not mentioned - also Llama 3.2 1B doesn't have report/paper
- [x] Write a monkey patch for FullModelHFCheckpointer 🙈🩹
    - not needed - HACK via change of _output_dir
- [x] Implement validation loop
- [ ] **Re-run run-001-interleaved-only (interleaved MLS)**
- [ ] **Run run-002-ASR-style-only (MLS ASR basic prompt template)**
- [ ] Check tokenization valid
- [ ] Implement stopping based on max iterations
- [ ] Optional: Update to latest torchtune (some LR impl? What about max steps stopping?)
- [ ] Add the MMS normalizer as well as the Whisper normalizer to the WER evaluation function or module

## TODO Recipe

- [x] max_steps_per_epoch -  check all gone
- [x] resolve max_steps by epoch or epoch by max_steps - resolver method
- [x] add hack to set checkpointer output_dir in save_checkpoint to specify output dir
- [x] handle the intermediate checkpoint variable on the basis of self.global_step vs max_steps
- [x] Add evaluation loop every eval_steps iterations
- [x] create MLS validation set (both interleaved)
- [ ] Add support for max iterations
- [ ] write ASR concat style collator
- [ ] compute ASR on validation as well - HF evaluate with normalizations
- [ ] fix out dir vs ckpt dir - config JSON is written to  checkpoints/ dir
- [ ] maybe save cfg and an instance variable?
- [ ] future: do we want to save only one recipe state; maybe for now we can remove the previous one?
              relatively easy to incorporate into save_checkpoint method - track previous out dir and
              remove recipe.pt
