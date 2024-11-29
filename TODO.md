# To Do

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
