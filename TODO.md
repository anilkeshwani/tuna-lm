# To Do

**Llama 3.2 1B Continued Pre-Training (CPT)**
- [ ] Set optimal batch size with gradient accumulation based on Llama 3.2 technical report
    - not mentioned - also Llama 3.2 1B doesn't have report/paper
- [ ] **Re-run run-001-interleaved-only (interleaved MLS)**
- [ ] Write a monkey patch for FullModelHFCheckpointer 🙈🩹
- [ ] **Run run-002-ASR-style-only (MLS ASR basic prompt template)**
- [ ] Check tokenization valid
- [ ] Implement validation loop
- [ ] Implement stopping based on max iterations
- [ ] Update to latest torchtune (some LR impl? What about max steps stopping?)
