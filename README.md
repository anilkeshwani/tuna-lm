# tuna-lm 

Recipes for fine-tuning and continued pre-training for (L)LMs based on [torchtune](https://pytorch.org/torchtune/stable/overview.html), written whilst at the 🐟 Sardine Lab.

# Setup

```bash
git submodule update --init --recursive &&
    conda create -y --name tunalm python=3.10.6 &&
    conda activate tunalm &&
    pip install -e ./submodules/torchtune/ &&
    pip install -e .["dev"]
```

> [!NOTE] 
> torchtune (local version) is not referenced by requirements.txt. It is not installable via `pip install -e .` despite being installable via `pip install -r requirements.txt`.

# Train

_Draft training signature example_

```bash
./tunalm/train.py \
    add_n_dsus=$add_n_dsus \
    iterations=$iterations \
    train_dataset=$train_dataset \
    eval_dataset=$eval_dataset \
    checkpoint=$checkpoint
```

- Should we also depend on sardalign for import of e.g. DSU PUA offsets?

# Test

Test using existing evaluation code (vLLM + 🤗 evaluate)

---

> I'm Sardine to think that fine-tuna-ing your language models is the deepest way to scale your learning!
