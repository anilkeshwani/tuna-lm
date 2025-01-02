# Debugging README

## Getting inference working as expected

Minimum _working_ example. This works:

```
tune run generate --config ./working_generation_config.yaml
```

Override the prompt specified in the config:

```
tune run generate --config ./working_generation_config.yaml \
prompt="Here's a funny joke"
```

## My inference script ("recipe")

Check our inference impl with the OOTB recipe and base model.

```bash
tune run tunalm/modern_inference.py \
    --config /mnt/scratch-artemis/anilkeshwani/tmp/debug_tt_inference/custom_generation_config.yaml \
    prompt="What are some interesting sites to visit in the Bay Area?"
```
