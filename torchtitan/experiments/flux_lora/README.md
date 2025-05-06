# Background (replicate)
- I copied the torchtitan/experiments/flux folder and made a new torchtitan/experiments/flux_lora folder, and made the necessary changes to imports to make it work. Working out of the flux_lora folder will allow us to strip out everything that we don't need, while still being able to use the flux folder for reference, and merging improvements since this is being actively developed
- recommend setting files.exclude in vscode to ignore the flux folder, so that you are always opening files from the flux_lora folder
- plan is to strip out everything that we don't need, add support for lora training, and then tune it to make it as fast as possible.

# TODO
### setup

**goal**: Turn this into a working LoRA trainer

- [x] initialize weights from a checkpoint
- [x] make eval work
- [x] make sure eval looks sensible and can produce seeded images
- [ ] add support for the standard zipfile dataset format as currently used by LoRA trainer, remove any dataset related code that is no longer necessary. train for a few epochs
- [ ] add LoRA layers to the new model, turn off gradient for all layers except the LoRA layers, and train for a few epochs
- [ ] train a working LoRA model on the zeke2.zip dataset, make sure quality is up to par with what we expect
- [ ] train a working LoRA model on the zeke2.zip dataset on n>1 GPUs, make sure quality matches single GPU training

### Performance tuning of training loop

**goal**: Make 1000 training steps happen on 8xH100s in <1 minute, and produce a high quality LoRA

- try different parallelism strategies (tensor parallelism, data parallelism, fully sharded data parallelism, etc)
- try different optimizers/learning rates/warmup strategies, batch sizes, etc
- try face cropping the inputs
...

### misc notes about the code base
- this repo contains sophisticated checkpointing functionality, we dont want to use this. The only checkpoint we should be saving is the LoRA weights at the end of training.

# FLUX_LORA model in torchtitan

## Overview
This directory contains the implementation of the [FLUX](https://github.com/black-forest-labs/flux/tree/main) model in torchtitan, adapted for LoRA fine-tuning. In torchtitan, we showcase the pre-training process of text-to-image part of the FLUX model, and this directory focuses on its LoRA adaptation.

## Usage
First, download the autoencoder model from HuggingFace with your own access token:
```bash
python torchtitan/experiments/flux_lora/scripts/download_autoencoder.py --repo_id black-forest-labs/FLUX.1-dev --ae_path ae.safetensors --hf_token <your_access_token>
```

This step will download the autoencoder model from HuggingFace and save it to the `torchtitan/experiments/flux_lora/assets/autoencoder/ae.safetensors` file.

Run the following command to train the model on a single GPU:
```bash
./torchtitan/experiments/flux_lora/run_train.sh

```

## Supported Features
- Parallelism: The model supports FSDP, HSDP for training on multiple GPUs.
- Activation checkpointing: The model uses activation checkpointing to reduce memory usage during training.


## TODO
- [ ] More parallesim support (Tensor Parallelism, Context Parallelism, etc)
- [ ] Support for distributed checkpointing and loading
- [ ] Implement the num_flops_per_token calculation in get_nparams_and_flops() function
- [ ] Implement test cases in CI for FLUX LoRA model. Adding more unit tests for FLUX LoRA model (eg, unit test for preprocessor, etc)
