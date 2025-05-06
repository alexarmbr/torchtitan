# Flux LoRA Training
- torchtitan/experiments/flux_lora is being developed for optimized, parallelized lora training.

### setup
- the .devcontainer/Dockerfile builds an environment which should be able to run this code.

### getting started
- run the following command
```bash
CUDA_VISIBLE_DEVICES=0 NGPU=1 ./torchtitan/experiments/flux_lora/run_train.sh
```
this will use the `flux-lora_dev_model.toml` config file, which is currently set to run 11 training steps, and run an eval every 2 steps.
currently eval consists of generating a single cat image and saving it to `outputs/img/image_rank<rank>_<step>.png`. Ensure you can run these
training loops and that the images look sensible. The first time you initialize the model, a flux dev checkpoint will be downloaded from huggingface
and saved to the huggingface home directory (set HF_HOME to configure this)


### suggestions
- make sure you are always working with files in the `flux_lora` directory, not the `flux` directory. I set vscode files.exclude to ignore the `flux` directory
- keep an eye out for comments tagged IMPORTANT(replicate), these are notes about important details

# TODO
**goal #1**: Turn this into a working LoRA trainer

- [x] initialize weights from a checkpoint
- [x] make eval work
- [x] make sure eval looks sensible and can produce seeded images
- [ ] add support for the standard zipfile dataset format as currently used by LoRA trainer, remove any dataset related code that is no longer necessary. train for a few epochs
- [ ] add LoRA layers to the new model, turn off gradient for all layers except the LoRA layers, and train for a few epochs
- [ ] train a working LoRA model on the zeke2.zip dataset, make sure quality is up to par with what we expect
- [ ] train a working LoRA model on the zeke2.zip dataset on n>1 GPUs, make sure quality matches single GPU training

**goal #2**: Make 1000 training steps happen on 8xH100s in <1 minute, and produce a high quality LoRA
- [ ] come up with a set of TODOs that will get us to this goal

**goal #3**: tbd, but probably something to do with turning this into a deployable system.



# original README:

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
