# Alex notes
- I copied the flux folder and made a new flux_lora folder, and made the necessary changes to imports to make it work
- recommend setting files.exclude in vscode to ignore the flux folder
- plan is to strip out everything that we don't need, and add support for lora training


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
