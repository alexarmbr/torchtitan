#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example command: 
# LOG_RANK=0,1 NGPU=4 ./torchtitan/experiments/flux_lora/run_train.sh

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./torchtitan/experiments/flux/run_train.sh
NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/experiments/flux_lora/train_configs/flux-lora_dev_model.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi


PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m torchtitan.experiments.flux_lora.train --job.config_file ${CONFIG_FILE} $overrides
