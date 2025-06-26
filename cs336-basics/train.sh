#!/usr/bin/env bash

uv run torchrun --standalone --nproc_per_node=4 \
    scripts/train.py --config-name=experiment/your_data \
    +training.save_checkpoints=True \
    +training.train_batch_size=32 \
    +training.gradient_accumulation_steps=2