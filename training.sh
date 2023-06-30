#!/bin/bash

cd ~/Documents/Code/Stable-Diffusion-Inpaint
PY_EXE=/home/richardbrown/Documents/Code/miniconda3/envs/ldm/bin/python

$PY_EXE main_inpainting.py \
  --train \
  --name "500ddim" \
  --base "inpainting_overfit_512.yaml" \
  --gpus 1 \
  --seed 42
