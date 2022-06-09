#!/bin/bash

python main.py --model INPAINTNET --dataset jsb_chorales --gpu_id 0 --train True --repeat 1000
python main.py --model SKETCHNET --dataset jsb_chorales --gpu_id 0 --train True --train_vae True --repeat 1000
python main.py --model GRU_VAE --dataset jsb_chorales --gpu_id 0 --train True --train_vae True --repeat 1000
