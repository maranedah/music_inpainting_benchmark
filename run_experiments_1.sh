#!/bin/bash

python main.py --model ARNN --dataset folk --gpu_id 1 --train True --repeat 10 --patience 10
python main.py --model INPAINTNET --dataset folk --gpu_id 1 --train True --train_vae True --repeat 10 --patience 10
python main.py --model SKETCHNET --dataset folk --gpu_id 1 --train True --train_vae True --repeat 10 --patience 10
python main.py --model GRU_VAE --dataset folk --gpu_id 1 --train True --train_vae True --repeat 10 --patience 10
