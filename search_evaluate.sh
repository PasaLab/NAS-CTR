#!/usr/bin/env bash

dataset_path=''

CUDA_VISIBLE_DEVICES=$1 python codes/search_evaluate.py --use_gpu=True --gpu=$1 --dataset_path=${dataset_path}
