#! bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 train.py -c config.toml
# ps -aux | grep khanhnnm | grep cmgan | awk '{print $2}' | xargs kill