#!/bin/bash
export LANG="en_US.UTF-8"

ROOT_DIR=../GAS
export PYTHONPATH=$ROOT_DIR

for seed in 10 15 20
do
    python $ROOT_DIR/main.py --seed $seed \
                   --task asqp \
                   --dataset rest15 \
                   --alpha 0.5 \
                   --beta 4e-2
done
