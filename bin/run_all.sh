#!/usr/bin/env bash

for DATASET in dunnhumby instacart tafeng
do
    for MODEL in top_personal top_popular up_cf tifuknn tifuknn_time_days tifuknn_time_days_next_ts
    do
        OPENBLAS_NUM_THREADS=4 PYTHONPATH=. python src/scripts/experiment.py \
            --dataset=$DATASET \
            --model=$MODEL \
            --num-trials=300 \
            --batch-size=20000 &
    done
done
