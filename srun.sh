#!/usr/bin/env sh

set -x

PARTITION=$1
GPU=$2
CPU=16
PY_ARGS=${@:3}

srun --partition $PARTITION \
     --job-name ghfeat \
     --ntasks 1 \
     --cpus-per-task $CPU \
     --gres gpu:$GPU \
     ${PY_ARGS}
