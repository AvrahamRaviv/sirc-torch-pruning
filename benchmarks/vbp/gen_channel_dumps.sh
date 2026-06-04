#!/usr/bin/env bash
# Generate the channel_scores matrix for the ExpHandler "Channels" viz.
#
# 2 models x {global,local} x {mean,none} x 4 criteria  ->  32 JSON files in $OUT.
# Uses real ImageNet-1k. torchvision pretrained weights ARE imagenet1k; pass a cluster
# checkpoint via RN50_CKPT / CNX_CKPT to override. No training (epochs 0) — dumps only,
# so it is fast (just sigma calibration + score + prune-mask per config).
#
# Usage:
#   DATA=/algo/.../imagenet OUT=/algo/.../NORMNET/channel_dumps bash benchmarks/vbp/gen_channel_dumps.sh
# Optional env:
#   RN50_CKPT=/algo/.../resnet50_imagenet1k.pth   (else torchvision pretrained)
#   CNX_CKPT=/algo/.../convnext_tiny.pth
#   CALIB=40   RATIO=0.5
#
# DATA must be an ImageFolder root holding  val/<wnid>/*.JPEG  (train/ optional, used for calib).
set -euo pipefail

DATA="${DATA:?set DATA=<imagenet ImageFolder root containing val/>}"
OUT="${OUT:-./channel_dumps}"
CALIB="${CALIB:-40}"
RATIO="${RATIO:-0.5}"
PY="python -u benchmarks/vbp/normnet_standalone.py"
COMMON="--dataset imagenet --data_path $DATA --batch_size 128 --epochs 0 --epochs_ft 0 --calib_batches $CALIB --pruning_ratio $RATIO --dump_scores --save_dir $OUT --limit_batches 2 --modes magnitude,nci,rel,nonrel"

mkdir -p "$OUT"

run_model () {   # $1 model   $2 tag   $3 ckpt_arg
  local M=$1 TAG=$2 CK=$3
  for G in "--global_prune" "--local"; do
    for N in mean none; do
      echo ">>> $M $G normalizer=$N"
      $PY --model "$M" --tag "$TAG" $CK $G --normalizer "$N" $COMMON
    done
  done
}

RN_CK=""; [ -n "${RN50_CKPT:-}" ] && RN_CK="--ckpt $RN50_CKPT"
CN_CK=""; [ -n "${CNX_CKPT:-}" ]  && CN_CK="--ckpt $CNX_CKPT"

run_model resnet50      imagenet_rn50 "$RN_CK"
run_model convnext_tiny imagenet_cnx  "$CN_CK"

echo "DONE -> $OUT ($(ls "$OUT"/*_channel_scores.json | wc -l) files)"
