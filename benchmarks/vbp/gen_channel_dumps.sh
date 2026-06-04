#!/usr/bin/env bash
# Generate the channel_scores matrix for the ExpHandler "Channels" viz.
#
# 2 models x {global,local} x {mean,none} x 4 criteria  ->  32 JSON files in $OUT.
# Real ImageNet-1k via the cluster cached pickle (<DATA>/{train,val}_samples.pkl, same as
# train_v2 / vbp_common; ImageFolder val/ is the fallback). torchvision pretrained weights
# ARE imagenet1k; pass a cluster checkpoint via RN50_CKPT / CNX_CKPT to override.
# No training (epochs 0) -> dumps only: sigma calibration + score + prune-mask per config.
#
# Usage (cluster defaults baked in; run from repo root):
#   OUT=/algo/NetOptimization/outputs/NORMNET/channel_dumps bash benchmarks/vbp/gen_channel_dumps.sh
# Override:
#   DATA=/algo/NetOptimization/outputs/VBP RN50_CKPT=/algo/.../resnet50_imagenet1k.pth \
#   CALIB=50 RATIO=0.5 VAL_RESIZE=232 bash benchmarks/vbp/gen_channel_dumps.sh
set -euo pipefail

DATA="${DATA:-/algo/NetOptimization/outputs/VBP}"     # holds {train,val}_samples.pkl
OUT="${OUT:-./channel_dumps}"
CALIB="${CALIB:-50}"
RATIO="${RATIO:-0.5}"
VAL_RESIZE="${VAL_RESIZE:-256}"
PY="python -u benchmarks/vbp/normnet_standalone.py"
COMMON="--dataset imagenet --data_path $DATA --val_resize $VAL_RESIZE --batch_size 128 --epochs 0 --epochs_ft 0 --calib_batches $CALIB --pruning_ratio $RATIO --dump_scores --save_dir $OUT --limit_batches 2 --modes magnitude,nci,rel,nonrel"

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

# RN50 defaults to the cluster imagenet1k ckpt (as in generate_e2_experiments); ConvNeXt
# falls back to torchvision pretrained unless CNX_CKPT is set. Missing file -> drop the flag.
RN50_CKPT="${RN50_CKPT:-/algo/NetOptimization/outputs/VBP/ResNet50_TP/resnet50_imagenet1k.pth}"
RN_CK=""; [ -f "$RN50_CKPT" ] && RN_CK="--ckpt $RN50_CKPT"
CN_CK=""; [ -n "${CNX_CKPT:-}" ] && [ -f "$CNX_CKPT" ] && CN_CK="--ckpt $CNX_CKPT"

run_model resnet50      imagenet_rn50 "$RN_CK"
run_model convnext_tiny imagenet_cnx  "$CN_CK"

echo "DONE -> $OUT ($(ls "$OUT"/*_channel_scores.json | wc -l) files)"
