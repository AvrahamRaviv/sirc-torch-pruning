"""Apply a dumped prune mask (JSON pruning_history) to a FRESH model and evaluate.

Decouples the prune MASK from the scoring/calibration pipeline: a mask dumped on machine A
(normnet_main.py --dump_prune → <path>.json) can be replayed EXACTLY on machine B's model and
evaluated on B's val set. This isolates MASK-divergence from CALIB/EVAL-divergence in the
cross-platform debug — same mask, two machines, compare pre-FT acc.

The JSON holds TP's canonical replay format: {"pruning_history": [[layer, is_out_channel, idxs],
...], ...}. DG.load_pruning_history re-derives all coupled in-channel/BN prunes on the fresh
model from the recorded ROOT out-channel ops, so out-channel idxs alone fully reconstruct the
pruned net (no fold/reinsert needed — TP prunes the torchvision BN via coupling).

Example:
  python apply_prune.py --model_type cnn --cnn_arch mobilenet_v2 \
    --model_name /path/mobilenet_v2_weights.pth --data_path /path/imagenet \
    --prune_json /path/champion_mnv2_os_cov_val_m21.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))           # benchmarks/vbp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

import torch
import torch_pruning as tp
from vbp_common import load_model, build_dataloaders, validate


def main():
    p = argparse.ArgumentParser(description="apply a dumped prune mask (JSON) + evaluate")
    p.add_argument("--prune_json", required=True, help="pruning_history JSON from --dump_prune")
    p.add_argument("--model_type", default="cnn", choices=["cnn", "convnext", "vit"])
    p.add_argument("--cnn_arch", default="mobilenet_v2")
    p.add_argument("--model_name", required=True, help="dense weights (.pth) for the base arch")
    p.add_argument("--data_path", required=True, help="ImageNet root (with val_samples.pkl or val/)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--val_resize", type=int, default=232)
    p.add_argument("--val_batch_size", type=int, default=256)
    p.add_argument("--train_batch_size", type=int, default=128)   # unused, build_dataloaders needs it
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[apply_prune] device={device}")

    # 1. fresh dense model
    model = load_model(args, device)
    model.eval()
    ex = torch.randn(1, 3, 224, 224).to(device)
    macs0, params0 = tp.utils.count_ops_and_params(model, ex)

    # 2. load mask + replay on the fresh model's dependency graph
    with open(args.prune_json) as f:
        meta = json.load(f)
    hist_raw = meta["pruning_history"]
    history = [(n, bool(is_out), [int(i) for i in idxs]) for n, is_out, idxs in hist_raw]
    print(f"[apply_prune] loaded {len(history)} prune ops from {args.prune_json}")
    print(f"[apply_prune] meta: mac_target_g={meta.get('mac_target_g')} "
          f"ratio={meta.get('ratio')} normalizer={meta.get('imp_normalizer')}")

    DG = tp.DependencyGraph().build_dependency(model, example_inputs=ex)
    DG.load_pruning_history(history)
    model.to(device)

    macs1, params1 = tp.utils.count_ops_and_params(model, ex)
    print(f"[apply_prune] pruned: {macs0/1e9:.3f}G→{macs1/1e9:.3f}G MACs "
          f"({100*macs1/macs0:.0f}%), {params0/1e6:.2f}M→{params1/1e6:.2f}M params")

    # 3. evaluate
    _, val_loader, _ = build_dataloaders(args, use_ddp=False)
    acc, loss = validate(model, val_loader, device, args.model_type)
    print(f"[apply_prune] STRUCTURAL-MASK pre-FT val acc = {acc:.4f}  (loss {loss:.4f})  "
          f"{macs1/1e9:.2f}G {params1/1e6:.2f}M")
    print("[apply_prune] NOTE: this replays only the STRUCTURAL mask (channels removed). It does "
          "NOT apply --bias_comp, which for MobileNetV2 is load-bearing (WITH≈0.28 vs WITHOUT≈0.002 "
          "dead) and is calib-derived (not encoded in the mask). For an EXACT pre-FT number, load "
          "the full pruned+compensated model saved by normnet_main at <save_dir>/<save_tag>.pth, "
          "not this structural replay.")


if __name__ == "__main__":
    main()
