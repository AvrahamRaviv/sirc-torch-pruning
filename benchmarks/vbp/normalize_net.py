"""
Normalize-net-by-construction: transform + verify (no training).

Applies a function-preserving reparametrization across the whole network: for every
nn.Linear and nn.Conv2d(groups==1), insert BN(affine=False) on its input and move the
trainable weight into normalized space (v_tilde = W·σ), folding the input mean into the
bias (m = b + W·μ). See torch_pruning/utils/reparam.py::NormalizedResidualManager.

This script only transforms and verifies — a general-training run is the next step and
resumes from the saved 'vnr' artifact. Two checkpoints are written:
  - <tag>_vnr.pth          : pre-merge (v_tilde / m / bn.*) — for training resume.
  - <tag>_merged_biased.pth: post-merge (standard weight / bias) — canonical, verified.
Each has a .meta.json sidecar so torch.load(weights_only=True) stays usable.

Usage (single device, MPS auto-selected on Apple Silicon):
    python benchmarks/vbp/normalize_net.py \
        --model_type convnext --model_name convnext_tiny \
        --checkpoint /path/to/convnext_tiny.pth \
        --data_path /path/to/imagenet --disable_ddp --num_workers 0
"""

import argparse
import json
import os
import sys

import torch

try:
    from .vbp_common import (
        logger, is_main, log_info, setup_logging,
        setup_distributed, cleanup,
        build_dataloaders, load_model, validate, forward_logits,
        build_whole_net_reparam_layers, attach_biases, _merge_vnr_state_dict,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vbp_common import (
        logger, is_main, log_info, setup_logging,
        setup_distributed, cleanup,
        build_dataloaders, load_model, validate, forward_logits,
        build_whole_net_reparam_layers, attach_biases, _merge_vnr_state_dict,
    )

from torch_pruning.utils.reparam import NormalizedResidualManager


def get_device():
    """Prefer CUDA, then MPS (Apple Silicon), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _broadcast_model_state(model):
    """Broadcast params and buffers from rank 0 after a structural change."""
    import torch.distributed as dist
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for buf in model.buffers():
        dist.broadcast(buf, src=0)


# ---------------------------------------------------------------------------
# Save / reload (raw state_dict + JSON sidecar)
# ---------------------------------------------------------------------------
def _ckpt_paths(args, fmt):
    base = os.path.join(args.save_dir, f"{args.save_tag}_{fmt}")
    return base + ".pth", base + ".meta.json"


def save_normnet_checkpoint(state_dict, fmt, biased_layers, args):
    """Save a raw state_dict + a .meta.json sidecar. Returns the .pth path."""
    os.makedirs(args.save_dir, exist_ok=True)
    pth_path, meta_path = _ckpt_paths(args, fmt)
    cpu_state = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
    torch.save(cpu_state, pth_path)
    meta = {
        "format": fmt,                       # "vnr" | "merged_biased"
        "biased_layers": list(biased_layers),
        "model_type": args.model_type,
        "model_name": args.model_name,
        "cnn_arch": getattr(args, "cnn_arch", None),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log_info(f"Saved {fmt} checkpoint to {pth_path} (+ sidecar)")
    return pth_path


def load_normnet_checkpoint(pth_path, device, args):
    """Build a fresh plain arch and strict-load a normnet checkpoint into it.

    Reads the sidecar for `format` and `biased_layers`, builds the plain architecture
    via load_model (with the original --checkpoint, then overwritten by strict load),
    attaches zero biases so keys match, converts VNR keys if needed, and strict-loads.
    """
    meta_path = os.path.splitext(pth_path)[0] + ".meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    model = load_model(args, device)
    attach_biases(model, meta["biased_layers"])

    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    if meta["format"] == "vnr":
        state = _merge_vnr_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
def _forward(model, xb, model_type):
    model.eval()
    with torch.no_grad():
        return forward_logits(model, xb, model_type).detach().float().cpu()


def verify_roundtrip(model, mgr, loader, args, device, use_ddp):
    """Transform -> save -> reload, asserting forward equivalence at every stage.

    Returns True if all max-abs logit diffs vs the baseline are < args.verify_atol.
    """
    # Baseline sample + logits
    batch = next(iter(loader))
    xb = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
    y0 = _forward(model, xb, args.model_type)

    acc_base = None
    if args.validate and is_main():
        acc_base, _ = validate(model, loader, device, args.model_type)

    # Stage 1: reparameterize (whole-net)
    mgr.reparameterize(loader)
    resolved = list(mgr._reparam_modules.keys())
    log_info(f"Reparameterized {len(resolved)} layers")
    if use_ddp:
        _broadcast_model_state(model)
    y1 = _forward(model, xb, args.model_type)
    vnr_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Stage 2: merge back to standard modules
    mgr.merge_back()
    y2 = _forward(model, xb, args.model_type)
    merged_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    diffs = {
        "reparam": (y1 - y0).abs().max().item(),
        "merge":   (y2 - y0).abs().max().item(),
    }

    # Stages 3-4: save both artifacts, reload, compare (rank 0 owns the files)
    if is_main():
        path_b = save_normnet_checkpoint(merged_state, "merged_biased", resolved, args)
        path_a = save_normnet_checkpoint(vnr_state, "vnr", resolved, args)

        model_b = load_normnet_checkpoint(path_b, device, args)
        y3 = _forward(model_b, xb, args.model_type)
        diffs["reload_merged"] = (y3 - y0).abs().max().item()

        model_a = load_normnet_checkpoint(path_a, device, args)
        y_vnr = _forward(model_a, xb, args.model_type)
        diffs["reload_vnr"] = (y_vnr - y0).abs().max().item()

        if args.validate and acc_base is not None:
            acc_reload, _ = validate(model_b, loader, device, args.model_type)
            log_info(f"Validation acc: baseline={acc_base:.4f} reloaded={acc_reload:.4f} "
                     f"(delta={acc_reload - acc_base:+.4f})")

    for stage, d in diffs.items():
        log_info(f"  max|y_{stage} - y0| = {d:.3e}")

    ok = all(d < args.verify_atol for d in diffs.values())
    log_info(f"Verify {'PASSED' if ok else 'FAILED'} (atol={args.verify_atol:.1e})")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv):
    args = parse_args()

    use_ddp = not args.disable_ddp and "RANK" in os.environ
    if use_ddp:
        device = setup_distributed(args)
    else:
        device = get_device()
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    setup_logging(args.save_dir)
    if is_main():
        log_info("=" * 60)
        log_info("Normalize-net-by-construction (transform + verify)")
        log_info("=" * 60)
        for k, v in vars(args).items():
            logger.info(f"  {k}: {v}")
        log_info(f"Device: {device}")

    _, val_loader, _ = build_dataloaders(args, use_ddp=use_ddp)

    model = load_model(args, device)

    layer_names = build_whole_net_reparam_layers(
        model,
        exclude=args.exclude_layers,
        exclude_classifier=args.exclude_classifier,
        exclude_stem=args.exclude_stem,
    )
    log_info(f"Whole-net selector: {len(layer_names)} candidate layers")

    mgr = NormalizedResidualManager(
        model, layer_names, device,
        lambda_reg=0.0, max_batches=args.max_batches,
    )

    ok = verify_roundtrip(model, mgr, val_loader, args, device, use_ddp)

    if use_ddp:
        cleanup()
    return 0 if ok else 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize-net-by-construction: whole-net VNR transform + verify",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--model_type", default="convnext", choices=["vit", "convnext", "cnn"])
    parser.add_argument("--model_name", default="convnext_tiny",
                        help="Architecture source (HF model ID/dir, ConvNeXt variant)")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional .pth checkpoint to load weights from")
    parser.add_argument("--cnn_arch", default=None,
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "mobilenet_v2"])

    # Data
    parser.add_argument("--data_path", default="/algo/NetOptimization/outputs/VBP/")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=50,
                        help="Calibration batches for BN stats")

    # Whole-net selection
    parser.add_argument("--exclude_layers", nargs="*", default=None,
                        help="Dotted module names to exclude from reparam")
    parser.add_argument("--exclude_classifier", action="store_true", default=True)
    parser.add_argument("--include_classifier", dest="exclude_classifier",
                        action="store_false",
                        help="Also reparameterize the logits head")
    parser.add_argument("--exclude_stem", action="store_true", default=False)

    # Verify / output
    parser.add_argument("--verify_atol", type=float, default=1e-4)
    parser.add_argument("--validate", action="store_true", default=False,
                        help="Also run full-val accuracy delta (slow)")
    parser.add_argument("--save_tag", default="normnet")
    parser.add_argument("--save_dir", default="./output/normalize_net")

    # DDP
    parser.add_argument("--disable_ddp", action="store_true")
    parser.add_argument("--local_rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", 0)))

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
