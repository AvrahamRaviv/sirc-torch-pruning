"""VBP retention accuracy sweep on MPS / CPU / CUDA.

Tests the Torch-Pruning 1.6 integrated VBPPruner + VarianceImportance.

Collects stats once, then sweeps pruning ratios 5%..50% (step 5%),
reloading the original model for each ratio. Prints a running table
and saves a comparison plot against paper Table 10 (DeiT-T).

Usage:
    # Sweep mode (default: 5% to 50%)
    python benchmarks/vbp/vbp_retention_mps.py \
        --model facebook/deit-tiny-patch16-224 --global_pruning --sweep

    # Single ratio (original behavior)
    python benchmarks/vbp/vbp_retention_mps.py \
        --model facebook/deit-tiny-patch16-224 \
        --keep_ratio 0.65 --global_pruning

    # Quick debug sweep (500 images)
    python benchmarks/vbp/vbp_retention_mps.py \
        --model facebook/deit-tiny-patch16-224 --global_pruning --sweep \
        --eval_samples 500 --stat_samples 500
"""

import argparse
import copy
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import torch_pruning as tp
from torch_pruning.pruner.importance import VarianceImportance
from transformers import ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTSelfAttention


# ---------------------------------------------------------------------------
# Paper reference data: Table 10, DeiT-T (arxiv 2507.12988)
# ---------------------------------------------------------------------------
PAPER_DEIT_T = {
    # prune_pct: (retention_acc, macs_G, params_M)
    0:  (72.02, 1.26, 5.72),
    5:  (71.67, 1.22, 5.54),
    10: (70.95, 1.19, 5.36),
    15: (70.05, 1.15, 5.18),
    20: (68.87, 1.12, 5.01),
    25: (67.37, 1.08, 4.83),
    30: (64.76, 1.05, 4.65),
    35: (61.12, 1.01, 4.48),
    40: (55.64, 0.98, 4.30),
    45: (49.77, 0.94, 4.12),
    50: (39.58, 0.91, 3.94),
}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_imagenet_val(args):
    import torchvision.transforms as T

    val_transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.data_path and os.path.isdir(args.data_path):
        from datasets import load_dataset
        parquets = [f for f in os.listdir(args.data_path)
                    if f.startswith("validation") and f.endswith(".parquet")]
        if parquets:
            print(f"Loading {len(parquets)} validation parquet files from {args.data_path}")
            parquet_paths = [os.path.join(args.data_path, f) for f in sorted(parquets)]
            hf_ds = load_dataset("parquet", data_files=parquet_paths, split="train")
            dataset = HFImageNetDataset(hf_ds, transform=val_transform)
            print(f"Loaded {len(dataset)} validation images")
        else:
            from torchvision.datasets import ImageFolder
            print(f"Loading from local ImageFolder: {args.data_path}")
            dataset = ImageFolder(args.data_path, transform=val_transform)
    else:
        print("Loading ImageNet-1k validation from HuggingFace...")
        from datasets import load_dataset
        hf_ds = load_dataset("ILSVRC/imagenet-1k", split="validation",
                             trust_remote_code=True)
        dataset = HFImageNetDataset(hf_ds, transform=val_transform)
        print(f"Loaded {len(dataset)} validation images from HuggingFace")
    return dataset


class HFImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(args, device):
    if args.model_type == "vit":
        model = ViTForImageClassification.from_pretrained(args.model, local_files_only=True)
        return model.to(device)
    raise ValueError(f"Unknown model_type: {args.model_type}")


def forward_logits(model, images):
    out = model(images)
    return out.logits if hasattr(out, "logits") else out


# ---------------------------------------------------------------------------
# ViT helpers for tp.pruner.VBPPruner
# ---------------------------------------------------------------------------
def build_vit_target_layers(model):
    """Build (module, post_act_fn) pairs for post-GELU stats on fc1."""
    blocks = model.vit.encoder.layer
    return [
        (block.intermediate.dense, block.intermediate.intermediate_act_fn)
        for block in blocks
    ]


def build_vit_ignored_layers(model):
    """Ignore everything except fc1 (intermediate.dense) for MLP-only pruning."""
    ignored = [model.classifier,
               model.vit.embeddings.patch_embeddings.projection]
    for block in model.vit.encoder.layer:
        # Attention layers
        ignored.append(block.attention.attention.query)
        ignored.append(block.attention.attention.key)
        ignored.append(block.attention.attention.value)
        ignored.append(block.attention.output.dense)
        # fc2 output channels are the residual stream — don't prune
        ignored.append(block.output.dense)
    return ignored


def remap_importance(imp, orig_model, new_model):
    """Remap VarianceImportance stats from orig_model modules to new_model modules."""
    orig_to_name = {m: n for n, m in orig_model.named_modules()}
    name_to_new = {n: m for n, m in new_model.named_modules()}

    imp_new = VarianceImportance(norm_per_layer=imp.norm_per_layer, eps=imp.eps)
    for mod, var in imp.variance.items():
        name = orig_to_name.get(mod)
        if name and name in name_to_new:
            new_mod = name_to_new[name]
            imp_new.variance[new_mod] = var
            if mod in imp.means:
                imp_new.means[new_mod] = imp.means[mod]
    return imp_new


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = forward_logits(model, images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------
def print_table(results, acc_orig):
    header = (f"{'Prune%':>7} | {'Keep':>5} | {'Ours Ret%':>9} | "
              f"{'Paper Ret%':>10} | {'Delta':>6} | "
              f"{'MACs(G)':>7} | {'Paper':>6} | {'Params(M)':>9} | {'Paper':>6}")
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(f"  Original accuracy: {acc_orig*100:.2f}%")
    print(f"  Paper baseline:    {PAPER_DEIT_T[0][0]:.2f}%")
    print(sep)
    print(header)
    print(sep)
    for pr, ret, macs, params in results:
        paper = PAPER_DEIT_T.get(pr)
        paper_ret = paper[0] if paper else None
        paper_macs = paper[1] if paper else None
        paper_params = paper[2] if paper else None
        paper_str = f"{paper_ret:.2f}" if paper_ret is not None else "  —"
        delta = f"{ret - paper_ret:+.2f}" if paper_ret is not None else "  —"
        p_macs = f"{paper_macs:.2f}" if paper_macs is not None else "  —"
        p_params = f"{paper_params:.2f}" if paper_params is not None else "  —"
        print(f"  {pr:5d}% | {1-pr/100:.2f} | {ret:8.2f}% | "
              f"{paper_str:>9}% | {delta:>6} | "
              f"{macs:6.2f}G | {p_macs:>5}G | {params:7.2f}M | {p_params:>5}M")
    print(sep)


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------
def save_plot(results, acc_orig, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    paper_prs = sorted(PAPER_DEIT_T.keys())
    paper_rets = [PAPER_DEIT_T[pr][0] for pr in paper_prs]
    ax.plot(paper_prs, paper_rets, "s--", color="#2196F3", linewidth=2,
            markersize=7, label="Paper (Table 10)", zorder=3)

    our_prs = [r[0] for r in results]
    our_rets = [r[1] for r in results]
    ax.plot(our_prs, our_rets, "o-", color="#FF5722", linewidth=2,
            markersize=7, label="Ours (TP 1.6 VBPPruner)", zorder=4)

    ax.axhline(y=acc_orig * 100, color="gray", linestyle=":", linewidth=1,
               label=f"Original ({acc_orig*100:.1f}%)")

    ax.set_xlabel("Pruning Ratio (%)", fontsize=13)
    ax.set_ylabel("Retention Accuracy (%)", fontsize=13)
    ax.set_title("VBP Retention — DeiT-T (TP 1.6 VBPPruner vs paper Table 10)", fontsize=14)
    ax.set_xticks(range(0, 55, 5))
    ax.set_ylim(30, 80)
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {save_path}")


# ---------------------------------------------------------------------------
# Single-ratio prune using tp.pruner.VBPPruner
# ---------------------------------------------------------------------------
def run_single(model, val_loader, device, args, imp, keep_ratio):
    """Prune a fresh copy at keep_ratio using tp.pruner.VBPPruner."""
    model_copy = copy.deepcopy(model)
    example_inputs = torch.randn(1, 3, 224, 224).to(device)

    # Remap importance stats to copy's module objects
    imp_mapped = remap_importance(imp, model, model_copy)

    # Build pruner (MLP-only: ignore attention + embeddings + classifier)
    ignored_layers = build_vit_ignored_layers(model_copy)

    pruner = tp.pruner.VBPPruner(
        model_copy,
        example_inputs,
        importance=imp_mapped,
        global_pruning=args.global_pruning,
        pruning_ratio=1.0 - keep_ratio,
        ignored_layers=ignored_layers,
        output_transform=lambda out: out.logits.sum(),
        mean_dict=imp_mapped.means,  # calibration means enable compensation
    )

    # Prune with compensation (uses calibration means directly, no hooks needed)
    model_copy.eval()
    pruner.step(interactive=False, enable_compensation=True)

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model_copy, example_inputs)
    acc_ret = validate(model_copy, val_loader, device)

    del model_copy
    return acc_ret, pruned_macs / 1e9, pruned_params / 1e6


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
def run_vbp(model, val_dataset, device, args):
    example_inputs = torch.randn(1, 3, 224, 224).to(device)

    use_pin = (device.type == "cuda")
    workers = args.num_workers if device.type != "mps" else 0

    n_stat = min(args.stat_samples, len(val_dataset))
    stat_dataset = Subset(val_dataset, range(n_stat))
    stat_loader = DataLoader(
        stat_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=workers, pin_memory=use_pin,
    )

    if args.eval_samples and args.eval_samples < len(val_dataset):
        eval_dataset = Subset(val_dataset, range(args.eval_samples))
    else:
        eval_dataset = val_dataset
    val_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=workers, pin_memory=use_pin,
    )

    # --- Baseline ---
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"\nBaseline: {base_macs/1e9:.2f}G MACs, {base_params/1e6:.2f}M params")

    print("Evaluating original model...")
    t0 = time.time()
    acc_orig = validate(model, val_loader, device)
    print(f"Original accuracy: {acc_orig*100:.2f}%  ({time.time()-t0:.0f}s)")

    # --- Collect post-GELU stats ONCE using tp.importance.VarianceImportance ---
    imp = VarianceImportance()
    target_layers = build_vit_target_layers(model)
    print(f"\nCollecting post-GELU activation statistics on {n_stat} samples "
          f"({len(target_layers)} fc1 layers)...")
    t0 = time.time()
    imp.collect_statistics(model, stat_loader, device, target_layers=target_layers)
    print(f"Statistics collected for {len(imp.variance)} layers  ({time.time()-t0:.0f}s)")

    if args.sweep:
        # --- Sweep mode ---
        prune_pcts = list(range(5, 55, 5))
        results = []

        for pr in prune_pcts:
            keep_ratio = 1.0 - pr / 100.0
            print(f"\n--- Pruning {pr}% (keep={keep_ratio:.2f}) ---")
            t0 = time.time()
            acc_ret, macs_g, params_m = run_single(
                model, val_loader, device, args, imp, keep_ratio,
            )
            print(f"  Retention: {acc_ret*100:.2f}%  "
                  f"MACs: {macs_g:.2f}G  Params: {params_m:.2f}M  "
                  f"({time.time()-t0:.0f}s)")
            results.append((pr, acc_ret * 100, macs_g, params_m))
            print_table(results, acc_orig)

        plot_path = os.path.join(os.path.dirname(__file__), "vbp_sweep_deit_t.png")
        save_plot(results, acc_orig, plot_path)

    else:
        # --- Single ratio mode ---
        prune_mode = "global" if args.global_pruning else "per_layer"
        print(f"\nPruning with keep_ratio={args.keep_ratio}, mode={prune_mode}...")
        acc_ret, macs_g, params_m = run_single(
            model, val_loader, device, args, imp, args.keep_ratio,
        )

        print()
        print("=" * 62)
        print(f"  Model:       {args.model}")
        print(f"  Device:      {device}")
        print(f"  Keep ratio:  {args.keep_ratio}")
        print(f"  Mode:        {prune_mode}")
        print(f"  Stat samples:{n_stat}")
        print("-" * 62)
        print(f"  MACs:        {base_macs/1e9:.2f}G  ->  {macs_g:.2f}G  "
              f"({macs_g/(base_macs/1e9)*100:.1f}% kept)")
        print(f"  Params:      {base_params/1e6:.2f}M  ->  {params_m:.2f}M  "
              f"({params_m/(base_params/1e6)*100:.1f}% kept)")
        print("-" * 62)
        print(f"  Original:    {acc_orig*100:.2f}%")
        print(f"  Retention:   {acc_ret*100:.2f}%")
        print(f"  Drop:        {(acc_orig - acc_ret)*100:.2f} pp")
        print("=" * 62)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="VBP retention accuracy test (MPS/CPU/CUDA) — uses tp.pruner.VBPPruner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="/algo/NetOptimization/outputs/VBP/DeiT_tiny",
                   help="HuggingFace model name")
    p.add_argument("--model_type", default="vit", choices=["vit"])
    p.add_argument("--data_path", default="../imagenet-1k-test/data",
                   help="Path to parquet dir or ImageFolder")
    p.add_argument("--keep_ratio", type=float, default=0.65,
                   help="Keep ratio for single-ratio mode")
    p.add_argument("--global_pruning", action="store_true")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep pruning ratios 5%%..50%% (step 5%%)")
    p.add_argument("--stat_samples", type=int, default=5000,
                   help="Number of images for variance statistics")
    p.add_argument("--eval_samples", type=int, default=None,
                   help="Limit eval to N images (default: all)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p.parse_args()


def main(argv):
    args = parse_args()
    device = get_device(force_cpu=args.cpu)
    print(f"Device: {device}")

    val_dataset = load_imagenet_val(args)
    model = load_model(args, device)
    print(f"Loaded: {args.model}")
    run_vbp(model, val_dataset, device, args)


if __name__ == '__main__':
    main(sys.argv[1:])
