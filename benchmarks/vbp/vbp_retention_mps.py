"""VBP retention accuracy test for CNNs on MPS / CPU / CUDA.

Tests VBPPruner + VarianceImportance on standard CNNs (ResNet, MobileNetV2).
Uses validation data for both stats collection and evaluation (no train set needed).

Usage:
    # ResNet-50 single ratio
    python benchmarks/vbp/vbp_retention_mps.py --cnn_arch resnet50 --keep_ratio 0.65 --global_pruning

    # MobileNetV2 sweep
    python benchmarks/vbp/vbp_retention_mps.py --cnn_arch mobilenet_v2 --global_pruning --sweep

    # Quick debug (500 images)
    python benchmarks/vbp/vbp_retention_mps.py --cnn_arch resnet50 --global_pruning --sweep --eval_samples 500 --stat_samples 500
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
from torch_pruning.pruner.importance import (
    VarianceImportance, build_cnn_ignored_layers, build_cnn_target_layers,
)


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
# Data loading (val only)
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
        # Check for parquet files first
        parquets = [f for f in os.listdir(args.data_path)
                    if f.startswith("validation") and f.endswith(".parquet")]
        if parquets:
            from datasets import load_dataset
            print(f"Loading {len(parquets)} validation parquet files from {args.data_path}")
            parquet_paths = [os.path.join(args.data_path, f) for f in sorted(parquets)]
            hf_ds = load_dataset("parquet", data_files=parquet_paths, split="train")
            dataset = HFImageNetDataset(hf_ds, transform=val_transform)
        else:
            # ImageFolder — look for val/ subdir or use as-is
            from torchvision.datasets import ImageFolder
            val_dir = os.path.join(args.data_path, "val")
            if os.path.isdir(val_dir):
                print(f"Loading from ImageFolder: {val_dir}")
                dataset = ImageFolder(val_dir, transform=val_transform)
            else:
                print(f"Loading from ImageFolder: {args.data_path}")
                dataset = ImageFolder(args.data_path, transform=val_transform)
    else:
        raise ValueError(f"Data path not found: {args.data_path}. "
                         "Provide --data_path to a dir with validation parquets or ImageFolder.")

    print(f"Loaded {len(dataset)} validation images")
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
    import torchvision.models as tv_models
    model_map = {
        "resnet18": tv_models.resnet18,
        "resnet34": tv_models.resnet34,
        "resnet50": tv_models.resnet50,
        "resnet101": tv_models.resnet101,
        "mobilenet_v2": tv_models.mobilenet_v2,
    }
    model_fn = model_map[args.cnn_arch]
    weights = "DEFAULT" if args.pretrained else None
    model = model_fn(weights=weights).to(device)
    print(f"Loaded {args.cnn_arch} (pretrained={args.pretrained})")
    return model


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
        logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Remap importance for deepcopy
# ---------------------------------------------------------------------------
def remap_importance(imp, orig_model, new_model):
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
# Single-ratio prune
# ---------------------------------------------------------------------------
def recalibrate_bn(model, loader, device):
    """Reset and recalibrate BN running stats on the pruned model.

    Essential for CNNs after structured pruning. Needs 1000+ samples
    (MobileNetV2 needs ~5000 for full recovery).
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
    model.train()
    with torch.no_grad():
        for images, _ in loader:
            model(images.to(device, non_blocking=True))
    model.eval()


def run_single(model, val_loader, stat_loader, device, args, imp, keep_ratio):
    model_copy = copy.deepcopy(model)
    example_inputs = torch.randn(1, 3, 224, 224).to(device)

    imp_mapped = remap_importance(imp, model, model_copy)
    ignored_layers = build_cnn_ignored_layers(
        model_copy, args.cnn_arch, interior_only=args.interior_only)

    pruner = tp.pruner.VBPPruner(
        model_copy,
        example_inputs,
        importance=imp_mapped,
        global_pruning=args.global_pruning,
        pruning_ratio=1.0 - keep_ratio,
        max_pruning_ratio=args.max_pruning_ratio,
        ignored_layers=ignored_layers,
        output_transform=lambda out: out.sum(),
        mean_dict=imp_mapped.means,
    )

    model_copy.eval()
    if not args.no_compensation:
        pruner.enable_meancheck(model_copy)
        with torch.no_grad():
            model_copy(example_inputs)

    pruner.step(interactive=False, enable_compensation=not args.no_compensation)

    if not args.no_compensation:
        pruner.disable_meancheck()

    # Recalibrate BN running stats after pruning
    recalibrate_bn(model_copy, stat_loader, device)

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model_copy, example_inputs)
    acc_ret = validate(model_copy, val_loader, device)

    del model_copy
    return acc_ret, pruned_macs / 1e9, pruned_params / 1e6


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------
def print_table(results, acc_orig):
    header = (f"{'Prune%':>7} | {'Keep':>5} | {'Ret%':>9} | "
              f"{'MACs(G)':>7} | {'Params(M)':>9}")
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(f"  Original accuracy: {acc_orig*100:.2f}%")
    print(sep)
    print(header)
    print(sep)
    for pr, ret, macs, params in results:
        print(f"  {pr:5d}% | {1-pr/100:.2f} | {ret:8.2f}% | "
              f"{macs:6.2f}G | {params:7.2f}M")
    print(sep)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
def run_vbp(model, val_dataset, device, args):
    example_inputs = torch.randn(1, 3, 224, 224).to(device)

    use_pin = (device.type == "cuda")
    workers = 0 if device.type == "mps" else args.num_workers

    # Split val data: first N for stats, rest (or all) for eval
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

    # --- Collect stats using val data ---
    imp = VarianceImportance()
    temp_DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
    target_layers = build_cnn_target_layers(model, temp_DG)
    print(f"\nCollecting activation statistics on {n_stat} val samples "
          f"({len(target_layers)} target layers)...")
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
                model, val_loader, stat_loader, device, args, imp, keep_ratio,
            )
            print(f"  Retention: {acc_ret*100:.2f}%  "
                  f"MACs: {macs_g:.2f}G  Params: {params_m:.2f}M  "
                  f"({time.time()-t0:.0f}s)")
            results.append((pr, acc_ret * 100, macs_g, params_m))
            print_table(results, acc_orig)

    else:
        # --- Single ratio mode ---
        prune_mode = "global" if args.global_pruning else "per_layer"
        print(f"\nPruning with keep_ratio={args.keep_ratio}, mode={prune_mode}...")
        acc_ret, macs_g, params_m = run_single(
            model, val_loader, stat_loader, device, args, imp, args.keep_ratio,
        )

        print()
        print("=" * 62)
        print(f"  Model:       {args.cnn_arch}")
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
        description="VBP retention test for CNNs (MPS/CPU/CUDA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cnn_arch", default="resnet50",
                   choices=["resnet18", "resnet34", "resnet50", "resnet101", "mobilenet_v2"])
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--interior_only", action="store_true", default=True)
    p.add_argument("--data_path",
                   default=os.path.expanduser("~/PycharmProjects/imagenet-1k-test/data"),
                   help="Path to ImageNet val (parquet dir or ImageFolder)")
    p.add_argument("--keep_ratio", type=float, default=0.65)
    p.add_argument("--global_pruning", action="store_true")
    p.add_argument("--max_pruning_ratio", type=float, default=1.0,
                   help="Max fraction of channels to prune per layer (e.g. 0.8 = keep at least 20%%)")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep pruning ratios 5%%..50%% (step 5%%)")
    p.add_argument("--stat_samples", type=int, default=5000,
                   help="Number of val images for variance statistics")
    p.add_argument("--eval_samples", type=int, default=None,
                   help="Limit eval to N images (default: all)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_compensation", action="store_true",
                   help="Disable VBP bias compensation (useful when DW convs lack calibration means)")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(force_cpu=args.cpu)
    print(f"Device: {device}")

    val_dataset = load_imagenet_val(args)
    model = load_model(args, device)
    run_vbp(model, val_dataset, device, args)


if __name__ == "__main__":
    main()
