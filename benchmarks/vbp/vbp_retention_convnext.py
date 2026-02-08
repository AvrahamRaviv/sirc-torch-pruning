"""VBP retention accuracy test for ConvNeXt on MPS / CPU / CUDA.

Tests the Torch-Pruning VBPPruner + VarianceImportance on ConvNeXt-Tiny.

Usage:
    # Single ratio (default: keep_ratio=0.70)
    python benchmarks/vbp/vbp_retention_convnext.py \
        --global_pruning --keep_ratio 0.70

    # MLP-only (prune only pwconv1 intermediate dim)
    python benchmarks/vbp/vbp_retention_convnext.py \
        --global_pruning --keep_ratio 0.70 --mlp_only

    # Use 22K->1K checkpoint (matches paper baseline 82.90%)
    python benchmarks/vbp/vbp_retention_convnext.py \
        --global_pruning --keep_ratio 0.50 --mlp_only --in_22k_1k

    # Sweep mode (5% to 50%)
    python benchmarks/vbp/vbp_retention_convnext.py \
        --global_pruning --sweep

    # Quick debug
    python benchmarks/vbp/vbp_retention_convnext.py \
        --global_pruning --keep_ratio 0.70 \
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

# Local ConvNeXt implementation (FB version)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convnext import convnext_tiny, convnext_small, convnext_base, Block


# ---------------------------------------------------------------------------
# Paper reference: Table 8 ConvNeXt-Tiny (arxiv 2507.12988)
# Baseline: 82.90% (convnext_tiny_22k_1k_224, 22K pretrained + 1K fine-tuned)
# Retention accuracy (before fine-tuning), derived from paper's relative perf.
# Paper post-FT: 81.30% at 50% pruning (20.3% relative retention → ~16.83%)
# ---------------------------------------------------------------------------
PAPER_CONVNEXT_T = {
    # prune_pct: (retention_acc, macs_G, params_M)
    0:  (82.90, 4.47, 28.59),     # baseline (22k→1k checkpoint)
    50: (16.83, 2.96, 12.61),     # keep_ratio=0.50
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
# Data loading  (same as vbp_retention_mps.py, BICUBIC for ConvNeXt)
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
VARIANT_MAP = {
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
    "convnext_base": convnext_base,
}


def load_model(args, device):
    model_fn = VARIANT_MAP.get(args.model, convnext_tiny)

    if args.checkpoint and os.path.exists(args.checkpoint):
        model = model_fn(pretrained=False)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        in_22k_1k = getattr(args, 'in_22k_1k', False)
        model = model_fn(pretrained=True, in_22k_1k=in_22k_1k)
        ckpt_tag = "22k→1k" if in_22k_1k else "1k"
        print(f"Loaded pretrained {args.model} ({ckpt_tag})")

    return model.to(device)


# ---------------------------------------------------------------------------
# ConvNeXt VBP helpers
# ---------------------------------------------------------------------------
def _make_post_gelu_nchw(act_fn):
    """Post-act fn: apply GELU then permute NHWC -> NCHW for stats hook.

    ConvNeXt Linear layers (pwconv1/pwconv2) output [N,H,W,C] (NHWC),
    but VarianceImportance hook assumes [N,C,H,W] for 4D tensors.
    """
    def fn(x):
        return act_fn(x).permute(0, 3, 1, 2)
    return fn


def _nhwc_to_nchw(x):
    """Permute NHWC -> NCHW for ConvNeXt Linear outputs."""
    return x.permute(0, 3, 1, 2)


def build_convnext_target_layers(model, mlp_only=True):
    """Build (module, post_act_fn) pairs for VarianceImportance.collect_statistics."""
    target_layers = []
    for stage in model.stages:
        for block in stage:
            # pwconv1 with post-GELU: primary VBP target
            target_layers.append(
                (block.pwconv1, _make_post_gelu_nchw(block.act))
            )
            if not mlp_only:
                # dwconv: output is already NCHW, no post_act needed
                target_layers.append((block.dwconv, None))
                # pwconv2: NHWC output, permute to NCHW
                target_layers.append((block.pwconv2, _nhwc_to_nchw))

    if not mlp_only:
        for ds in model.downsample_layers:
            for m in ds.modules():
                if isinstance(m, nn.Conv2d):
                    target_layers.append((m, None))

    return target_layers


def build_convnext_ignored_layers(model, mlp_only=True):
    """Build ignored_layers for ConvNeXt pruning."""
    ignored = [model.head]

    if mlp_only:
        # Ignore everything except pwconv1 (intermediate MLP dim)
        for ds in model.downsample_layers:
            for m in ds.modules():
                if isinstance(m, nn.Conv2d):
                    ignored.append(m)
        for stage in model.stages:
            for block in stage:
                ignored.append(block.dwconv)
                ignored.append(block.pwconv2)

    return ignored


def build_unwrapped_parameters(model, mlp_only=True):
    """Build unwrapped_parameters list for ConvNeXt layer_scale (gamma).

    Only needed for all-layers mode where residual stream dim is pruned.
    """
    if mlp_only:
        return None

    params = []
    for stage in model.stages:
        for block in stage:
            if block.gamma is not None:
                params.append((block.gamma, 0))
    return params if params else None


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
        logits = model(images)
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
    print(f"  Paper baseline:    {PAPER_CONVNEXT_T[0][0]:.2f}%")
    print(sep)
    print(header)
    print(sep)
    for pr, ret, macs, params in results:
        paper = PAPER_CONVNEXT_T.get(pr)
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
# Single-ratio prune
# ---------------------------------------------------------------------------
def run_single(model, val_loader, device, args, imp, keep_ratio):
    """Prune a fresh copy at keep_ratio using tp.pruner.VBPPruner."""
    model_copy = copy.deepcopy(model)
    example_inputs = torch.randn(1, 3, 224, 224).to(device)

    imp_mapped = remap_importance(imp, model, model_copy)
    ignored_layers = build_convnext_ignored_layers(model_copy, mlp_only=args.mlp_only)
    unwrapped_params = build_unwrapped_parameters(model_copy, mlp_only=args.mlp_only)

    pruner_kwargs = dict(
        importance=imp_mapped,
        global_pruning=args.global_pruning,
        pruning_ratio=1.0 - keep_ratio,
        ignored_layers=ignored_layers,
        output_transform=lambda out: out.sum(),
        mean_dict=imp_mapped.means,
    )
    if unwrapped_params is not None:
        pruner_kwargs["unwrapped_parameters"] = unwrapped_params

    pruner = tp.pruner.VBPPruner(
        model_copy, example_inputs, **pruner_kwargs,
    )

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

    # --- Collect post-GELU stats using target_layers ---
    imp = VarianceImportance()
    target_layers = build_convnext_target_layers(model, mlp_only=args.mlp_only)
    scope_str = "MLP-only (pwconv1)" if args.mlp_only else "all layers"
    print(f"\nCollecting activation statistics ({scope_str}, "
          f"{len(target_layers)} layers, {n_stat} samples)...")
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

    else:
        # --- Single ratio mode ---
        prune_mode = "global" if args.global_pruning else "per_layer"
        scope = "mlp_only" if args.mlp_only else "all_layers"
        print(f"\nPruning with keep_ratio={args.keep_ratio}, "
              f"mode={prune_mode}, scope={scope}...")
        acc_ret, macs_g, params_m = run_single(
            model, val_loader, device, args, imp, args.keep_ratio,
        )

        pr = round((1.0 - args.keep_ratio) * 100)
        results = [(pr, acc_ret * 100, macs_g, params_m)]
        print_table(results, acc_orig)

        print()
        print("=" * 62)
        print(f"  Model:       {args.model}")
        print(f"  Device:      {device}")
        print(f"  Keep ratio:  {args.keep_ratio}")
        print(f"  Mode:        {prune_mode}")
        print(f"  Scope:       {scope}")
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
        description="VBP retention accuracy test for ConvNeXt — uses tp.pruner.VBPPruner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="convnext_tiny",
                   choices=list(VARIANT_MAP.keys()),
                   help="ConvNeXt variant")
    p.add_argument("--checkpoint", default=None,
                   help="Path to ConvNeXt checkpoint (.pth)")
    p.add_argument("--in_22k_1k", action="store_true",
                   help="Use 22K-pretrained, 1K-finetuned checkpoint "
                        "(82.90%% baseline, matches paper)")
    p.add_argument("--data_path", default="../imagenet-1k-test/data",
                   help="Path to parquet dir or ImageFolder")
    p.add_argument("--keep_ratio", type=float, default=0.70,
                   help="Keep ratio for single-ratio mode")
    p.add_argument("--global_pruning", action="store_true")
    p.add_argument("--mlp_only", action="store_true",
                   help="Only prune MLP intermediate dim (pwconv1→pwconv2). "
                        "Default: prune all layers (only ignore head)")
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


def main():
    args = parse_args()
    device = get_device(force_cpu=args.cpu)
    print(f"Device: {device}")

    val_dataset = load_imagenet_val(args)
    model = load_model(args, device)
    print(f"Loaded: {args.model}")
    run_vbp(model, val_dataset, device, args)


if __name__ == "__main__":
    main()
