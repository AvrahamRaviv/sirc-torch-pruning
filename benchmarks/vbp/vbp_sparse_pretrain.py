"""Sparse pre-training → VBP pruning → optional fine-tuning.

Two-stage pipeline: a sparse pre-training stage concentrates information
into fewer fc1 neurons, sharpening the variance distribution so VBP's
global thresholding yields better channel selection at aggressive ratios.

Modes:
  --sparse_mode l1_group : L2,1 group regularization on fc1 weights
  --sparse_mode gmp      : Gradual magnitude pruning (Zhu & Gupta 2017)
  --sparse_mode none     : Baseline VBP (no sparse pre-training)

Usage:
    # L1 group sparse pre-training + VBP sweep
    python benchmarks/vbp/vbp_sparse_pretrain.py \\
        --model facebook/deit-tiny-patch16-224 \\
        --data_path ../imagenet-1k-test/data \\
        --sparse_mode l1_group --epochs_sparse 5 --l1_lambda 1e-4 \\
        --global_pruning --sweep

    # GMP sparse pre-training + VBP sweep
    python benchmarks/vbp/vbp_sparse_pretrain.py \\
        --model facebook/deit-tiny-patch16-224 \\
        --data_path ../imagenet-1k-test/data \\
        --sparse_mode gmp --epochs_sparse 5 --gmp_target_sparsity 0.5 \\
        --global_pruning --sweep

    # Baseline (no sparse pre-training)
    python benchmarks/vbp/vbp_sparse_pretrain.py \\
        --model facebook/deit-tiny-patch16-224 \\
        --data_path ../imagenet-1k-test/data \\
        --sparse_mode none --global_pruning --sweep

    # Quick smoke test
    python benchmarks/vbp/vbp_sparse_pretrain.py \\
        --model facebook/deit-tiny-patch16-224 \\
        --data_path ../imagenet-1k-test/data \\
        --sparse_mode l1_group --epochs_sparse 2 \\
        --stat_samples 500 --eval_samples 500 \\
        --global_pruning --sweep
"""

import argparse
import copy
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import torch_pruning as tp
from torch_pruning.pruner.importance import VarianceImportance
from transformers import ViTForImageClassification

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparse_utils import (
    get_fc1_modules,
    l21_regularization,
    gmp_sparsity_schedule,
    apply_unstructured_pruning,
    remove_pruning_reparametrization,
    compute_variance_entropy,
    compute_weight_sparsity,
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
# Data loading (from vbp_retention_mps.py pattern)
# ---------------------------------------------------------------------------
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


def load_imagenet(args):
    """Load ImageNet validation and optional training splits."""
    import torchvision.transforms as T

    val_transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_dataset = None
    train_dataset = None

    if args.data_path and os.path.isdir(args.data_path):
        from datasets import load_dataset

        # Validation
        parquets = [f for f in os.listdir(args.data_path)
                    if f.startswith("validation") and f.endswith(".parquet")]
        if parquets:
            print(f"Loading {len(parquets)} validation parquet files")
            paths = [os.path.join(args.data_path, f) for f in sorted(parquets)]
            hf_ds = load_dataset("parquet", data_files=paths, split="train")
            val_dataset = HFImageNetDataset(hf_ds, transform=val_transform)
        else:
            val_dir = os.path.join(args.data_path, "val")
            if os.path.isdir(val_dir):
                from torchvision.datasets import ImageFolder
                val_dataset = ImageFolder(val_dir, transform=val_transform)
            else:
                from torchvision.datasets import ImageFolder
                val_dataset = ImageFolder(args.data_path, transform=val_transform)

        # Training (parquet or ImageFolder)
        train_parquets = [f for f in os.listdir(args.data_path)
                          if f.startswith("train") and f.endswith(".parquet")]
        if train_parquets:
            print(f"Loading {len(train_parquets)} training parquet files")
            paths = [os.path.join(args.data_path, f) for f in sorted(train_parquets)]
            hf_ds = load_dataset("parquet", data_files=paths, split="train")
            train_dataset = HFImageNetDataset(hf_ds, transform=train_transform)
        else:
            train_dir = os.path.join(args.data_path, "train")
            if os.path.isdir(train_dir):
                from torchvision.datasets import ImageFolder
                train_dataset = ImageFolder(train_dir, transform=train_transform)

    if val_dataset is None:
        print("Loading ImageNet-1k validation from HuggingFace...")
        from datasets import load_dataset
        hf_ds = load_dataset("ILSVRC/imagenet-1k", split="validation",
                             trust_remote_code=True)
        val_dataset = HFImageNetDataset(hf_ds, transform=val_transform)

    print(f"Val samples: {len(val_dataset)}"
          + (f", Train samples: {len(train_dataset)}" if train_dataset else ""))
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(args, device):
    model = ViTForImageClassification.from_pretrained(
        args.model, local_files_only=os.path.isdir(args.model))
    return model.to(device)


def forward_logits(model, images):
    out = model(images)
    return out.logits if hasattr(out, "logits") else out


# ---------------------------------------------------------------------------
# ViT helpers (from vbp_retention_mps.py)
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
        ignored.append(block.attention.attention.query)
        ignored.append(block.attention.attention.key)
        ignored.append(block.attention.attention.value)
        ignored.append(block.attention.output.dense)
        ignored.append(block.output.dense)
    return ignored


def remap_importance(imp, orig_model, new_model):
    """Remap VarianceImportance stats from orig_model modules to new_model."""
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
# Sparse pre-training: L1 group mode
# ---------------------------------------------------------------------------
def _run_l1_group_pretraining(model, train_loader, val_loader, device, args):
    """Sparse pre-training with L2,1 group regularization on fc1 weights.

    loss = CE + lambda * L2,1(fc1_weights)
    """
    fc1_pairs = get_fc1_modules(model, model_type="vit")
    fc1_modules = [m for _, m in fc1_pairs]
    print(f"L1 group pre-training: {len(fc1_modules)} fc1 layers, "
          f"lambda={args.l1_lambda}, epochs={args.epochs_sparse}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_sparse,
                                  weight_decay=0.01)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_sparse * steps_per_epoch, eta_min=1e-8)

    for epoch in range(args.epochs_sparse):
        model.train()
        total_ce = 0.0
        total_l21 = 0.0
        num_batches = 0
        log_interval = max(len(train_loader) // 20, 1)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = forward_logits(model, images)
            ce_loss = F.cross_entropy(logits, labels)
            l21_loss = l21_regularization(fc1_modules, device)

            loss = ce_loss + args.l1_lambda * l21_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_ce += ce_loss.item()
            total_l21 += l21_loss.item()
            num_batches += 1

            if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
                avg_ce = total_ce / num_batches
                avg_l21 = total_l21 / num_batches
                print(f"  Epoch {epoch+1}/{args.epochs_sparse} "
                      f"[{batch_idx+1}/{len(train_loader)}] "
                      f"CE={avg_ce:.4f} L21={avg_l21:.2f}")

        # Validate
        acc = validate(model, val_loader, device)
        print(f"  Epoch {epoch+1} done — val_acc={acc*100:.2f}%")

        # Sanity: model didn't collapse
        if acc < 0.01:
            print("WARNING: Model accuracy collapsed below 1%. "
                  "Consider reducing l1_lambda or lr_sparse.")
            break

    return model


# ---------------------------------------------------------------------------
# Sparse pre-training: GMP mode
# ---------------------------------------------------------------------------
def _run_gmp_pretraining(model, train_loader, val_loader, device, args):
    """Sparse pre-training with gradual magnitude pruning on fc1 weights."""
    fc1_pairs = get_fc1_modules(model, model_type="vit")
    fc1_modules = [m for _, m in fc1_pairs]
    print(f"GMP pre-training: {len(fc1_modules)} fc1 layers, "
          f"target_sparsity={args.gmp_target_sparsity}, epochs={args.epochs_sparse}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_sparse,
                                  weight_decay=0.01)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_sparse * steps_per_epoch, eta_min=1e-8)

    for epoch in range(args.epochs_sparse):
        # Compute and apply target sparsity for this epoch
        target_s = gmp_sparsity_schedule(
            epoch, args.epochs_sparse,
            init_s=0.0, target_s=args.gmp_target_sparsity)
        apply_unstructured_pruning(fc1_modules, target_s)

        # Verify actual sparsity
        sparsity_info = compute_weight_sparsity(fc1_modules)
        print(f"  Epoch {epoch+1}: target_sparsity={target_s:.4f}, "
              f"actual={sparsity_info['global']:.4f}")

        # Sanity: sparsity matches within tolerance
        if abs(sparsity_info["global"] - target_s) > 0.02:
            print(f"  WARNING: Sparsity mismatch > 2% "
                  f"(target={target_s:.4f}, actual={sparsity_info['global']:.4f})")

        # Train one epoch
        model.train()
        total_loss = 0.0
        num_batches = 0
        log_interval = max(len(train_loader) // 20, 1)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = forward_logits(model, images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
                avg_loss = total_loss / num_batches
                print(f"  Epoch {epoch+1}/{args.epochs_sparse} "
                      f"[{batch_idx+1}/{len(train_loader)}] loss={avg_loss:.4f}")

        acc = validate(model, val_loader, device)
        print(f"  Epoch {epoch+1} done — val_acc={acc*100:.2f}%")

        if acc < 0.01:
            print("WARNING: Model accuracy collapsed below 1%. "
                  "Consider reducing gmp_target_sparsity.")
            break

    # Bake masks into weights before VBP stats collection
    remove_pruning_reparametrization(fc1_modules)

    # Verify no masks remain
    for m in fc1_modules:
        assert not hasattr(m, "weight_mask"), \
            f"weight_mask still present after remove_pruning_reparametrization"

    final_sparsity = compute_weight_sparsity(fc1_modules)
    print(f"GMP done — final weight sparsity: {final_sparsity['global']:.4f}")

    return model


# ---------------------------------------------------------------------------
# VBP pipeline: stats → prune → eval
# ---------------------------------------------------------------------------
def run_single(model, val_loader, device, args, imp, keep_ratio):
    """Prune a fresh copy at keep_ratio using VBPPruner."""
    model_copy = copy.deepcopy(model)
    example_inputs = torch.randn(1, 3, 224, 224).to(device)

    imp_mapped = remap_importance(imp, model, model_copy)
    ignored_layers = build_vit_ignored_layers(model_copy)

    pruner = tp.pruner.VBPPruner(
        model_copy,
        example_inputs,
        importance=imp_mapped,
        global_pruning=args.global_pruning,
        pruning_ratio=1.0 - keep_ratio,
        ignored_layers=ignored_layers,
        output_transform=lambda out: out.logits.sum(),
        mean_dict=imp_mapped.means,
    )

    model_copy.eval()
    pruner.step(interactive=False, enable_compensation=True)

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(
        model_copy, example_inputs)
    acc_ret = validate(model_copy, val_loader, device)

    del model_copy
    return acc_ret, pruned_macs / 1e9, pruned_params / 1e6


# ---------------------------------------------------------------------------
# Fine-tuning with optional KD (from vbp_imagenet.py pattern)
# ---------------------------------------------------------------------------
def fine_tune(model, teacher, train_loader, val_loader, device, args):
    """Fine-tune pruned model with optional knowledge distillation."""
    print(f"\nFine-tuning for {args.epochs_ft} epochs "
          f"(KD={'on' if args.use_kd else 'off'})...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_ft,
                                  weight_decay=0.01)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_ft * steps_per_epoch, eta_min=1e-8)

    use_kd = args.use_kd and teacher is not None
    best_acc = 0.0

    for epoch in range(args.epochs_ft):
        model.train()
        total_loss = 0.0
        num_batches = 0
        log_interval = max(len(train_loader) // 20, 1)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            student_logits = forward_logits(model, images)
            ce_loss = F.cross_entropy(student_logits, labels)

            if use_kd:
                with torch.no_grad():
                    teacher_logits = forward_logits(teacher, images)
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / args.kd_T, dim=1),
                    F.softmax(teacher_logits / args.kd_T, dim=1),
                    reduction="batchmean",
                ) * (args.kd_T ** 2)
                loss = args.kd_alpha * ce_loss + (1 - args.kd_alpha) * kd_loss
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
                avg_loss = total_loss / num_batches
                print(f"  FT Epoch {epoch+1}/{args.epochs_ft} "
                      f"[{batch_idx+1}/{len(train_loader)}] loss={avg_loss:.4f}")

        acc = validate(model, val_loader, device)
        print(f"  FT Epoch {epoch+1} done — val_acc={acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc

    return best_acc


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------
PAPER_DEIT_T = {
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


def print_table(results, acc_orig, sparse_mode):
    header = (f"{'Prune%':>7} | {'Keep':>5} | {'Ours Ret%':>9} | "
              f"{'Paper Ret%':>10} | {'Delta':>6} | "
              f"{'MACs(G)':>7} | {'Params(M)':>9}")
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(f"  Sparse mode:       {sparse_mode}")
    print(f"  Original accuracy: {acc_orig*100:.2f}%")
    print(sep)
    print(header)
    print(sep)
    for pr, ret, macs, params in results:
        paper = PAPER_DEIT_T.get(pr)
        paper_ret = paper[0] if paper else None
        paper_str = f"{paper_ret:.2f}" if paper_ret is not None else "  —"
        delta = f"{ret - paper_ret:+.2f}" if paper_ret is not None else "  —"
        print(f"  {pr:5d}% | {1-pr/100:.2f} | {ret:8.2f}% | "
              f"{paper_str:>9}% | {delta:>6} | "
              f"{macs:6.2f}G | {params:7.2f}M")
    print(sep)


# ---------------------------------------------------------------------------
# Variance metrics reporting
# ---------------------------------------------------------------------------
def print_variance_metrics(label, metrics):
    print(f"\n  Variance distribution ({label}):")
    print(f"    Entropy (normalized): {metrics['entropy']:.4f}")
    print(f"    CV:                   {metrics['cv']:.4f}")
    print(f"    Gini:                 {metrics['gini']:.4f}")
    print(f"    Top 10% concentration:{metrics['top10_pct']:.4f}")
    print(f"    Top 20% concentration:{metrics['top20_pct']:.4f}")
    print(f"    Top 50% concentration:{metrics['top50_pct']:.4f}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args):
    device = get_device(force_cpu=args.cpu)
    print(f"Device: {device}")

    use_pin = (device.type == "cuda")
    workers = args.num_workers if device.type != "mps" else 0

    # Load data
    train_dataset, val_dataset = load_imagenet(args)

    # Build loaders
    n_stat = min(args.stat_samples, len(val_dataset))
    stat_dataset = Subset(val_dataset, range(n_stat))
    stat_loader = DataLoader(
        stat_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=workers, pin_memory=use_pin)

    if args.eval_samples and args.eval_samples < len(val_dataset):
        eval_dataset = Subset(val_dataset, range(args.eval_samples))
    else:
        eval_dataset = val_dataset
    val_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=workers, pin_memory=use_pin)

    # Training loader (subset if requested)
    train_loader = None
    if train_dataset is not None and args.sparse_mode != "none":
        if args.train_samples and args.train_samples < len(train_dataset):
            train_sub = Subset(train_dataset, range(args.train_samples))
        else:
            train_sub = train_dataset
        train_loader = DataLoader(
            train_sub, batch_size=args.train_batch_size, shuffle=True,
            num_workers=workers, pin_memory=use_pin, drop_last=True)
    elif args.sparse_mode != "none":
        print("WARNING: No training data available. Using validation data "
              "for sparse pre-training (NOT recommended for real experiments).")
        train_loader = DataLoader(
            eval_dataset, batch_size=args.train_batch_size, shuffle=True,
            num_workers=workers, pin_memory=use_pin, drop_last=True)

    # Fine-tuning loader
    ft_loader = None
    if args.epochs_ft > 0:
        if train_dataset is not None:
            if args.train_samples and args.train_samples < len(train_dataset):
                ft_sub = Subset(train_dataset, range(args.train_samples))
            else:
                ft_sub = train_dataset
            ft_loader = DataLoader(
                ft_sub, batch_size=args.train_batch_size, shuffle=True,
                num_workers=workers, pin_memory=use_pin, drop_last=True)
        else:
            ft_loader = val_loader

    # Load model
    model = load_model(args, device)
    print(f"Loaded: {args.model}")

    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Baseline: {base_macs/1e9:.2f}G MACs, {base_params/1e6:.2f}M params")

    # --- Baseline evaluation ---
    print("\nEvaluating original model...")
    t0 = time.time()
    acc_orig = validate(model, val_loader, device)
    print(f"Original accuracy: {acc_orig*100:.2f}%  ({time.time()-t0:.0f}s)")

    # --- Baseline VBP variance metrics ---
    print("\nCollecting baseline variance metrics...")
    imp_baseline = VarianceImportance()
    target_layers = build_vit_target_layers(model)
    imp_baseline.collect_statistics(
        model, stat_loader, device, target_layers=target_layers)
    baseline_metrics = compute_variance_entropy(imp_baseline)
    print_variance_metrics("baseline", baseline_metrics)

    # --- Sparse pre-training ---
    if args.sparse_mode == "none":
        print("\nSparse mode: none (baseline VBP)")
        if args.epochs_ft > 0:
            print(f"NOTE: For fair comparison with sparse modes, "
                  f"baseline should use epochs_ft = epochs_sparse + epochs_ft "
                  f"= {args.epochs_sparse + args.epochs_ft}")
        imp = imp_baseline
    else:
        assert train_loader is not None, "Train loader required for sparse pre-training"

        if args.sparse_mode == "l1_group":
            print(f"\n{'='*60}")
            print("Stage 1: L2,1 group sparse pre-training")
            print(f"{'='*60}")
            t0 = time.time()
            model = _run_l1_group_pretraining(
                model, train_loader, val_loader, device, args)
            print(f"L1 group pre-training done ({time.time()-t0:.0f}s)")

        elif args.sparse_mode == "gmp":
            print(f"\n{'='*60}")
            print("Stage 1: GMP sparse pre-training")
            print(f"{'='*60}")
            t0 = time.time()
            model = _run_gmp_pretraining(
                model, train_loader, val_loader, device, args)
            print(f"GMP pre-training done ({time.time()-t0:.0f}s)")

        # Post-sparse accuracy
        print("\nEvaluating post-sparse model...")
        acc_sparse = validate(model, val_loader, device)
        print(f"Post-sparse accuracy: {acc_sparse*100:.2f}%")
        if acc_sparse < 0.50:
            print("WARNING: Post-sparse accuracy < 50%. Model may have "
                  "collapsed. VBP results may be unreliable.")

        # Post-sparse VBP variance metrics
        print("\nCollecting post-sparse variance metrics...")
        imp = VarianceImportance()
        target_layers = build_vit_target_layers(model)
        imp.collect_statistics(
            model, stat_loader, device, target_layers=target_layers)
        sparse_metrics = compute_variance_entropy(imp)
        print_variance_metrics("post-sparse", sparse_metrics)

        # Comparison
        print(f"\n  Variance sharpening comparison:")
        print(f"    Entropy:  {baseline_metrics['entropy']:.4f} → "
              f"{sparse_metrics['entropy']:.4f} "
              f"({'sharper' if sparse_metrics['entropy'] < baseline_metrics['entropy'] else 'flatter'})")
        print(f"    CV:       {baseline_metrics['cv']:.4f} → "
              f"{sparse_metrics['cv']:.4f}")
        print(f"    Gini:     {baseline_metrics['gini']:.4f} → "
              f"{sparse_metrics['gini']:.4f}")
        print(f"    Top 10%:  {baseline_metrics['top10_pct']:.4f} → "
              f"{sparse_metrics['top10_pct']:.4f}")

        # Weight sparsity (for GMP)
        if args.sparse_mode == "gmp":
            fc1_modules = [m for _, m in get_fc1_modules(model, "vit")]
            ws = compute_weight_sparsity(fc1_modules)
            print(f"    Weight sparsity: {ws['global']:.4f}")

    # --- VBP pruning ---
    print(f"\n{'='*60}")
    print("Stage 2: VBP pruning")
    print(f"{'='*60}")

    # Create teacher for KD before pruning
    teacher = None
    if args.use_kd and args.epochs_ft > 0:
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    if args.sweep:
        prune_pcts = list(range(5, 55, 5))
        results = []

        for pr in prune_pcts:
            keep_ratio = 1.0 - pr / 100.0
            print(f"\n--- Pruning {pr}% (keep={keep_ratio:.2f}) ---")
            t0 = time.time()
            acc_ret, macs_g, params_m = run_single(
                model, val_loader, device, args, imp, keep_ratio)
            print(f"  Retention: {acc_ret*100:.2f}%  "
                  f"MACs: {macs_g:.2f}G  Params: {params_m:.2f}M  "
                  f"({time.time()-t0:.0f}s)")
            results.append((pr, acc_ret * 100, macs_g, params_m))
            print_table(results, acc_orig, args.sparse_mode)

    else:
        keep_ratio = args.keep_ratio
        prune_mode = "global" if args.global_pruning else "per_layer"
        print(f"Pruning with keep_ratio={keep_ratio}, mode={prune_mode}")
        acc_ret, macs_g, params_m = run_single(
            model, val_loader, device, args, imp, keep_ratio)

        print()
        print("=" * 62)
        print(f"  Model:        {args.model}")
        print(f"  Sparse mode:  {args.sparse_mode}")
        print(f"  Keep ratio:   {keep_ratio}")
        print(f"  Mode:         {prune_mode}")
        print("-" * 62)
        print(f"  MACs:         {base_macs/1e9:.2f}G  ->  {macs_g:.2f}G  "
              f"({macs_g/(base_macs/1e9)*100:.1f}% kept)")
        print(f"  Params:       {base_params/1e6:.2f}M  ->  {params_m:.2f}M  "
              f"({params_m/(base_params/1e6)*100:.1f}% kept)")
        print("-" * 62)
        print(f"  Original:     {acc_orig*100:.2f}%")
        print(f"  Retention:    {acc_ret*100:.2f}%")
        print(f"  Drop:         {(acc_orig - acc_ret)*100:.2f} pp")
        print("=" * 62)

    # --- Optional fine-tuning (single ratio only) ---
    if args.epochs_ft > 0 and not args.sweep and ft_loader is not None:
        # Re-prune a fresh copy for fine-tuning
        model_ft = copy.deepcopy(model)
        example_inputs = torch.randn(1, 3, 224, 224).to(device)
        imp_ft = remap_importance(imp, model, model_ft)
        ignored = build_vit_ignored_layers(model_ft)

        pruner = tp.pruner.VBPPruner(
            model_ft, example_inputs,
            importance=imp_ft,
            global_pruning=args.global_pruning,
            pruning_ratio=1.0 - args.keep_ratio,
            ignored_layers=ignored,
            output_transform=lambda out: out.logits.sum(),
            mean_dict=imp_ft.means,
        )
        model_ft.eval()
        pruner.step(interactive=False, enable_compensation=True)

        # Remap teacher if using KD
        teacher_ft = None
        if teacher is not None:
            teacher_ft = teacher  # unpruned teacher

        best_acc = fine_tune(
            model_ft, teacher_ft, ft_loader, val_loader, device, args)

        print(f"\n  Fine-tuning best accuracy: {best_acc*100:.2f}%")
        del model_ft

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Sparse mode:       {args.sparse_mode}")
    print(f"  Original acc:      {acc_orig*100:.2f}%")
    print(f"  Baseline entropy:  {baseline_metrics['entropy']:.4f}")
    if args.sparse_mode != "none":
        print(f"  Post-sparse acc:   {acc_sparse*100:.2f}%")
        print(f"  Sparse entropy:    {sparse_metrics['entropy']:.4f}")
        print(f"  Entropy change:    {sparse_metrics['entropy'] - baseline_metrics['entropy']:+.4f}")
    if args.sweep:
        print(f"  Sweep results:     {len(results)} ratios evaluated")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Sparse pre-training → VBP pruning pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model", default="/algo/NetOptimization/outputs/VBP/DeiT_tiny",
                   help="HuggingFace model name or path")
    p.add_argument("--model_type", default="vit", choices=["vit"])

    # Data
    p.add_argument("--data_path", default="../imagenet-1k-test/data",
                   help="Path to parquet dir or ImageFolder root")
    p.add_argument("--stat_samples", type=int, default=5000,
                   help="Images for variance statistics")
    p.add_argument("--eval_samples", type=int, default=None,
                   help="Limit eval to N images")
    p.add_argument("--train_samples", type=int, default=None,
                   help="Limit training to N images")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Eval/stats batch size")
    p.add_argument("--train_batch_size", type=int, default=32,
                   help="Training batch size")
    p.add_argument("--num_workers", type=int, default=4)

    # Sparse pre-training
    p.add_argument("--sparse_mode", default="l1_group",
                   choices=["l1_group", "gmp", "none"],
                   help="Sparse pre-training mode")
    p.add_argument("--epochs_sparse", type=int, default=5,
                   help="Sparse pre-training epochs")
    p.add_argument("--lr_sparse", type=float, default=1e-4,
                   help="Learning rate for sparse phase")
    p.add_argument("--l1_lambda", type=float, default=1e-4,
                   help="L2,1 regularization strength (l1_group mode)")
    p.add_argument("--gmp_target_sparsity", type=float, default=0.5,
                   help="Target weight sparsity for GMP mode")

    # VBP pruning
    p.add_argument("--keep_ratio", type=float, default=0.65,
                   help="Keep ratio for single-ratio mode")
    p.add_argument("--global_pruning", action="store_true")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep pruning ratios 5%%..50%%")

    # Fine-tuning
    p.add_argument("--epochs_ft", type=int, default=0,
                   help="Fine-tuning epochs (0 = skip)")
    p.add_argument("--lr_ft", type=float, default=1.5e-5,
                   help="Fine-tuning learning rate")
    p.add_argument("--use_kd", action="store_true",
                   help="Enable knowledge distillation during fine-tuning")
    p.add_argument("--kd_alpha", type=float, default=0.7,
                   help="CE weight in KD loss")
    p.add_argument("--kd_T", type=float, default=2.0,
                   help="KD temperature")

    # Misc
    p.add_argument("--cpu", action="store_true", help="Force CPU")

    return p.parse_args()


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
