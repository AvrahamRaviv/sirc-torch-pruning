"""
train_v2.py — from-scratch ResNet training under the OFFICIAL torchvision v2 recipe, with
an optional CONTINUOUS mid-run switch to normalized coordinates (--reparam_at_epoch X).

Goal: test §7 — does moving to the normalized parametrization mid-training converge
faster and/or to higher accuracy than the plain official protocol?

Official v2 recipe (the ResNet50_Weights.IMAGENET1K_V2 / 80.86 protocol), self-contained
here so the existing vbp_common training paths (tp_variance etc.) stay byte-identical:
  - SGD, lr (linearly scaled to batch), cosine + linear warmup, nesterov
  - weight decay 2e-5 on weights only (NO wd on norm/bias)
  - TrivialAugmentWide + RandomErasing, train RandomResizedCrop(176), val Resize(232)+Crop(224)
  - MixUp(0.2) / CutMix(1.0) per batch, label smoothing 0.1
  - Model EMA (evaluated alongside the raw model)
  - Repeated-augmentation sampler (DDP)

Two arms (see generate_v2_experiments.py):
  A  baseline:  plain official recipe, N epochs.
  B  switch:    identical, but at epoch X reparameterize (normalize) + rebuild optimizer +
                reset EMA + resume cosine, then train the rest in normalized coords.

Per-epoch val_acc (raw + EMA) is logged to <tag>_metrics.jsonl for curve comparison.
"""
import argparse
import copy
import math
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from vbp_common import load_model, validate, build_whole_net_reparam_layers, FastImageNet
from normalize_net import (
    build_reparam_manager, log_info, get_device, append_metrics, write_run,
    setup_logging, is_main, _broadcast_model_state,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------------------
# Repeated-augmentation distributed sampler (torchvision references/classification)
# --------------------------------------------------------------------------------------
class RASampler(torch.utils.data.Sampler):
    """Each rank draws shuffled indices, each repeated `reps` times, then takes its slice —
    the repeated-augmentation sampler from the v2 recipe."""
    def __init__(self, dataset, num_replicas, rank, shuffle=True, reps=4):
        self.dataset = dataset; self.num_replicas = num_replicas; self.rank = rank
        self.shuffle = shuffle; self.reps = reps; self.epoch = 0
        # stride basis for the reps-expanded pool …
        self.num_samples = math.ceil(len(dataset) * reps / num_replicas)
        self.total_size = self.num_samples * num_replicas
        # … but each rank YIELDS only len/world per epoch (torchvision RASampler), so one
        # epoch ≈ one pass over the data (drawn from the reps-repeated pool), NOT reps passes.
        self.num_selected = len(dataset) // num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator(); g.manual_seed(self.epoch)
        idx = (torch.randperm(len(self.dataset), generator=g) if self.shuffle
               else torch.arange(len(self.dataset))).tolist()
        idx = [i for i in idx for _ in range(self.reps)][:self.total_size]
        idx = idx[self.rank:self.total_size:self.num_replicas]
        return iter(idx[:self.num_selected])

    def __len__(self):
        return self.num_selected


def build_loaders(args, use_ddp):
    train_tf = T.Compose([
        T.RandomResizedCrop(args.train_crop, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(0.5),
        T.TrivialAugmentWide(interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(args.random_erase),
    ])
    val_tf = T.Compose([
        T.Resize(args.val_resize, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    # Clean transform (no TA / erase) for σ,μ calibration at the reparam switch.
    calib_tf = T.Compose([
        T.Resize(args.val_resize, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    # Match vbp_common.build_dataloaders: cached pickle sample-lists (FastImageNet) fast
    # path, ImageFolder only as fallback. The cluster has <data_path>/{train,val}_samples.pkl,
    # NOT a {train,val}/ ImageFolder tree.
    def _make_dst(split, transform):
        pkl = os.path.join(args.data_path, f"{split}_samples.pkl")
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                samples = pickle.load(f)
            return FastImageNet(samples, transform=transform)
        return ImageFolder(os.path.join(args.data_path, split), transform=transform)

    train_dst = _make_dst("train", train_tf)
    val_dst = _make_dst("val", val_tf)
    calib_dst = _make_dst("train", calib_tf)   # clean transform on train images for σ,μ

    if use_ddp:
        train_sampler = RASampler(train_dst, args.world_size, args.rank, reps=args.ra_reps)
        val_sampler = DistributedSampler(val_dst, shuffle=False)
    else:
        train_sampler = None; val_sampler = None
    train_loader = DataLoader(train_dst, batch_size=args.train_batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dst, batch_size=args.val_batch_size, sampler=val_sampler,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    calib_loader = DataLoader(calib_dst, batch_size=args.val_batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, calib_loader, train_sampler


# --------------------------------------------------------------------------------------
# MixUp / CutMix (per batch) + EMA + optimizer (no wd on norm/bias) + lr schedule
# --------------------------------------------------------------------------------------
def mixup_cutmix(x, y, mixup_alpha, cutmix_alpha):
    """Return (x_mixed, y_a, y_b, lam). Soft loss = lam*CE(o,y_a)+(1-lam)*CE(o,y_b)."""
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return x, y, y, 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    if cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < 0.5):
        lam = float(torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample())
        H, W = x.shape[-2:]
        r = math.sqrt(1.0 - lam); cw, ch = int(W * r), int(H * r)
        cx, cy = random.randint(0, W), random.randint(0, H)
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (H * W))
    else:
        lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
        x = lam * x + (1.0 - lam) * x[perm]
    return x, y, y[perm], lam


class EMA:
    """Exponential moving average of params AND float buffers (BN stats)."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(m.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(m)

    def rebuild(self, model):
        """Re-init from a structurally-changed model (after reparam)."""
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)


def _collect_switch_sigma(mgr):
    """Per-layer input σ at the switch point. bn variant → sqrt(bn.running_var+eps); mean
    variant → the sigma_x buffer. σ drives the post-switch (grad/param)=plain/σ² inflation,
    so this is the data the switch-LR fix is calibrated from."""
    out = {}
    for name, rp in mgr._reparam_modules.items():
        bn = getattr(rp, "bn", None)
        if bn is not None and hasattr(bn, "running_var"):
            out[name] = torch.sqrt(bn.running_var + bn.eps).detach().cpu().clone()
        else:
            s = getattr(rp, "sigma_x", None)
            if s is not None:
                out[name] = s.detach().cpu().clone()
    return out


def _install_switch_precond(mgr):
    """S1 fix: cancel the post-switch 1/σ² LR inflation by scaling each v_tilde gradient by σ².

    After the switch the trainable weight is v_tilde=σW. By chain rule grad_vtilde=grad_W/σ,
    so the per-step RELATIVE move (grad/param) inflates by 1/σ² vs plain coords — same LR then
    overshoots ~50× (×∞ for dead channels) and destroys the net in one epoch. Multiplying
    grad_vtilde by σ² restores the plain-coords relative step exactly:
        lr·(σ²·grad_vtilde)/v_tilde = lr·(σ²·grad_W/σ)/(σW) = lr·grad_W/W   (plain).
    σ² is FROZEN at switch from bn.running_var (already broadcast → identical across DDP ranks,
    so every rank scales identically). Dead channels (σ²→0) get a ~0 factor → frozen at their
    function-preserving value (they contribute ~nothing, so freezing is safe, not a collapse).
    Returns the hook handles (kept alive by the caller)."""
    handles = []
    for rp in mgr._reparam_modules.values():
        bn = getattr(rp, "bn", None)
        vt = getattr(rp, "v_tilde", None)
        if bn is None or vt is None or not hasattr(bn, "running_var"):
            continue
        sig2 = bn.running_var.detach().clone()                  # [in] frozen σ² at switch
        shape = [1, sig2.numel()] + [1] * (vt.dim() - 2)        # broadcast on in-dim
        factor = sig2.view(*shape)
        handles.append(vt.register_hook(lambda g, f=factor: g * f))
    return handles


def build_opt(model, lr, args):
    decay, no_decay = [], []
    for _n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)   # norm + bias (1-D) → no wd
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, momentum=args.momentum, nesterov=args.nesterov)


def lr_at(epoch, args):
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / max(1, args.warmup_epochs)
    t = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    return args.eta_min + 0.5 * (args.lr - args.eta_min) * (1 + math.cos(math.pi * t))


def train_one_epoch(model, loader, sampler, optimizer, device, epoch, args, criterion, ema=None):
    model.train()
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    n = len(loader) if not args.limit_batches else min(args.limit_batches, len(loader))
    log_every = max(1, n // 20)
    running = 0.0
    for i, (x, y) in enumerate(loader):
        if args.limit_batches and i >= args.limit_batches:
            break
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        x, ya, yb, lam = mixup_cutmix(x, y, args.mixup_alpha, args.cutmix_alpha)
        out = model(x)
        loss = lam * criterion(out, ya) + (1.0 - lam) * criterion(out, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)          # EMA must step EVERY batch (window 1/(1-decay) steps)
        running += loss.item()
        if is_main() and (i % log_every == 0 or i == n - 1):
            log_info(f"  [e{epoch+1} {i+1}/{n}] loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.4f}")
    return running / max(1, n)


@torch.no_grad()
def _eval(model, loader, device, args):
    """Top-1. Full val via vbp_common.validate, or a quick N-batch pass when --limit_batches."""
    if not args.limit_batches:
        return validate(model, loader, device, args.model_type)[0]
    model.eval(); correct = total = 0
    for i, (x, y) in enumerate(loader):
        if i >= args.limit_batches:
            break
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().item(); total += y.numel()
    return correct / max(total, 1)


def _unwrap(m):
    return m.module if isinstance(m, nn.parallel.DistributedDataParallel) else m


def main(argv):
    args = parse_args(argv[1:])
    use_ddp = not args.disable_ddp and "RANK" in os.environ
    if use_ddp:
        from normalize_net import setup_distributed
        device = setup_distributed(args)
    else:
        device = get_device(); args.rank = 0; args.world_size = 1; args.local_rank = 0
    setup_logging(args.save_dir)
    arm = "switch" if args.reparam_at_epoch >= 0 else "baseline"
    log_info("=" * 60)
    log_info(f"train_v2 (official recipe) arm={arm} "
             f"reparam_at_epoch={args.reparam_at_epoch} epochs={args.epochs}")
    log_info("=" * 60)
    for k, v in vars(args).items():
        log_info(f"  {k}: {v}")

    train_loader, val_loader, calib_loader, train_sampler = build_loaders(args, use_ddp)
    model = load_model(args, device)                       # random init (no --checkpoint)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = build_opt(model, lr_at(0, args), args)
    ema = EMA(_unwrap(model), args.ema_decay) if args.ema_decay > 0 else None
    train_model = (nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
                   if use_ddp else model)
    mgr = None
    write_run(args, {"status": "running", "arm": arm, "config": vars(args)})

    for epoch in range(args.epochs):
        # ---- continuous switch to normalized coordinates ----
        if mgr is None and args.reparam_at_epoch >= 0 and epoch == args.reparam_at_epoch:
            raw = _unwrap(train_model)
            # Save the PLAIN net right before normalizing. Only the weights are needed to
            # retry the switch with a different lr — optimizer is rebuilt and σ,μ are
            # re-calibrated at switch time (deterministic from these weights). Retry via:
            #   --checkpoint <preswitch> --reparam_at_epoch 0 --epochs <remaining> --lr <new>
            if is_main():
                os.makedirs(args.save_dir, exist_ok=True)
                pre = os.path.join(args.save_dir, f"{args.save_tag}_preswitch_e{epoch}.pth")
                torch.save({k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}, pre)
                log_info(f"saved pre-switch checkpoint → {pre}")
            names = build_whole_net_reparam_layers(
                raw, exclude_classifier=True, exclude_stem=args.exclude_stem)
            args.max_batches = args.calib_batches
            mgr = build_reparam_manager(raw, names, device, args)
            mgr.reparameterize(calib_loader)               # normalize (calibrate σ,μ)
            if use_ddp:
                _broadcast_model_state(raw)
            optimizer = build_opt(raw, lr_at(epoch, args), args)   # new params (v_tilde,m)
            if args.switch_precond:
                _switch_precond_handles = _install_switch_precond(mgr)
                log_info(f"switch σ² grad-preconditioner ON ({len(_switch_precond_handles)} "
                         f"layers) — cancels the 1/σ² LR inflation")
            if ema is not None:
                ema.rebuild(raw)                            # EMA stale after structural change
            train_model = (nn.parallel.DistributedDataParallel(raw, device_ids=[args.local_rank])
                           if use_ddp else raw)
            # Sidecar (rank 0): the per-layer σ at the switch + EMA/epoch/lr. σ is what drives
            # the post-switch dynamics — (grad/param) inflates by 1/σ², so this is the data the
            # switch-LR fix is designed from. Lets a fix be iterated from the checkpoint (load
            # → reparam → fix → a few epochs) instead of retraining to this epoch each time.
            if is_main():
                sig = _collect_switch_sigma(mgr)
                allsig = torch.cat([s.flatten() for s in sig.values()]) if sig else torch.zeros(1)
                meta = os.path.join(args.save_dir, f"{args.save_tag}_preswitch_e{epoch}_meta.pt")
                torch.save({"epoch": epoch, "lr": lr_at(epoch, args), "arch": args.cnn_arch,
                            "ema_state": (ema.shadow.state_dict() if ema is not None else None),
                            "sigma_per_layer": sig,
                            "sigma_summary": {"median": float(allsig.median()),
                                              "p10": float(allsig.quantile(0.1)),
                                              "min": float(allsig.min()),
                                              "frac_below_0.05": float((allsig < 0.05).float().mean())}},
                           meta)
                log_info(f"SWITCH σ: median={allsig.median():.3f} p10={allsig.quantile(0.1):.3f} "
                         f"frac(σ<0.05)={float((allsig<0.05).float().mean()):.2f} → 1/σ² LR inflation "
                         f"median≈{float((1.0/allsig.clamp(min=1e-3)**2).median()):.0f}× (saved {meta})")
            log_info(f"SWITCHED to normalized coords at epoch {epoch} "
                     f"({len(names)} layers; optimizer + EMA rebuilt)")

        for g in optimizer.param_groups:
            g["lr"] = lr_at(epoch, args)
        train_loss = train_one_epoch(train_model, train_loader, train_sampler,
                                     optimizer, device, epoch, args, criterion, ema=ema)

        if is_main():
            acc = _eval(_unwrap(train_model), val_loader, device, args)
            ema_acc = (_eval(ema.shadow, val_loader, device, args)
                       if ema is not None else None)
            log_info(f"[{arm}] Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f} "
                     f"val_acc={acc:.4f}" + (f" ema_acc={ema_acc:.4f}" if ema_acc else "")
                     + f" lr={lr_at(epoch, args):.4f}"
                     + (" [normalized]" if mgr is not None else ""))
            append_metrics(args, {"arm": arm, "epoch": epoch + 1, "epochs": args.epochs,
                                  "train_loss": round(train_loss, 6), "val_acc": round(acc, 6),
                                  "ema_val_acc": (round(ema_acc, 6) if ema_acc else None),
                                  "lr": lr_at(epoch, args),
                                  "normalized": mgr is not None})
        if use_ddp:
            torch.distributed.barrier()

    # ---- merge normalized net back + save (rank 0) ----
    if mgr is not None and mgr.is_active:
        mgr.merge_back()                                   # raw model: collective, all ranks
    if is_main():
        from ckpt import save_ckpt, merge_reparam_modules
        os.makedirs(args.save_dir, exist_ok=True)
        raw = _unwrap(train_model)
        # Switch arm: the EMA shadow is still in reparam'd (BNResidual) form — merge it to
        # plain before saving so the deployable EMA checkpoint loads as a standard net.
        if ema is not None and mgr is not None:
            merge_reparam_modules(ema.shadow)
        final = _eval(raw, val_loader, device, args)
        ema_final = (_eval(ema.shadow, val_loader, device, args) if ema is not None else None)
        path = os.path.join(args.save_dir, f"{args.save_tag}.pth")
        # Bundle: raw (trajectory endpoint) + EMA (the model you report/deploy — 80.86 is EMA).
        save_ckpt(path, raw, kind="ema-trained", arch=args.cnn_arch,
                  ema_model=(ema.shadow if ema is not None else None),
                  meta={"arm": arm, "epochs": args.epochs,
                        "val_acc": final, "ema_val_acc": ema_final})
        log_info(f"DONE arm={arm}: final acc={final:.4f}"
                 + (f" ema={ema_final:.4f} (← reported)" if ema_final else "")
                 + f" → {path} (load_ckpt prefer='ema')")
        write_run(args, {"status": "done", "arm": arm, "config": vars(args),
                         "final_val_acc": final, "ema_final_val_acc": ema_final,
                         "checkpoint": path, "reported_metric": "ema_val_acc"})
    if use_ddp:
        from normalize_net import cleanup
        cleanup()
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description="official v2 recipe + optional normalize switch")
    p.add_argument("--model_type", default="cnn"); p.add_argument("--model_name", default="resnet50")
    p.add_argument("--cnn_arch", default="resnet50")
    p.add_argument("--checkpoint", default=None, help="None → from scratch")
    p.add_argument("--data_path", required=True)
    p.add_argument("--epochs", type=int, default=100)
    # official recipe knobs
    p.add_argument("--lr", type=float, default=0.5, help="for batch 1024; scale to your batch")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true", default=True)
    p.add_argument("--wd", type=float, default=2e-5)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--eta_min", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--cutmix_alpha", type=float, default=1.0)
    p.add_argument("--random_erase", type=float, default=0.1)
    p.add_argument("--ema_decay", type=float, default=0.9998, help="0 = no EMA")
    p.add_argument("--ra_reps", type=int, default=4)
    p.add_argument("--train_crop", type=int, default=176)
    p.add_argument("--val_resize", type=int, default=232)
    p.add_argument("--train_batch_size", type=int, default=128)
    p.add_argument("--val_batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    # normalize switch
    p.add_argument("--reparam_at_epoch", type=int, default=-1, help="-1 = never (baseline)")
    p.add_argument("--reparam_variant", default="bn", choices=["bn", "mean"])
    p.add_argument("--norm_bn_momentum", type=float, default=0.01)
    p.add_argument("--reparam_lambda", type=float, default=0.0)
    p.add_argument("--mu_ema_momentum", type=float, default=0.0)
    p.add_argument("--exclude_stem", action="store_true")
    p.add_argument("--switch_precond", action="store_true",
                   help="S1 fix: scale post-switch v_tilde grads by σ² to cancel the 1/σ² LR "
                        "inflation that otherwise collapses the net the epoch after the switch.")
    p.add_argument("--calib_batches", type=int, default=50)
    # io / ddp
    p.add_argument("--save_dir", required=True); p.add_argument("--save_tag", default="v2")
    p.add_argument("--disable_ddp", action="store_true")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--limit_batches", type=int, default=0,
                   help="cap train + val batches per epoch (0=full). Use 2-3 for a fast "
                        "functionality check.")
    p.add_argument("--local_rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", 0)))
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
