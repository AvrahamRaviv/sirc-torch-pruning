"""
Normalize-net-by-construction: transform + verify (no training).

Applies a function-preserving reparametrization across the whole network: for every
nn.Linear and nn.Conv2d(groups==1), fold the input mean into the bias (m = b + W·μ) and
train a residual weight. Two parametrizations, via --reparam_variant:
  - mean (default, CORRECT): trainable v = W, forward v·(x−μ) + m. Gradient ∝ (x−μ),
    uniform W-space lr → fine-tunes like the plain baseline. (MeanResidualManager)
  - bn   (legacy): insert BN(affine=False), trainable v_tilde = W·σ on normalized input.
    Forward divides back by σ → effective W-space lr = lr/σ², a per-channel overshoot that
    caused the v1–v6 converged-FT accuracy cliffs. (NormalizedResidualManager)
See torch_pruning/utils/reparam.py.

This script only transforms and verifies — a general-training run is the next step and
resumes from the saved 'vnr' artifact. Two checkpoints are written:
  - <tag>_vnr.pth          : pre-merge state — preserves v / m / mu_x / sigma_x /
                              sigma_out_x (Mean variant) or v_tilde / m / bn.* (BN
                              variant). Reload via _merge_vnr_state_dict folds back to
                              plain weight/bias. (Training-resume into a fresh Mean
                              variant manager would need a separate helper — not wired.)
  - <tag>_merged_biased.pth: post-merge (standard weight / bias) — canonical, verified.
Each has a .meta.json sidecar so torch.load(weights_only=True) stays usable.

Usage (single device, MPS auto-selected on Apple Silicon):
    python benchmarks/vbp/normalize_net.py \
        --model_type convnext --model_name convnext_tiny \
        --checkpoint /path/to/convnext_tiny.pth \
        --data_path /path/to/imagenet --disable_ddp --num_workers 0
"""

import argparse
import copy
import json
import logging
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch_pruning as tp

try:
    from .vbp_common import (
        logger, is_main, log_info, setup_logging,
        setup_distributed, cleanup,
        build_dataloaders, load_model, validate, forward_logits,
        train_one_epoch, build_ft_scheduler,
        build_whole_net_reparam_layers, attach_biases, _merge_vnr_state_dict,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vbp_common import (
        logger, is_main, log_info, setup_logging,
        setup_distributed, cleanup,
        build_dataloaders, load_model, validate, forward_logits,
        train_one_epoch, build_ft_scheduler,
        build_whole_net_reparam_layers, attach_biases, _merge_vnr_state_dict,
    )

from torch_pruning.utils.reparam import (
    NormalizedResidualManager, MeanResidualManager, _channel_group_norm,
)


def build_reparam_manager(model, layer_names, device, args):
    """Construct the reparam manager for the chosen --reparam_variant.

    CANONICAL — bn (the "BN trick", boss spec): normalize input by the LIVE EMA σ
        (BN(affine=False), running stats), trainable v_tilde = σ_cal·W (σ-scaling applied
        ONCE at init, frozen into v_tilde), fold μ into the bias. ‖v_tilde‖ = the
        contribution score, so plain weight-decay on v_tilde = the pruning regularizer.
        Gradient w.r.t v_tilde is on unit-variance normalized input → well-scaled (no
        1/σ² overshoot; that earlier diagnosis assumed a W-trainable / σ-divide impl,
        not this v_tilde-trainable one). bn_momentum controls the σ EMA.

    ABLATION — mean: trainable v = W, forward v·(x−μ) + m. σ is ABSENT from the forward
        (input not normalized); σ only a detached factor in the ‖σ·v‖ score / aux reg.
        Use to ablate the input-normalization; not the boss spec.
    """
    lam = float(getattr(args, "reparam_lambda", 0.0))
    variant = getattr(args, "reparam_variant", "bn")
    if variant == "mean":
        mu_ema = float(getattr(args, "mu_ema_momentum", 0.0))
        log_info("--reparam_variant=mean (ABLATION): σ NOT in the forward, input not "
                 "normalized — σ is only a detached score factor. Boss spec is the BN "
                 "trick (--reparam_variant=bn): normalize input by live-EMA σ + trainable "
                 "v_tilde=σW. Use mean only to ablate input-normalization.")
        return MeanResidualManager(
            model, layer_names, device, lambda_reg=lam, max_batches=args.max_batches,
            ema_momentum=mu_ema)
    bn_mom = getattr(args, "norm_bn_momentum", None)
    bn_mom = 0.1 if bn_mom is None else float(bn_mom)
    log_info(f"--reparam_variant=bn (CANONICAL, BN trick): input normalized by live-EMA "
             f"σ (momentum={bn_mom}), trainable v_tilde=σ·W, μ folded to bias. "
             f"WD on v_tilde = contribution-score regularizer.")
    return NormalizedResidualManager(
        model, layer_names, device, lambda_reg=lam, max_batches=args.max_batches,
        bn_momentum=bn_mom)


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


# ---------------------------------------------------------------------------
# Structured logging (machine-parseable artifacts for later parse / ExpHandler)
# ---------------------------------------------------------------------------
def _metrics_path(args):
    return os.path.join(args.save_dir, f"{args.save_tag}_metrics.jsonl")


def _run_path(args):
    return os.path.join(args.save_dir, f"{args.save_tag}_run.json")


def append_metrics(args, record):
    """Append one JSON line (one training epoch) to <save_tag>_metrics.jsonl."""
    os.makedirs(args.save_dir, exist_ok=True)
    with open(_metrics_path(args), "a") as f:
        f.write(json.dumps(record) + "\n")


def write_run(args, payload):
    """Write/overwrite the run summary JSON (config + final results)."""
    os.makedirs(args.save_dir, exist_ok=True)
    with open(_run_path(args), "w") as f:
        json.dump(payload, f, indent=2)


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
# Training
# ---------------------------------------------------------------------------
def build_optimizer(model, args, mgr=None):
    """Optimizer with weight decay ON the residual weight (free-by-design reg).

    Only m (the folded mean/bias term) is excluded from decay; the residual weight
    stays in the decayed group.
      - mean variant: WD on ‖v‖ = ‖W‖² (magnitude — same geometry as plain).
      - bn   variant: WD on ‖v_tilde‖ = ‖W·σ‖² (contribution variance / NCI).
    (Deliberately NOT reusing mgr.reparam_param_ids(), which lumps the weight + m
    together — that would zero WD on the weight, the opposite of what we want here.)
    """
    params = list(model.parameters())
    active = mgr is not None and mgr.is_active
    # σ²-lr-scaling only exists to undo the bn variant's 1/σ² overshoot (needs v_tilde
    # + bn). The mean variant already has uniform W-space lr, so it's a conceptual no-op.
    # Detect the variant from the modules, not args — single source of truth.
    is_bn_variant = active and all(
        hasattr(rp, "v_tilde") for rp in mgr._reparam_modules.values())
    scale_by_sigma2 = getattr(args, "lr_scale_by_sigma2", False)
    if scale_by_sigma2 and not is_bn_variant:
        log_info("lr_scale_by_sigma2 ignored: only applies to the bn reparam variant")
        scale_by_sigma2 = False

    if not active:
        groups = [{"params": params, "weight_decay": args.wd}]
    elif scale_by_sigma2:
        # Neutralize the 1/σ² W-space rescaling (bn variant): SGD on v_tilde gives
        # effective W-space lr = lr/σ², so per-layer lr ∝ mean σ² recovers a uniform W-step.
        # Per-LAYER (mean σ²) — corrects cross-layer scale, not intra-layer channel spread.
        v_ids = {id(rp.v_tilde) for rp in mgr._reparam_modules.values()}
        m_ids = {id(rp.m) for rp in mgr._reparam_modules.values()}
        groups, scales = [], []
        for rp in mgr._reparam_modules.values():
            sigma2 = (rp.bn.running_var + rp.bn.eps).mean().item()
            groups.append({"params": [rp.v_tilde], "lr": args.lr * sigma2,
                           "weight_decay": args.wd})
            scales.append(sigma2)
        groups.append({"params": [rp.m for rp in mgr._reparam_modules.values()],
                       "lr": args.lr, "weight_decay": 0.0})
        rest = [p for p in params if id(p) not in v_ids and id(p) not in m_ids]
        if rest:
            groups.append({"params": rest, "lr": args.lr, "weight_decay": args.wd})
        if is_main() and scales:
            import numpy as _np
            s = _np.array(scales)
            log_info(f"lr_scale_by_sigma2: {len(scales)} layers, σ² "
                     f"min={s.min():.3g} median={_np.median(s):.3g} max={s.max():.3g} "
                     f"→ per-layer lr {args.lr*s.min():.2e}..{args.lr*s.max():.2e}")
    else:
        m_ids = {id(rp.m) for rp in mgr._reparam_modules.values()}
        decayed = [p for p in params if id(p) not in m_ids]   # includes residual weight v
        no_decay = [p for p in params if id(p) in m_ids]
        groups = [
            {"params": decayed, "weight_decay": args.wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    if args.opt == "sgd":
        return torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum)
    return torch.optim.AdamW(groups, lr=args.lr)


def train_normalized(model, mgr, train_loader, val_loader, train_sampler,
                     args, device, use_ddp, teacher=None):
    """Short training of the (optionally normalized) model. Returns best val acc.

    Normalized arm (mgr active): WD acts on v_tilde; mgr stays active across all epochs
    so BN(affine=False) tracks live stats by EMA — no manual refresh. Baseline arm
    (mgr=None): plain training with standard WD, for an apples-to-apples comparison.
    teacher: if given and args.use_kd, train_one_epoch applies KD.
    """
    normalized = mgr is not None and mgr.is_active
    optimizer = build_optimizer(model, args, mgr)
    scheduler, step_per_batch = build_ft_scheduler(
        optimizer, args.epochs, len(train_loader),
        eta_min=args.ft_eta_min, warmup_epochs=args.ft_warmup_epochs)

    train_model = model
    if use_ddp:
        train_model = DDP(model, device_ids=[args.local_rank],
                          output_device=args.local_rank)

    phase = "NormTrain" if normalized else "Baseline"
    arm = "normalized" if normalized else "baseline"
    # Contribution-score regularizer (M1+). Active only when the manager exists,
    # is active, and λ > 0. Bare closure so the closure captures the mgr; reads
    # current v / σ_x each call (and σ_x.detach() inside the manager method).
    aux_loss_fn = None
    lam_train = float(getattr(args, "reparam_lambda", 0.0))
    if normalized and lam_train > 0:
        aux_loss_fn = lambda: mgr.regularization_loss()
        variant = getattr(args, "reparam_variant", "mean")
        log_info(f"Contribution-score regularizer ACTIVE: λ={lam_train} "
                 f"({'‖σ·v‖' if variant == 'mean' else '‖v_tilde‖'}). "
                 f"Aux loss added every step.")
    elif normalized:
        log_info("Contribution-score regularizer disabled (--reparam_lambda 0). "
                 "Mean-variant training is plain CE/KD only.")

    # DDP: with M4 per-step EMA each rank diverges μ_x / σ_x / σ_out_x from its
    # local shard. Sync once per epoch to keep ranks coherent. Cheap (few
    # all-reduces of small vectors) compared to per-step sync.
    ddp_active = use_ddp and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    sync_each_epoch = (normalized and lam_train >= 0 and ddp_active and
                       getattr(args, "mu_ema_momentum", 0.0) > 0)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, aux = train_one_epoch(
            train_model, train_loader, train_sampler, optimizer, scheduler,
            device, epoch, args, teacher=teacher,
            step_per_batch=step_per_batch, phase=phase,
            aux_loss_fn=aux_loss_fn)

        if sync_each_epoch:
            mgr.sync_ema_buffers()

        if is_main():
            eval_model = train_model.module if isinstance(train_model, DDP) else train_model
            acc, val_loss = validate(eval_model, val_loader, device, args.model_type)
            best_acc = max(best_acc, acc)
            cur_lr = optimizer.param_groups[0]["lr"]
            # reg_loss = the λ‖σ·v‖ aux term (avg per step); CE/KD = train_loss − reg.
            reg_loss = float(aux.get("aux", 0.0)) if aux else 0.0
            # During the reg phase, append the ‖ṽ‖ sparsity so the reg effect is visible live:
            # reg_loss (the λ‖σv‖ term) shrinking + frac_below growing = channels going prunable.
            reg_str = ""
            if normalized:
                _s = mgr.vnorm_summary()
                if _s:
                    reg_str = (f", reg_loss={reg_loss:.3e}, vnorm_mean={_s['mean']:.3e}, "
                               f"frac‖ṽ‖<0.1={_s['frac_below_0.1']:.3f}")
            # Human line (capital "Epoch", comma-separated → parse_logs-compatible style)
            log_info(f"[{phase}] Epoch {epoch+1}/{args.epochs}: "
                     f"train_loss={train_loss:.4f}, val_acc={acc:.4f}, "
                     f"val_loss={val_loss:.4f}, best={best_acc:.4f}, lr={cur_lr:.2e}{reg_str}")
            # Machine line (one JSON object per epoch). reg_loss + the ‖σ·v‖ sparsity
            # summary let parse_normnet plot the λ-sweep natural-sparsity surface.
            rec = {
                "arm": arm, "epoch": epoch + 1, "epochs": args.epochs,
                "train_loss": round(train_loss, 6), "val_acc": round(acc, 6),
                "val_loss": round(val_loss, 6), "best_val_acc": round(best_acc, 6),
                "lr": cur_lr,
                "reg_loss": round(reg_loss, 8),
                "ce_kd_loss": round(train_loss - reg_loss, 6),
            }
            if normalized:
                summ = mgr.vnorm_summary()
                if summ:
                    rec["vnorm_mean"] = round(summ["mean"], 6)
                    rec["vnorm_min"] = round(summ["min"], 6)
                    rec["vnorm_max"] = round(summ["max"], 6)
                    rec["frac_below_0.01"] = round(summ["frac_below_0.01"], 6)
                    rec["frac_below_0.1"] = round(summ["frac_below_0.1"], 6)
            append_metrics(args, rec)
            if normalized:
                mgr.log_channel_stats(verbose=False)
            # Periodic salvage checkpoint (rank 0). Dumps the ACTIVE reparam state as a
            # vnr ckpt — no merge_back (destructive + collective), just torch.save. Lets a
            # hung/crashed long run be recovered instead of lost. Overwrites each interval;
            # the end-of-training save writes the final vnr/merged pair. Skip the last
            # epoch (the final save covers it).
            ckpt_interval = int(getattr(args, "ckpt_interval", 0))
            if (normalized and ckpt_interval > 0 and (epoch + 1) % ckpt_interval == 0
                    and (epoch + 1) < args.epochs):
                resolved = list(mgr._reparam_modules.keys())
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                save_normnet_checkpoint(state, "vnr", resolved, args)
                log_info(f"Periodic salvage vnr checkpoint @ epoch {epoch+1}")
        if use_ddp:
            dist.barrier()
    return best_acc


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
    # Route torch_pruning.utils.reparam logs (the per-epoch "V-norm aggregate"
    # health line) into the same handlers as vbp_imagenet.log. That logger has
    # propagate=False and no handler of its own, so without this bridge the
    # V-norm INFO lines are silently dropped (and never reach the log file).
    _rlog = logging.getLogger("torch_pruning.utils.reparam")
    _rlog.setLevel(logging.INFO)
    _rlog.handlers = list(logging.getLogger("vbp_imagenet").handlers)
    _rlog.propagate = False
    mode = "transform + verify" if args.epochs == 0 else f"short training ({args.epochs}e)"
    if is_main():
        log_info("=" * 60)
        log_info(f"Normalize-net-by-construction ({mode})")
        log_info("=" * 60)
        for k, v in vars(args).items():
            logger.info(f"  {k}: {v}")
        log_info(f"Device: {device}")

    train_loader, val_loader, train_sampler = build_dataloaders(args, use_ddp=use_ddp)
    model = load_model(args, device)

    # -- Transform + verify only (no training) --
    if args.epochs == 0:
        layer_names = build_whole_net_reparam_layers(
            model, exclude=args.exclude_layers,
            exclude_classifier=args.exclude_classifier, exclude_stem=args.exclude_stem)
        log_info(f"Whole-net selector: {len(layer_names)} candidate layers")
        mgr = build_reparam_manager(model, layer_names, device, args)
        ok = verify_roundtrip(model, mgr, val_loader, args, device, use_ddp)
        if use_ddp:
            cleanup()
        return 0 if ok else 1

    # -- Short training: normalized arm vs --no_reparam baseline arm --
    arm = "baseline" if args.no_reparam else "normalized"
    acc0 = None
    base_macs = base_params = None
    if is_main():
        log_info("command: " + " ".join([sys.executable, "benchmarks/vbp/normalize_net.py"] + argv))
        try:
            ex = torch.randn(1, 3, 224, 224).to(device)
            base_macs, base_params = tp.utils.count_ops_and_params(model, ex)
            log_info(f"Baseline: {base_macs / 1e9:.2f}G MACs, {base_params / 1e6:.2f}M params")
        except Exception as e:  # MACs are informational; never block the run
            log_info(f"MACs count skipped: {e}")
        acc0, _ = validate(model, val_loader, device, args.model_type)
        log_info(f"Pre-train val_acc={acc0:.4f}")
        write_run(args, {
            "arm": arm, "status": "running", "config": vars(args),
            "pre_train_val_acc": acc0,
            "macs_g": base_macs / 1e9 if base_macs else None,
            "params_m": base_params / 1e6 if base_params else None,
        })

    # KD teacher = frozen plain pretrained net (snapshot BEFORE reparam).
    teacher = None
    if args.use_kd:
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        log_info(f"KD enabled: teacher = frozen pretrained (alpha={args.kd_alpha}, T={args.kd_T})")

    mgr = None
    if args.no_reparam:
        log_info("Baseline arm: plain training, no normalization")
        # R7 UX: warn on silently-dropped reparam-only flags so misconfigs surface.
        if float(getattr(args, "reparam_lambda", 0.0)) > 0:
            log_info(f"WARNING: --reparam_lambda={args.reparam_lambda} ignored under "
                     f"--no_reparam (no reparam manager → no contribution-score regularizer).")
        if float(getattr(args, "mu_ema_momentum", 0.0)) > 0:
            log_info(f"WARNING: --mu_ema_momentum={args.mu_ema_momentum} ignored under "
                     f"--no_reparam (no Mean modules to EMA).")
    else:
        layer_names = build_whole_net_reparam_layers(
            model, exclude=args.exclude_layers,
            exclude_classifier=args.exclude_classifier, exclude_stem=args.exclude_stem)
        log_info(f"Whole-net selector: {len(layer_names)} candidate layers")
        if not layer_names:
            log_info("WARNING: build_whole_net_reparam_layers returned EMPTY list "
                     "(check --exclude_layers / --exclude_classifier / --exclude_stem). "
                     "No layers will be reparameterized.")
        mgr = build_reparam_manager(model, layer_names, device, args)
        mgr.reparameterize(train_loader)
        if args.norm_bn_momentum is not None:
            if args.reparam_variant == "bn":
                for rp in mgr._reparam_modules.values():
                    rp.bn.momentum = args.norm_bn_momentum
                tag = "frozen" if args.norm_bn_momentum == 0 else f"momentum={args.norm_bn_momentum}"
                log_info(f"Inserted-BN running stats set {tag} "
                         f"(σ {'fixed at calibration' if args.norm_bn_momentum == 0 else 'slow EMA'})")
            else:
                log_info("norm_bn_momentum ignored: only applies to --reparam_variant bn "
                         "(mean variant has no inserted BN; μ_x is frozen at calibration)")
        if use_ddp:
            _broadcast_model_state(model)
        log_info(f"Reparameterized {len(mgr._reparam_modules)} layers (WD acts on v_tilde)")

        # Log calibrated σ_x spread + baseline ‖σ·v‖ range — sanity-check for the
        # contribution-score path (mean variant only; bn variant carries σ inside v_tilde).
        if is_main() and args.reparam_variant == "mean":
            sigmas = []
            contrib_norms = []
            for rp in mgr._reparam_modules.values():
                s = rp.sigma_x.detach()
                v = rp.v.detach()
                if v.dim() == 4:
                    w_eff = v * s[None, :, None, None]
                else:
                    w_eff = v * s[None, :]
                sigmas.append(s.flatten())
                contrib_norms.append(_channel_group_norm(w_eff, mgr.norm_dim))
            import torch as _torch
            s_all = _torch.cat(sigmas)
            n_all = _torch.cat(contrib_norms)
            log_info(f"σ_x: min={s_all.min():.3g} median={s_all.median():.3g} "
                     f"max={s_all.max():.3g} ({len(mgr._reparam_modules)} layers); "
                     f"baseline ‖σ·v‖: min={n_all.min():.3g} max={n_all.max():.3g}")

        # Immediate post-reparam (init / "epoch 0") eval. Reparam is exact in eval mode,
        # so this MUST match pre-train acc0; a nonzero delta flags a wiring/DDP bug
        # before we commit to the full run.
        if is_main():
            acc_init, _ = validate(model, val_loader, device, args.model_type)
            log_info(f"Post-reparam (init) val_acc={acc_init:.4f} "
                     f"(Δ vs pre-train={acc_init - acc0:+.4f})")
            append_metrics(args, {
                "arm": arm, "epoch": 0, "epochs": args.epochs,
                "train_loss": None, "val_acc": round(acc_init, 6),
                "val_loss": None, "best_val_acc": round(acc_init, 6),
                "lr": None, "stage": "post_reparam_init",
            })

    best = train_normalized(model, mgr, train_loader, val_loader, train_sampler,
                            args, device, use_ddp, teacher=teacher)

    # -- Merge + save. merge_back() runs a COLLECTIVE all_reduce (_sync_bn_stats) so it
    # MUST be called on ALL ranks — never inside `if is_main()` (that deadlocks: rank 0
    # blocks in the collective while other ranks reach cleanup()). State-dict capture +
    # file write stay rank-0-only. --
    resolved = vnr_state = merged_state = None
    if mgr is not None and mgr.is_active:
        resolved = list(mgr._reparam_modules.keys())
        if is_main():
            vnr_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        mgr.merge_back()                       # ALL ranks (collective inside)
        if is_main():
            merged_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if is_main():
        ckpts = {}
        if merged_state is not None:
            ckpts["merged_biased"] = save_normnet_checkpoint(merged_state, "merged_biased", resolved, args)
            ckpts["vnr"] = save_normnet_checkpoint(vnr_state, "vnr", resolved, args)
        else:
            os.makedirs(args.save_dir, exist_ok=True)
            base_path = os.path.join(args.save_dir, f"{args.save_tag}_baseline.pth")
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       base_path)
            log_info(f"Saved baseline checkpoint to {base_path}")
            ckpts["baseline"] = base_path
        log_info(f"RESULT arm={arm}, epochs={args.epochs}, "
                 f"pre_train_val_acc={acc0:.4f}, best_val_acc={best:.4f}")
        write_run(args, {
            "arm": arm, "status": "done", "config": vars(args),
            "pre_train_val_acc": acc0, "best_val_acc": best,
            "macs_g": base_macs / 1e9 if base_macs else None,
            "params_m": base_params / 1e6 if base_params else None,
            "checkpoints": ckpts, "metrics_file": _metrics_path(args),
        })

    if use_ddp:
        cleanup()
    return 0


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
    parser.add_argument("--val_resize", type=int, default=232,
                        help="Val resize before 224 crop. 232 (default) matches "
                             "torchvision ResNet50 IMAGENET1K_V2 (80.858); use 256 for V1 weights")

    # Whole-net selection
    parser.add_argument("--exclude_layers", nargs="*", default=None,
                        help="Dotted module names to exclude from reparam")
    parser.add_argument("--exclude_classifier", action="store_true", default=True)
    parser.add_argument("--include_classifier", dest="exclude_classifier",
                        action="store_false",
                        help="Also reparameterize the logits head")
    parser.add_argument("--exclude_stem", action="store_true", default=False)

    # Training (epochs=0 -> transform+verify only; >0 -> short training)
    parser.add_argument("--epochs", type=int, default=0,
                        help="Training epochs (0 = transform+verify only)")
    parser.add_argument("--no_reparam", action="store_true", default=False,
                        help="Baseline arm: plain training, skip normalization")
    parser.add_argument("--reparam_variant", default="bn", choices=["mean", "bn"],
                        help="Reparam parametrization (normalized arm). 'bn' (default, "
                             "CANONICAL = boss's BN trick): normalize input by live-EMA σ, "
                             "trainable v_tilde=σ·W (σ frozen-in at init), μ folded to bias; "
                             "‖v_tilde‖ = contribution score, WD on it = pruning reg. "
                             "'mean' (ABLATION): trainable v=W, forward v·(x−μ)+m, σ absent "
                             "from forward (input NOT normalized) — σ only a detached score "
                             "factor. Use mean to ablate input-normalization.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.05,
                        help="Weight decay (acts ON v_tilde in the normalized arm)")
    parser.add_argument("--opt", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--reparam_lambda", type=float, default=0.0,
                        help="M1+: λ for the contribution-score regularizer ‖σ·v‖ "
                             "(mean variant) / ‖v_tilde‖ (bn variant). 0 (default) "
                             "disables the auxiliary loss term. Plumbed into "
                             "build_reparam_manager(lambda_reg=...) AND added to the "
                             "training loss as an aux_loss_fn each step.")
    parser.add_argument("--mu_ema_momentum", type=float, default=0.0,
                        help="M4: per-step EMA momentum for μ_x / σ_x / σ_out_x inside "
                             "Mean modules. 0 (default) = frozen at calibration. "
                             "Recommended for from-scratch / long training: 0.01 (slow). "
                             "Faster than 0.01 risks aug-driven drift (see runs v3/v6). "
                             "Mean variant only; bn variant uses BN's own momentum.")
    parser.add_argument("--lr_scale_by_sigma2", action="store_true", default=False,
                        help="Scale each reparam layer's SGD lr by its mean σ² (from "
                             "calibration BN running_var) → neutralizes the 1/σ² W-space "
                             "rescaling. Per-layer (not per-channel). Use with cosine, not flat.")
    parser.add_argument("--norm_bn_momentum", type=float, default=None,
                        help="Override inserted-BN EMA momentum after calibration. "
                             "0=freeze σ at calibration (conv: true freeze), "
                             "0.01=slow/stable EMA, unset=PyTorch default 0.1 (live).")
    parser.add_argument("--ft_eta_min", type=float, default=1e-5)
    parser.add_argument("--ft_warmup_epochs", type=float, default=0)
    parser.add_argument("--use_kd", action="store_true", default=False)
    parser.add_argument("--kd_alpha", type=float, default=0.7)
    parser.add_argument("--kd_T", type=float, default=2.0)

    # Verify / output
    parser.add_argument("--verify_atol", type=float, default=1e-4)
    parser.add_argument("--validate", action="store_true", default=False,
                        help="Also run full-val accuracy delta (slow)")
    parser.add_argument("--save_tag", default="normnet")
    parser.add_argument("--save_dir", default="./output/normalize_net")
    parser.add_argument("--ckpt_interval", type=int, default=0,
                        help="Save a salvage vnr checkpoint every N epochs during training "
                             "(rank 0, non-destructive, overwrites). 0=off. Guards long "
                             "runs against crash/hang loss.")

    # DDP
    parser.add_argument("--disable_ddp", action="store_true")
    parser.add_argument("--local_rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", 0)))

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
