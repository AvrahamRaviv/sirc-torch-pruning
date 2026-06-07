"""
Generate the propagation-criterion (I) pruning experiments — ALL on normnet_main (DDP),
ALL from the v2 (80.86) lineage, targeting a fixed MAC budget. run_ddp.py fashion.

Hero criterion = the PDF's PROPAGATION importance I^l = W̄^l·I^{l+1} (--scorer propagation),
NOT NCI. Both PDF derivations are swept:
  relative     W̄ = M^p·D  (columns normalized → within-layer/local)          [default]
  non-relative W̄ = M^p     (raw product, no column-norm → cross-layer, compounds)

Carries the two fixes that sabotaged the earlier run (both now on by default):
  P1  σ calibrated on CLEAN center-crop images (was the scale-0.08 augmented loader →
      distorted the normalized weights every score is built on).
  P2  --imp_normalizer none: keep raw cross-layer-comparable scores. The old per-group
      mean-1 normalization erased the global scale BEFORE the global threshold, collapsing
      global pruning to per-layer-uniform and nullifying non-relative's σ_out^p. nonrel was
      never actually tested as global until now.

MAC target (not channel ratio): --mac_target_g 2.0 → normnet_main binary-searches the
global ratio that hits 2 GMAC. On RN50 (4.1G dense) 2G ≈ 30% channels, NOT 50% — channel
ratio ≠ MAC ratio, so the ratio knob would be misleading.

Two routes, same DDP path (normnet_main loads either ckpt kind + prunes-by-I + FT):

  OPTION B — reuse the COMPLETED RN_bn λ-sweep vnr ckpts (30-epoch bn-trick sparse runs).
    normnet_main auto-detects the *_vnr.pth via its .meta.json sidecar (merged_biased is
    broken on the cluster). No reg recompute (--epochs_norm_ft 0); just normalize→prune→FT.

  OPTION A — self-contained from the dense 80.86: short λ-reg norm-ft (--epochs_norm_ft 5
    --reparam_lambda 1e-4) → prune-by-I → FT.

Both reparam the MEAN variant at scoring time (propagation needs σ_out branch weighting);
reparam-at-prune is a cold calibration on the loaded weights, independent of how the ckpt
was trained. No magnitude baseline here (we focus v2 + the I criterion; the v2 magnitude
number, if needed, we produce separately).

Run:
    python benchmarks/vbp/generate_prune_I_experiments.py
    python run_ddp.py --out_dir_name B_prop_rel_l1e-4     # submit one (repeat per folder)
"""
import os
import stat

BASE_OUT = os.environ.get("PRUNE_BASE_OUT", "/algo/NetOptimization/outputs/NORMNET/ResNet50")
REPO = os.environ.get("EXP_REPO", "/home/avrahamra/PycharmProjects/sirc-torch-pruning")
NORMNET = f"{REPO}/benchmarks/vbp/normnet_main.py"
DATA = "/algo/NetOptimization/outputs/VBP/"
NPROC = int(os.environ.get("EXP_NPROC", "4"))

DENSE_8086 = os.environ.get(
    "CKPT", "/algo/NetOptimization/outputs/VBP/ResNet50_TP/resnet50_imagenet1k.pth")
MAC_TARGET_G = float(os.environ.get("MAC_TARGET_G", "2.0"))   # GMAC budget
FT = int(os.environ.get("FT_EPOCHS", "30"))                    # post-prune FT (DDP)
A_REG = int(os.environ.get("A_REG_EPOCHS", "10"))             # Option A λ-reg norm-ft epochs
A_LAMBDA = os.environ.get("A_LAMBDA", "1e-4")                 # λ on ‖ṽ‖ (sweepable)
A_MU_EMA = os.environ.get("A_MU_EMA", "0.1")                  # σ/μ EMA momentum → LIVE σ in reg

# Which arms to emit. Default: Option A relative only (the viable global criterion). nonrel is
# parked (raw M^p compounds ~1e9 by depth → global sort ≈ by depth, not contribution). A0 (cold,
# no reg) already characterized. Flip these on to regenerate them.
INCLUDE_NONREL = os.environ.get("INCLUDE_NONREL", "0") != "0"
INCLUDE_A = os.environ.get("INCLUDE_A", "0") != "0"   # Option A (λ-reg sparse phase) prop arms
INCLUDE_A0 = os.environ.get("INCLUDE_A0", "0") != "0"
# NCI = the bounded cross-layer one-hop scorer (‖σv‖=√NCI = --scorer per_layer): σ folded
# once, NO propagation, NO depth compound. The unbiased cross-layer baseline vs prop rel/nonrel.
INCLUDE_NCI = os.environ.get("INCLUDE_NCI", "0") != "0"
# Magnitude = the classical GroupMagnitudeImportance (L2) baseline, same harness (--scorer
# magnitude). The known-good control NCI/prop must beat.
INCLUDE_MAGNITUDE = os.environ.get("INCLUDE_MAGNITUDE", "0") != "0"
# NCI with native BN-fold OFF — isolate whether BN-fold drives the depth-profile gap vs the
# proven old tp_variance (which did NOT fold BN). Same as A0_nci otherwise (per_layer, mean
# normalizer, interior_only). DEFAULT ON (the only arm emitted by a bare run).
# ===================================================================================
# WHICH ARMS TO EMIT — edit these True/False directly (env var still overrides if set).
# Currently ON: tp_variance, nci_fbn_full, nci_fbn_bc, nci_fbn_bc_nobn.
# ===================================================================================
INCLUDE_NCI_FBN = os.environ.get("INCLUDE_NCI_FBN", "0") != "0"        # nci_fbn (already run)
# tp_variance = the OLD vbp_imagenet_pat criterion (group-L2 both-sides × sqrt(conv-output
# var), no fold, per-layer mean-1). ABLATION vs nci_fbn — same harness, fold OFF.
INCLUDE_TP_VARIANCE = os.environ.get("INCLUDE_TP_VARIANCE", "1") != "0"
# nci_fbn at FULL scope (no interior_only) — also prune the residual stream (conv3/downsample).
INCLUDE_NCI_FBN_FULL = os.environ.get("INCLUDE_NCI_FBN_FULL", "1") != "0"
# nci_fbn + bias compensation (add W[:,c]·μ_c to consumer bias before removal).
INCLUDE_NCI_FBN_BC = os.environ.get("INCLUDE_NCI_FBN_BC", "1") != "0"
# nci_fbn + bias_comp + --no_bn_recalib — the CLEAN bias-comp test: with recalib on, BN
# re-centers and absorbs the correction; off, the compensation actually stands in for it.
INCLUDE_NCI_FBN_BC_NOBN = os.environ.get("INCLUDE_NCI_FBN_BC_NOBN", "1") != "0"
# nci_fbn + bias_comp but NO per-layer norm (--imp_normalizer none) — raw ‖σv‖ keeps the
# cross-layer/kernel-size scale (3x3 vs 1x1) instead of mean-1 per layer.
INCLUDE_NCI_FBN_BC_NONORM = os.environ.get("INCLUDE_NCI_FBN_BC_NONORM", "1") != "0"
# nonrel propagation WITH per-layer norm (--prop_non_relative --imp_normalizer mean). mean-1
# erases the σ_out^p cross-layer transfer → nonrel collapses toward rel (per-layer rank
# identical). Tests whether nonrel's depth-compound matters once the scale is normalized away.
INCLUDE_NONREL_NORM = os.environ.get("INCLUDE_NONREL_NORM", "1") != "0"
# nci_cov = COVARIANCE-aware NCI (--scorer nci_cov). Drop-one output-variance change
# 2Σ_k M_ck Σ_ck − M_cc Σ_cc accounts for the off-diagonal channel covariance the plain NCI
# (independence) drops — boss's check showed 53-89% of Var(Z) lives there. Same recipe as the
# winner nci_fbn (interior, fold off, mean norm) so it's a clean ablation of the cov term.
INCLUDE_NCI_COV = os.environ.get("INCLUDE_NCI_COV", "1") != "0"
# nci_cov at FULL scope (residual stream prunable). nci_cov hooks each consumer's INPUT = the
# post-add tensor, so its measured covariance ALREADY carries σ_c and the cross-branch term →
# this is the EMPIRICAL "cov + σ_c (both)" cell. (Interior nci_cov never crosses a join, so the
# σ_c correction only shows up at full scope.)
INCLUDE_NCI_COV_FULL = os.environ.get("INCLUDE_NCI_COV_FULL", "1") != "0"
# prop + σ_c = propagation criterion with the PDF skip factor (--skip_sigma_c): residual-join
# denominator = MEASURED post-add σ_c instead of the independence sum Σσ_branch^p. The "no cov,
# yes σ_c" cell (σ_c lives in propagation; cov lives in the one-hop nci_cov — different criteria).
INCLUDE_PROP_SIGMAC = os.environ.get("INCLUDE_PROP_SIGMAC", "1") != "0"
# NONREL 2×2 (all --prop_non_relative --imp_normalizer none — keeps the cross-layer scale the
# corrections restore, instead of erasing it with mean-1). Both knobs are the SAME fix — measured
# node variance vs the independence-summed estimate — at two points:
#   cov (--prop_measured_var): per-LAYER denominator colsum Σ(σW)² → measured σ_out² (true Var Z_j)
#   σ_c (--skip_sigma_c):      residual-JOIN denominator Σσ_branch² → measured σ_c² (true Var C)
# 4 cells: base / cov / σ_c / both.
INCLUDE_NONREL_2X2 = os.environ.get("INCLUDE_NONREL_2X2", "1") != "0"

# Option B (reuse pre-regularized RN_bn vnr ckpts) is OFF by default: all the RN_bn sparse
# ckpts were lost/corrupted. Re-enable with INCLUDE_OPTION_B=1 once a valid vnr ckpt exists
# (edit RN_CKPTS to point at it). Until then only Option A (self-contained from 80.86) runs.
INCLUDE_B = os.environ.get("INCLUDE_OPTION_B", "0") != "0"
# C0_magnitude already run + kept → classical baselines OFF by default now. Set
# INCLUDE_CLASSICAL=1 to regenerate them.
INCLUDE_CLASSICAL = os.environ.get("INCLUDE_CLASSICAL", "0") != "0"   # magnitude + bn_scale
RN_CKPTS = {
    "l1e-3": f"{BASE_OUT}/RN_bn_l1e-3/RN_bn_l1e-3_vnr.pth",
    "l3e-3": f"{BASE_OUT}/RN_bn_l3e-3/RN_bn_l3e-3_vnr.pth",
}

# Shared knobs for both routes (reparam mean for propagation σ_out; KD on; bs 128 for DDP×4).
# --global_pruning is ESSENTIAL: the I criterion must rank channels ACROSS layers (otherwise
# per-layer uniform throws away the propagation's cross-layer information — and non-relative
# I, whose only effect is depth-compounding of cross-layer order, becomes meaningless). The
# mac_target search then finds the global ratio whose globally-bottom channels hit the budget.
# Fold native Conv->BN before reparameterize (default ON): bakes the BN scale into M and
# makes the propagation transfer sigma_out POST-BN. Applied to EVERY arm (incl. magnitude)
# so the comparison stays apples-to-apples. Set FOLD_NATIVE_BN=0 to reproduce the old
# (BN-unfolded) scores.
FOLD_BN = os.environ.get("FOLD_NATIVE_BN", "1") != "0"
# Restrict prune scope to conv1/conv2 of each bottleneck (ignore stem conv1, conv3, downsample
# = the residual-stream width). Default ON: matches the proven NCI runs (old vbp_imagenet_pat
# interior_only=True, 32 groups). Full scope (INTERIOR_ONLY=0) also cuts the wide residual dims
# → ~10M params @ 2G MAC, pre-FT acc ~0 (residual surgery).
INTERIOR_ONLY = os.environ.get("INTERIOR_ONLY", "1") != "0"
# Composable shared strings:
#   SHARED_BASE   = no fold, no interior_only (full scope, BN unfolded)
#   SHARED_NOFOLD = SHARED_BASE + interior_only (when INTERIOR_ONLY) — the fbn arm's base
#   SHARED        = SHARED_NOFOLD + fold (when FOLD_BN)              — the default arms' base
SHARED_BASE = (
    f"--model_type cnn --cnn_arch resnet50 --data_path {DATA} --reparam_variant mean "
    f"--scorer propagation --global_pruning --mac_target_g {MAC_TARGET_G} --max_prune_ratio 0.8 "
    f"--calib_batches 50 "
    f"--epochs_ft {FT} --lr_ft 2e-2 --wd 1e-4 --momentum 0.9 --use_kd --kd_alpha 0.5 "
    f"--kd_T 2.0 --train_batch_size 128 --val_resize 232"
)
SHARED_NOFOLD = SHARED_BASE + (" --interior_only" if INTERIOR_ONLY else "")
SHARED = SHARED_NOFOLD + (" --fold_native_bn" if FOLD_BN else "")
# Option B: load vnr, no reg recompute (already 30-ep trained).
B_EXTRA = "--epochs_train 0 --epochs_norm_ft 0"
# Option A (THE thesis test): dense 80.86 → λ-reg norm-ft (push low-contribution ‖ṽ‖→0 so the
# criterion has genuinely-redundant channels to drop) → prune-by-I → FT. mu_ema_momentum>0 keeps
# σ LIVE during the reg phase (mean variant). normnet_main brackets the reg phase with ‖ṽ‖
# snapshots (_contrib_pre_reg / _contrib_post_reg json) — the delta = the reg effect.
A_EXTRA = (f"--epochs_train 0 --epochs_norm_ft {A_REG} --reparam_lambda {A_LAMBDA} "
           f"--lr_norm_ft 0.01 --mu_ema_momentum {A_MU_EMA}")
# Option A0: dense 80.86, NO sparse phase — cold prune at init (already characterized).
A0_EXTRA = "--epochs_train 0 --epochs_norm_ft 0"

# rel = the hero (mass-conserving, span~1e4, true contribution-to-output). nonrel optional.
REL_VARIANTS = [("rel", "")] + ([("nonrel", " --prop_non_relative")] if INCLUDE_NONREL else [])


def _write(tag, ckpt, extra, rflag, shared=SHARED):
    out_dir = os.path.join(BASE_OUT, tag)
    os.makedirs(out_dir, exist_ok=True)
    line = (f"python3 -m torch.distributed.launch --nproc_per_node={NPROC} {NORMNET} "
            f"{shared} {extra}{rflag} --checkpoint {ckpt} "
            f"--save_tag {tag} --save_dir {out_dir}")
    sh_path = os.path.join(out_dir, "run_ddp.sh")
    with open(sh_path, "w") as f:
        f.write(f"#!/bin/bash\nset -e\ncd {REPO}\n{line}\n")
    os.chmod(sh_path, os.stat(sh_path).st_mode | stat.S_IEXEC)
    print(f"wrote {sh_path}")
    return tag


def main():
    made = []
    # OPTION B — propagation (rel + nonrel) on a pre-regularized RN_bn vnr ckpt.
    # OFF unless INCLUDE_OPTION_B=1 (all RN_bn ckpts currently lost).
    if INCLUDE_B:
        for lam, ckpt in RN_CKPTS.items():
            for rname, rflag in REL_VARIANTS:
                made.append(_write(f"B_prop_{rname}_{lam}", ckpt, B_EXTRA, rflag))
    # OPTION A — propagation from the dense 80.86, WITH λ-reg sparse phase (THE thesis test).
    if INCLUDE_A:
        for rname, rflag in REL_VARIANTS:
            made.append(_write(f"A_prop_{rname}", DENSE_8086, A_EXTRA, rflag))
    # OPTION A0 — same, NO sparse phase (cold). Off by default (already characterized).
    if INCLUDE_A0:
        for rname, rflag in REL_VARIANTS:
            made.append(_write(f"A0_prop_{rname}", DENSE_8086, A0_EXTRA, rflag))
    # A0 NCI — cold prune by ‖σv‖=√NCI (--scorer per_layer overrides SHARED's propagation).
    # The bounded cross-layer one-hop baseline against A0_prop_rel/nonrel; same harness.
    # --imp_normalizer mean: per-layer mean-1 BEFORE the global threshold = the proven NCI
    # setting (old vbp_imagenet norm_per_layer=True) AND tp's DepGraph default. WITHOUT it raw
    # ‖σv‖ carries the kernel-size bias (3x3 conv2 ~9x a 1x1 conv1) → conv2 never pruned, conv1
    # gutted. The prop arms keep none (their cross-layer scale IS the criterion).
    if INCLUDE_NCI:
        made.append(_write("A0_nci", DENSE_8086, A0_EXTRA, " --scorer per_layer --imp_normalizer mean"))
    # A0 magnitude — cold prune by L2 weight magnitude (--scorer magnitude). Classical control,
    # mean normalizer = tp DepGraph default (original config the user asked for).
    if INCLUDE_MAGNITUDE:
        made.append(_write("A0_magnitude", DENSE_8086, A0_EXTRA, " --scorer magnitude --imp_normalizer mean"))
    # A0 NCI, BN-fold OFF — same as A0_nci but SHARED_NOFOLD (no --fold_native_bn). Isolates
    # whether folding native BN into the score drives the depth-profile gap vs old tp_variance.
    if INCLUDE_NCI_FBN:
        made.append(_write("A0_nci_fbn", DENSE_8086, A0_EXTRA,
                           " --scorer per_layer --imp_normalizer mean", shared=SHARED_NOFOLD))
    # A0 tp_variance — old vbp_imagenet_pat criterion, fold OFF, mean norm. Ablation vs nci_fbn.
    if INCLUDE_TP_VARIANCE:
        made.append(_write("A0_tp_variance", DENSE_8086, A0_EXTRA,
                           " --scorer tp_variance --imp_normalizer mean", shared=SHARED_NOFOLD))
    # A0 nci_fbn FULL scope — nci_fbn but residual stream prunable (SHARED_BASE = no interior_only).
    if INCLUDE_NCI_FBN_FULL:
        made.append(_write("A0_nci_fbn_full", DENSE_8086, A0_EXTRA,
                           " --scorer per_layer --imp_normalizer mean", shared=SHARED_BASE))
    # A0 nci_fbn + bias compensation — interior_only, fold OFF, + --bias_comp.
    if INCLUDE_NCI_FBN_BC:
        made.append(_write("A0_nci_fbn_bc", DENSE_8086, A0_EXTRA,
                           " --scorer per_layer --imp_normalizer mean --bias_comp", shared=SHARED_NOFOLD))
    # A0 nci_fbn + bias_comp + NO bn recalib — clean bias-comp isolation (recalib would absorb it).
    if INCLUDE_NCI_FBN_BC_NOBN:
        made.append(_write("A0_nci_fbn_bc_nobn", DENSE_8086, A0_EXTRA,
                           " --scorer per_layer --imp_normalizer mean --bias_comp --no_bn_recalib",
                           shared=SHARED_NOFOLD))
    # A0 nci_fbn + bias_comp, NO per-layer norm (--imp_normalizer none = raw ‖σv‖).
    if INCLUDE_NCI_FBN_BC_NONORM:
        made.append(_write("A0_nci_fbn_bc_nonorm", DENSE_8086, A0_EXTRA,
                           " --scorer per_layer --imp_normalizer none --bias_comp",
                           shared=SHARED_NOFOLD))
    # A0 nonrel + per-layer norm — propagation, non-relative, mean-1 (--imp_normalizer mean).
    # Default SHARED (fold ON) like the other prop arms. mean kills the σ_out^p transfer → nonrel≈rel.
    if INCLUDE_NONREL_NORM:
        made.append(_write("A0_nonrel_norm", DENSE_8086, A0_EXTRA,
                           " --prop_non_relative --imp_normalizer mean"))
    # A0 nci_cov — covariance-aware NCI, same recipe as the winning nci_fbn (interior, fold off,
    # mean norm). Ablation: does the off-diagonal covariance term beat plain (independent) NCI?
    if INCLUDE_NCI_COV:
        made.append(_write("A0_nci_cov", DENSE_8086, A0_EXTRA,
                           " --scorer nci_cov --imp_normalizer mean", shared=SHARED_NOFOLD))
    # A0 nci_cov FULL scope — empirical "both" (cov + σ_c): hooking the post-add consumer input
    # measures real Var(C) incl. cross-branch covariance. SHARED_BASE = no interior_only.
    if INCLUDE_NCI_COV_FULL:
        made.append(_write("A0_nci_cov_full", DENSE_8086, A0_EXTRA,
                           " --scorer nci_cov --imp_normalizer mean", shared=SHARED_BASE))
    # A0 prop + σ_c — propagation rel with the PDF measured-σ_c skip factor. "no cov, yes σ_c".
    if INCLUDE_PROP_SIGMAC:
        made.append(_write("A0_prop_sigmac", DENSE_8086, A0_EXTRA,
                           " --skip_sigma_c --imp_normalizer mean", shared=SHARED_NOFOLD))
    # NONREL 2×2 — all --prop_non_relative --imp_normalizer none (cross-layer scale KEPT). cov =
    # --prop_measured_var (layer var), σ_c = --skip_sigma_c (join var). 4 cells.
    if INCLUDE_NONREL_2X2:
        NR = " --prop_non_relative --imp_normalizer none"
        made.append(_write("A0_nonrel_base",   DENSE_8086, A0_EXTRA, NR, shared=SHARED_NOFOLD))
        made.append(_write("A0_nonrel_cov",    DENSE_8086, A0_EXTRA, NR + " --prop_measured_var", shared=SHARED_NOFOLD))
        made.append(_write("A0_nonrel_sigmac", DENSE_8086, A0_EXTRA, NR + " --skip_sigma_c", shared=SHARED_NOFOLD))
        made.append(_write("A0_nonrel_both",   DENSE_8086, A0_EXTRA, NR + " --prop_measured_var --skip_sigma_c", shared=SHARED_NOFOLD))
    # CLASSICAL baselines (same harness: 80.86, mac 2G global, KD, no sparse phase). The
    # --scorer override (last-wins over SHARED's propagation) swaps the criterion. These are
    # the controls that should recover — magnitude/bn_scale lack the propagation gutting.
    if INCLUDE_CLASSICAL:
        for sc in ("magnitude", "bn_scale"):
            made.append(_write(f"C0_{sc}", DENSE_8086, A0_EXTRA, f" --scorer {sc}"))

    print(f"\n{len(made)} experiments (mac_target={MAC_TARGET_G}G, FT={FT} DDP×{NPROC}, "
          f"A_reg={A_REG}ep λ={A_LAMBDA} μ_ema={A_MU_EMA}). Submit each:")
    for tag in made:
        print(f"  python run_ddp.py --out_dir_name {tag}")


if __name__ == "__main__":
    main()
