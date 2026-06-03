"""
Generate the propagation-criterion (I) pruning experiments — ALL on normnet_main (DDP),
ALL from the v2 (80.86) lineage, targeting a fixed MAC budget. run_ddp.py fashion.

Hero criterion = the PDF's PROPAGATION importance I^l = W̄^l·I^{l+1} (--scorer propagation),
NOT NCI. Both PDF derivations are swept:
  relative     within-layer redistribution (drops the inter-layer transfer)   [default]
  non-relative keeps σ_out^p inter-layer transfer → GLOBAL by design  (--prop_non_relative)

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
A_REG = int(os.environ.get("A_REG_EPOCHS", "5"))              # Option A short λ-reg norm-ft

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
SHARED = (
    f"--model_type cnn --cnn_arch resnet50 --data_path {DATA} --reparam_variant mean "
    f"--scorer propagation --global_pruning --mac_target_g {MAC_TARGET_G} --max_prune_ratio 0.8 "
    f"--calib_batches 50 "
    f"--epochs_ft {FT} --lr_ft 2e-2 --wd 1e-4 --momentum 0.9 --use_kd --kd_alpha 0.5 "
    f"--kd_T 2.0 --train_batch_size 128 --val_resize 232"
)
# Option B: load vnr, no reg recompute (already 30-ep trained).
B_EXTRA = "--epochs_train 0 --epochs_norm_ft 0"
# Option A: dense 80.86, short λ-reg norm-ft (sparse phase) before pruning.
A_EXTRA = f"--epochs_train 0 --epochs_norm_ft {A_REG} --reparam_lambda 1e-4 --lr_norm_ft 0.01"
# Option A0: dense 80.86, NO sparse phase — prune the net at init (tests whether the reg
# phase actually helps the I-prune, or the criterion alone suffices).
A0_EXTRA = "--epochs_train 0 --epochs_norm_ft 0"

REL_VARIANTS = [("rel", ""), ("nonrel", " --prop_non_relative")]


def _write(tag, ckpt, extra, rflag):
    out_dir = os.path.join(BASE_OUT, tag)
    os.makedirs(out_dir, exist_ok=True)
    line = (f"python3 -m torch.distributed.launch --nproc_per_node={NPROC} {NORMNET} "
            f"{SHARED} {extra}{rflag} --checkpoint {ckpt} "
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
    # OPTION A — propagation (rel + nonrel) from the dense 80.86, WITH 5ep λ-reg sparse phase.
    for rname, rflag in REL_VARIANTS:
        made.append(_write(f"A_prop_{rname}", DENSE_8086, A_EXTRA, rflag))
    # OPTION A0 — same, NO sparse phase (prune at init) → does the reg phase help?
    for rname, rflag in REL_VARIANTS:
        made.append(_write(f"A0_prop_{rname}", DENSE_8086, A0_EXTRA, rflag))
    # CLASSICAL baselines (same harness: 80.86, mac 2G global, KD, no sparse phase). The
    # --scorer override (last-wins over SHARED's propagation) swaps the criterion. These are
    # the controls that should recover — magnitude/bn_scale lack the propagation gutting.
    if INCLUDE_CLASSICAL:
        for sc in ("magnitude", "bn_scale"):
            made.append(_write(f"C0_{sc}", DENSE_8086, A0_EXTRA, f" --scorer {sc}"))

    print(f"\n{len(made)} experiments (mac_target={MAC_TARGET_G}G, FT={FT} DDP×{NPROC}, "
          f"A_reg={A_REG}). Submit each:")
    for tag in made:
        print(f"  python run_ddp.py --out_dir_name {tag}")


if __name__ == "__main__":
    main()
