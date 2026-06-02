"""
Generate the propagation-criterion (I) pruning experiments — ALL on normnet_main (DDP),
ALL from the v2 (80.86) lineage, targeting a fixed MAC budget. run_ddp.py fashion.

Hero criterion = the PDF's PROPAGATION importance I^l = W̄^l·I^{l+1} (--scorer propagation),
NOT NCI. Both PDF derivations are swept:
  relative     W̄ = M^p·D   (column-normalized, mass-preserving)        [default]
  non-relative W̄ = M^p     (--prop_non_relative; compounds through depth)

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

# Completed RN_bn sparse vnr ckpts to prune (Option B). l1e-4 ≈ 20% channels <0.1, l3e-4 ≈ 65%.
RN_CKPTS = {
    "l1e-4": f"{BASE_OUT}/RN_bn_l1e-4/RN_bn_l1e-4_vnr.pth",
    "l3e-4": f"{BASE_OUT}/RN_bn_l3e-4/RN_bn_l3e-4_vnr.pth",
}

# Shared knobs for both routes (reparam mean for propagation σ_out; KD on; bs 128 for DDP×4).
SHARED = (
    f"--model_type cnn --cnn_arch resnet50 --data_path {DATA} --reparam_variant mean "
    f"--scorer propagation --mac_target_g {MAC_TARGET_G} --calib_batches 50 "
    f"--epochs_ft {FT} --lr_ft 2e-2 --wd 1e-4 --momentum 0.9 --use_kd --kd_alpha 0.5 "
    f"--kd_T 2.0 --train_batch_size 128 --val_resize 232"
)
# Option B: load vnr, no reg recompute (already 30-ep trained).
B_EXTRA = "--epochs_train 0 --epochs_norm_ft 0"
# Option A: dense 80.86, short λ-reg norm-ft before pruning.
A_EXTRA = f"--epochs_train 0 --epochs_norm_ft {A_REG} --reparam_lambda 1e-4 --lr_norm_ft 0.01"

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
    # OPTION B — propagation (rel + nonrel) on each completed RN_bn vnr ckpt.
    for lam, ckpt in RN_CKPTS.items():
        for rname, rflag in REL_VARIANTS:
            made.append(_write(f"B_prop_{rname}_{lam}", ckpt, B_EXTRA, rflag))
    # OPTION A — propagation (rel + nonrel) from the dense 80.86.
    for rname, rflag in REL_VARIANTS:
        made.append(_write(f"A_prop_{rname}", DENSE_8086, A_EXTRA, rflag))

    print(f"\n{len(made)} experiments (mac_target={MAC_TARGET_G}G, FT={FT} DDP×{NPROC}, "
          f"A_reg={A_REG}). Submit each:")
    for tag in made:
        print(f"  python run_ddp.py --out_dir_name {tag}")


if __name__ == "__main__":
    main()
