"""
Generate the propagation-criterion (I) pruning experiments, run_ddp.py fashion.

The hero criterion is the PDF's PROPAGATION importance I^l = W̄^l·I^{l+1} (--scorer
propagation), NOT NCI (per_layer ‖σ·v‖). Both PDF derivations are swept:
  relative     W̄ = M^p·D   (column-normalized, mass-preserving)        [default]
  non-relative W̄ = M^p     (--prop_non_relative; compounds through depth)

Two routes to the same place (run BOTH; they answer different questions):

  OPTION B — reg-then-prune on the COMPLETED RN_bn λ-sweep ckpts (the 30-epoch
    bn-trick sparse runs). prune_e2 loads the *_vnr.pth (the merged_biased save is
    broken on the cluster → reloads to ~0.001; vnr reloads to 0.808). Single-process
    (pruning is a structural edit) → each arm takes 1 GPU; pack several per node.

  OPTION A — self-contained from the dense 80.86 ckpt via normnet_main (DDP):
    load → short λ-reg norm-ft → prune-by-I → FT, one job, NPROC GPUs.

Propagation needs σ_out branch weighting. We reparam the MEAN variant at scoring time
(the reference impl); reparam at prune is a cold calibration on the loaded weights, so
it is independent of how the ckpt was trained (RN_bn = bn-trained, scored via mean is
fine — the induced sparsity lives in the weights).

A single MAGNITUDE arm per route is kept as the baseline-for-comparison (NOT NCI — plain
‖v‖). Drop it with INCLUDE_MAG=0 if you only want I.

Run:
    python benchmarks/vbp/generate_prune_I_experiments.py
    python run_ddp.py --out_dir_name B_prop_rel_l1e-4_r50     # submit one (repeat per folder)
"""
import os
import stat

BASE_OUT = os.environ.get("PRUNE_BASE_OUT", "/algo/NetOptimization/outputs/NORMNET/ResNet50")
REPO = os.environ.get("EXP_REPO", "/home/avrahamra/PycharmProjects/sirc-torch-pruning")
PRUNE_E2 = f"{REPO}/benchmarks/vbp/prune_e2.py"
NORMNET = f"{REPO}/benchmarks/vbp/normnet_main.py"
DATA = "/algo/NetOptimization/outputs/VBP/"
NPROC = int(os.environ.get("EXP_NPROC", "4"))

DENSE_8086 = os.environ.get(
    "CKPT", "/algo/NetOptimization/outputs/VBP/ResNet50_TP/resnet50_imagenet1k.pth")
RATIO = float(os.environ.get("PRUNE_RATIO", "0.5"))
B_FT = int(os.environ.get("B_FT_EPOCHS", "10"))      # single-GPU FT (Option B)
A_FT = int(os.environ.get("A_FT_EPOCHS", "30"))      # DDP FT (Option A)
INCLUDE_MAG = os.environ.get("INCLUDE_MAG", "1") != "0"

R = int(round(RATIO * 100))

# Completed RN_bn sparse ckpts to prune (Option B). l1e-4 ≈ 20% channels <0.1 (cheap
# knee), l3e-4 ≈ 65% (heavy). Prune both at the SAME ratio for a clean criterion compare.
RN_CKPTS = {
    "l1e-4": f"{BASE_OUT}/RN_bn_l1e-4/RN_bn_l1e-4_vnr.pth",
    "l3e-4": f"{BASE_OUT}/RN_bn_l3e-4/RN_bn_l3e-4_vnr.pth",
}

# Option B recipe: prune_e2, single-process, loads a vnr ckpt, no sparse phase (already
# 30-ep reg-trained). reparam mean for the propagation σ_out weighting.
B_COMMON = (
    f"--model_type cnn --cnn_arch resnet50 --data_path {DATA} --epochs_ft {B_FT} "
    f"--opt sgd --lr 2e-2 --wd 1e-4 --momentum 0.9 --ft_eta_min 1e-6 --ft_warmup_epochs 1 "
    f"--use_kd --kd_alpha 0.5 --kd_T 2.0 --train_batch_size 256 --val_resize 232 "
    f"--calib_batches 50 --epochs_sparse 0 --reparam_variant mean"
)

# Option A recipe: normnet_main, DDP, dense 80.86, short reg then prune-by-I.
A_COMMON = (
    f"--model_type cnn --cnn_arch resnet50 --checkpoint {DENSE_8086} --data_path {DATA} "
    f"--epochs_train 0 --reparam_variant mean --epochs_norm_ft 5 --reparam_lambda 1e-4 "
    f"--lr_norm_ft 0.01 --calib_batches 50 --epochs_ft {A_FT} --lr_ft 2e-2 --wd 1e-4 "
    f"--momentum 0.9 --use_kd --kd_alpha 0.5 --kd_T 2.0 --train_batch_size 128 --val_resize 232"
)

REL_VARIANTS = [("rel", ""), ("nonrel", " --prop_non_relative")]


def _write(tag, line):
    out_dir = os.path.join(BASE_OUT, tag)
    os.makedirs(out_dir, exist_ok=True)
    sh_path = os.path.join(out_dir, "run_ddp.sh")
    with open(sh_path, "w") as f:
        f.write(f"#!/bin/bash\nset -e\ncd {REPO}\n{line} --save_tag {tag} --save_dir {out_dir}\n")
    os.chmod(sh_path, os.stat(sh_path).st_mode | stat.S_IEXEC)
    print(f"wrote {sh_path}")
    return tag


def _b_line(ckpt, scorer, rflag):
    return (f"python {PRUNE_E2} {B_COMMON} --checkpoint {ckpt} "
            f"--scorer {scorer}{rflag} --pruning_ratio {RATIO}")


def _a_line(scorer, rflag):
    return (f"python3 -m torch.distributed.launch --nproc_per_node={NPROC} {NORMNET} "
            f"{A_COMMON} --scorer {scorer}{rflag} --pruning_ratio {RATIO}")


def main():
    made = []
    # OPTION B — propagation (rel + nonrel) on each completed RN_bn ckpt.
    for lam, ckpt in RN_CKPTS.items():
        for rname, rflag in REL_VARIANTS:
            made.append(_write(f"B_prop_{rname}_{lam}_r{R}", _b_line(ckpt, "propagation", rflag)))
    if INCLUDE_MAG:
        made.append(_write(f"B_mag_l1e-4_r{R}", _b_line(RN_CKPTS["l1e-4"], "magnitude", "")))

    # OPTION A — propagation (rel + nonrel) from the dense 80.86.
    for rname, rflag in REL_VARIANTS:
        made.append(_write(f"A_prop_{rname}_r{R}", _a_line("propagation", rflag)))
    if INCLUDE_MAG:
        made.append(_write(f"A_mag_r{R}", _a_line("magnitude", "")))

    print(f"\n{len(made)} experiments (ratio={RATIO}, B_ft={B_FT} single-GPU, "
          f"A_ft={A_FT} DDP×{NPROC}). Submit each:")
    for tag in made:
        print(f"  python run_ddp.py --out_dir_name {tag}")


if __name__ == "__main__":
    main()
