"""
Generate the 6 overnight RN50 ImageNet-1k prune+FT experiments on the PROVEN
vbp_imagenet_pat.py harness (DDP, 4 GPUs) — the recipe that reaches ~0.76 Best.

Tests paper §2 (σ-weighted importance > plain magnitude) and §4 (gap grows with
sparsity): 3 criteria × 2 MAC keep-ratios. Everything else is held IDENTICAL to the
known-good command (sparse_mode vnr + reparam_lambda 1e-3 + bn_recalibration + lr 0.08
+ 10ep warmup + 90ep FT + KD α0.5 T2.0 + global mac_target), so the only moving part is
the pruning criterion → the comparison is clean and anchored to the 0.76 baseline.

Criteria:
  mag   : --criterion magnitude                                  → ‖v‖ baseline (§2 null)
  tpvar : --criterion variance --importance_mode tp_variance     → group-mag × σ (NCI, the
          --norm_per_layer                                          proven arm, §2 hero)
  var   : --criterion variance --importance_mode variance        → activation σ² (VBP)
          --norm_per_layer

Run:
    python benchmarks/vbp/generate_experiments.py     # writes 6 folders + run.sh
    # then launch each printed bash (each grabs 4 GPUs via torch.distributed.launch)
"""
import os
import stat

# --- cluster paths (override via env for dry-runs) ---
BASE_OUT = os.environ.get("EXP_BASE_OUT", "/algo/NetOptimization/outputs/VBP/ResNet50_TP")
REPO = os.environ.get("EXP_REPO", "/home/avrahamra/PycharmProjects/sirc-torch-pruning")
SCRIPT = f"{REPO}/benchmarks/vbp/vbp_imagenet_pat.py"
CKPT = f"{BASE_OUT}/resnet50_imagenet1k_v1.pth"
TEACHER = f"{BASE_OUT}/resnet50_imagenet1k_mag_sparse.pth"
DATA = "/algo/NetOptimization/outputs/VBP/"
RUN_ROOT = os.path.join(BASE_OUT, "repro_norm_overnight")
NPROC = int(os.environ.get("EXP_NPROC", "4"))

# --- experiment matrix ---
# (tag, criterion-flags) — held identical otherwise.
ARMS = [
    ("mag",   "--criterion magnitude"),
    ("tpvar", "--criterion variance --importance_mode tp_variance --norm_per_layer"),
    ("var",   "--criterion variance --importance_mode variance --norm_per_layer"),
]
KEEP_RATIOS = [0.5, 0.3]   # MAC keep target; 0.5 = proven working point, 0.3 = harder

# --- shared recipe (EXACT proven command, minus criterion / keep_ratio / save_dir) ---
COMMON = (
    f"--model_type cnn --cnn_arch resnet50 "
    f"--checkpoint {CKPT} --teacher_checkpoint {TEACHER} --data_path {DATA} "
    f"--global_pruning --mac_target --bn_recalibration "
    f"--sparse_mode vnr --reparam_lambda 1e-3 --epochs_sparse 0 "
    f"--epochs_ft 90 --opt sgd --lr 0.08 --ft_lr 0.08 --ft_eta_min 1e-6 "
    f"--ft_warmup_epochs 10 --wd 1e-4 --train_batch_size 256 "
    f"--pat_steps 1 --pat_epochs_per_step 0 --use_kd --kd_alpha 0.5 --kd_T 2.0"
)

SH_TEMPLATE = (
    "#!/bin/bash\n"
    "set -e\n"
    f"cd {REPO}\n"
    f"python3 -m torch.distributed.launch --nproc_per_node={NPROC} {{script}} "
    "{common} {arm} --keep_ratio {kr} --save_dir {out_dir}\n"
)


def main():
    made = []
    for arm_tag, arm_flags in ARMS:
        for kr in KEEP_RATIOS:
            tag = f"E_{arm_tag}_kr{int(round(kr * 100))}"
            out_dir = os.path.join(RUN_ROOT, tag)
            os.makedirs(out_dir, exist_ok=True)
            sh_path = os.path.join(out_dir, "run.sh")
            text = SH_TEMPLATE.format(
                script=SCRIPT, common=COMMON, arm=arm_flags, kr=kr, out_dir=out_dir)
            with open(sh_path, "w") as f:
                f.write(text)
            os.chmod(sh_path, os.stat(sh_path).st_mode | stat.S_IEXEC)
            made.append(sh_path)
            print(f"wrote {sh_path}")
    print(f"\n{len(made)} experiments. Launch each (each grabs {NPROC} GPUs):")
    for sh in made:
        print(f"  bash {sh}")


if __name__ == "__main__":
    main()
