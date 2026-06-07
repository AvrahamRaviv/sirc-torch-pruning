"""
Generate the 6 overnight RN50 ImageNet-1k experiments on the NEW normalize_net.py
script (the canonical BN-trick training), output under NORMNET — same run_ddp.py
fashion as generate_e2_experiments.py.

normalize_net.py TRAINS only (reparameterize → train under the BN trick → merge_back →
save vnr + merged ckpts). No pruning here; prune the resulting vnr ckpt later via
prune_e2 / generate_e2_experiments. So this set tests the TRAINING-side theory:

  §7 / E1 (BN trick trains clean, no cliff):  RN_bn_l0  vs  RN_baseline (--no_reparam).
  §6 / E4 (λ‖v_tilde‖ induces channel sparsity at ~no acc cost): λ sweep. Watch
        vnorm_summary.frac_below_0.1 grow in <tag>_metrics.jsonl as λ rises.

KD teacher = the frozen pretrained net (built inside normalize_net, no --teacher flag).
σ EMA kept slow via --norm_bn_momentum.

>>> ASSUMPTIONS to eyeball before submit (top constants): EPOCHS, LR, WARMUP. <<<
This is a FT of a converged net (not prune-recovery), so moderate lr + short cosine.

Run (run_ddp.py fashion):
    python benchmarks/vbp/generate_experiments.py        # writes 6 run_ddp.sh
    python run_ddp.py --out_dir_name RN_bn_l1e-3          # submit one (repeat per folder)

Each run_ddp.sh's last token is `--save_dir <out_dir>` so run_ddp.py patches it at
submit time. The .sh keeps the torch.distributed.launch line (DDP); run_ddp.py just
schedules the bash onto a node.
"""
import os
import stat

# --- paths (override via env for dry-runs) ---
BASE_OUT = os.environ.get("EXP_BASE_OUT", "/algo/NetOptimization/outputs/NORMNET/ResNet50")
REPO = os.environ.get("EXP_REPO", "/home/avrahamra/PycharmProjects/sirc-torch-pruning")
SCRIPT = f"{REPO}/benchmarks/vbp/normalize_net.py"
# Pretrained dense RN50 to fine-tune under the BN trick. v2 (80.86) over v1 (76.13):
# v2's NATIVE eval is resize 232 (matches --val_resize 232 below, no crop mismatch), and
# a stronger teacher = better KD. v1 at resize 232 reads ~75.89 (its native eval is 256).
# Override via env CKPT=... if the filename differs.
CKPT = os.environ.get(
    "CKPT", "/algo/NetOptimization/outputs/VBP/ResNet50_TP/resnet50_imagenet1k.pth")
DATA = "/algo/NetOptimization/outputs/VBP/"
NPROC = int(os.environ.get("EXP_NPROC", "4"))

# --- assumed FT knobs (tweak before submit) ---
EPOCHS = int(os.environ.get("EXP_EPOCHS", "30"))
LR = os.environ.get("EXP_LR", "0.01")
WARMUP = int(os.environ.get("EXP_WARMUP", "3"))

# --- experiment matrix (6) ---
# baseline (plain) + BN-trick λ sweep. tag → arm-specific flags.
ARMS = [
    ("RN_baseline", "--no_reparam"),
    ("RN_bn_l0",    "--reparam_variant bn --reparam_lambda 0"),
    ("RN_bn_l1e-4", "--reparam_variant bn --reparam_lambda 1e-4"),
    ("RN_bn_l3e-4", "--reparam_variant bn --reparam_lambda 3e-4"),
    ("RN_bn_l1e-3", "--reparam_variant bn --reparam_lambda 1e-3"),
    ("RN_bn_l3e-3", "--reparam_variant bn --reparam_lambda 3e-3"),
]
# Dropped λ=1e-2 (dead: <0.1=99.6%, acc gutted to 0.64). Knee is below 1e-3; this ½-decade
# ladder {1e-4,3e-4,1e-3,3e-3} gives finer resolution near it.

# --- shared recipe (proven hyperparams; KD α0.5 T2.0, slow σ-EMA) ---
COMMON = (
    f"--model_type cnn --cnn_arch resnet50 --checkpoint {CKPT} --data_path {DATA} "
    f"--epochs {EPOCHS} --opt sgd --lr {LR} --wd 1e-4 --ft_eta_min 1e-6 "
    f"--ft_warmup_epochs {WARMUP} --train_batch_size 128 --val_resize 232 "
    f"--use_kd --kd_alpha 0.5 --kd_T 2.0 --norm_bn_momentum 0.01 --ckpt_interval 5"
)

SH_TEMPLATE = (
    "#!/bin/bash\n"
    "set -e\n"
    f"cd {REPO}\n"
    f"python3 -m torch.distributed.launch --nproc_per_node={NPROC} {{script}} "
    "{common} {arm} --save_tag {tag} --save_dir {out_dir}\n"
)


def main():
    made = []
    for tag, arm_flags in ARMS:
        out_dir = os.path.join(BASE_OUT, tag)
        os.makedirs(out_dir, exist_ok=True)
        sh_path = os.path.join(out_dir, "run_ddp.sh")
        # out_dir is a placeholder; run_ddp.py re-patches --save_dir to the resolved
        # path at submit time (kept last for the regex).
        text = SH_TEMPLATE.format(
            script=SCRIPT, common=COMMON, arm=arm_flags, tag=tag, out_dir=out_dir)
        with open(sh_path, "w") as f:
            f.write(text)
        os.chmod(sh_path, os.stat(sh_path).st_mode | stat.S_IEXEC)
        made.append(tag)
        print(f"wrote {sh_path}")
    print(f"\n{len(made)} experiments (epochs={EPOCHS} lr={LR} warmup={WARMUP}). "
          f"Submit each ({NPROC} GPUs):")
    for tag in made:
        print(f"  python run_ddp.py --out_dir_name {tag}")


if __name__ == "__main__":
    main()
