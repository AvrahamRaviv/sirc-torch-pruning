"""
Generate the v2-recipe normalize-switch experiments (100 epochs each), run_ddp.py fashion.

ROUND-1 (already run / running, dirs exist): v2_baseline (EMA 78.1), v2_switch_e30_precond,
v2_switch_e30_lr025, v2_switch_e30_lr050, v2_switch_e15_lr025. Readout @e35: live-σ precond
ties baseline (frozen-σ throttle fixed); the lr_scale switch arms lead early but run at 4×/2×
LOWER effective LR (0.25×scale) than baseline → the lead is confounded with low-LR noise drop,
NOT yet a proven normalization gain.

ROUND-2 (this file, 3 weekend arms) isolates §7 from the LR confound:

  v2_ctrl_e30_lr025_plain  PLAIN training (no normalize), resume e30, --lr 0.0625. lr_at is
                           linear in --lr, so this reproduces the EXACT e30→100 LR of the
                           lr025 switch arm. §7 effect at this LR = lr025_switch − this ctrl.
  v2_ctrl_e30_lr050_plain  PLAIN, --lr 0.125 → matched control for the lr050 switch arm.
  v2_switch_e30_lr015      normalize switch, --switch_lr_scale 0.15. Lower §7 point; with
                           lr025/lr050 makes a 3-point sweep + mirage check.

§7 is real iff a switch arm beats its matched plain control at the SAME LR. Otherwise the
early lead was just the lower LR. All arms resume from the round-1 pre-switch checkpoint
(RESUME30) → 70 epochs each. Recipe held identical: --calib_transform train, bn variant,
norm_bn_momentum 0.01.

Compare per-epoch val_acc (raw + EMA) in <tag>_metrics.jsonl. lr is scaled for the effective
batch: official 0.5 @ batch 1024 (8×128); here 4 GPUs × 128 = 512 → base lr 0.25.

Run (one at a time, 4 GPUs each):
    python benchmarks/vbp/generate_v2_experiments.py
    python run_ddp.py --out_dir_name v2_ctrl_e30_lr025_plain
"""
import os
import stat

BASE_OUT = os.environ.get("V2_BASE_OUT", "/algo/NetOptimization/outputs/NORMNET/ResNet50")
REPO = os.environ.get("EXP_REPO", "/home/avrahamra/PycharmProjects/sirc-torch-pruning")
SCRIPT = f"{REPO}/benchmarks/vbp/train_v2.py"
DATA = "/algo/NetOptimization/outputs/VBP/"
NPROC = int(os.environ.get("V2_NPROC", "4"))
EPOCHS = int(os.environ.get("V2_EPOCHS", "100"))
LR = os.environ.get("V2_LR", "0.25")                         # 0.5 @ batch1024 → 0.25 @ 4×128
# Pre-switch plain checkpoint from the round-1 run (saved at the e30 switch). The 3 e30 arms
# load it and resume the 100-epoch cosine at e30 (--start_epoch 30), skipping the identical
# plain 0–30 phase → saves 90 GPU-epochs. The e15 arm has no e15 ckpt, runs full.
PRESWITCH = os.environ.get(
    "V2_PRESWITCH", f"{BASE_OUT}/v2_switch_e30/v2_switch_e30_preswitch_e30.pth")
RESUME30 = f"--checkpoint {PRESWITCH} --start_epoch 30"

# official v2 recipe held identical across all arms (from-scratch: no --checkpoint).
COMMON = (
    f"--model_type cnn --cnn_arch resnet50 --data_path {DATA} --epochs {EPOCHS} "
    f"--lr {LR} --wd 2e-5 --momentum 0.9 --warmup_epochs 5 --label_smoothing 0.1 "
    f"--mixup_alpha 0.2 --cutmix_alpha 1.0 --random_erase 0.1 --ema_decay 0.9998 "
    f"--ra_reps 4 --train_crop 176 --val_resize 232 --train_batch_size 128 "
    f"--reparam_variant bn --norm_bn_momentum 0.01 --calib_batches 50 --calib_transform train"
)

# ROUND-1 arms (already run / running — dirs exist, not regenerated here):
#   v2_baseline, v2_switch_e30_precond, v2_switch_e30_lr025, v2_switch_e30_lr050,
#   v2_switch_e15_lr025.
# Round-1 readout: live-σ precond ties baseline (drift artifact fixed). The lr_scale switch
# arms lead early BUT run at 4×/2× lower effective LR than baseline (0.25×scale) → the early
# val boost is confounded with low-LR noise reduction, not proven to be a normalization gain.
#
# ROUND-2 (this file, 3 weekend arms) isolates §7 from the LR confound:
#   - 2 PLAIN controls at the SAME tail-LR trajectory as the lr025/lr050 switch arms (no
#     normalize). lr_at scales linearly with --lr, so --lr 0.0625 / 0.125 reproduce exactly
#     the e30→100 LR of switch_lr_scale 0.25 / 0.50. §7 effect = switch − matched plain ctrl.
#   - 1 lower §7 point (scale 0.15) → lr015/025/050 is a 3-point sweep + a mirage check
#     (if even-lower LR only delays the fade, the lead is LR not coords).
ARMS = [
    ("v2_ctrl_e30_lr025_plain", f"--reparam_at_epoch -1 --lr 0.0625 {RESUME30}"),
    ("v2_ctrl_e30_lr050_plain", f"--reparam_at_epoch -1 --lr 0.125 {RESUME30}"),
    ("v2_switch_e30_lr015",     f"--reparam_at_epoch 30 --switch_lr_scale 0.15 {RESUME30}"),
]

SH = (
    "#!/bin/bash\nset -e\n"
    f"cd {REPO}\n"
    f"python3 -m torch.distributed.launch --nproc_per_node={NPROC} {{script}} "
    "{common} {arm} --save_tag {tag} --save_dir {out_dir}\n"
)


def main():
    made = []
    for tag, arm in ARMS:
        out_dir = os.path.join(BASE_OUT, tag)
        os.makedirs(out_dir, exist_ok=True)
        sh_path = os.path.join(out_dir, "run_ddp.sh")
        # --save_dir kept last so run_ddp.py re-patches it at submit time.
        with open(sh_path, "w") as f:
            f.write(SH.format(script=SCRIPT, common=COMMON, arm=arm, tag=tag, out_dir=out_dir))
        os.chmod(sh_path, os.stat(sh_path).st_mode | stat.S_IEXEC)
        made.append(tag)
        print(f"wrote {sh_path}")
    print(f"\n{len(made)} experiments (epochs={EPOCHS}, lr={LR}). "
          f"Submit each ({NPROC} GPUs):")
    for tag in made:
        print(f"  python run_ddp.py --out_dir_name {tag}")


if __name__ == "__main__":
    main()
