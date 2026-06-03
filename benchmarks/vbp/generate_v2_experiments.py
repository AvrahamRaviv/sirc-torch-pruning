"""
Generate the v2-recipe from-scratch experiments (100 epochs each), run_ddp.py fashion.
v2_baseline is KEPT (already run); regenerate only the switch arm:

  v2_switch_e{X}   official recipe, switch to normalized coords at epoch X, WITH the S1
                   σ² grad-preconditioner (--switch_precond) that cancels the 1/σ² post-
                   switch LR inflation. Without it the net collapsed the epoch after the
                   switch (grad/param ×~50). The switch also dumps a σ sidecar
                   (<tag>_preswitch_e{X}_meta.pt) + pre-switch weights so the fix can be
                   iterated from the checkpoint without retraining to epoch X.

Compare per-epoch val_acc (raw + EMA) curves in <tag>_metrics.jsonl vs v2_baseline → §7:
does the mid-training switch converge faster / higher than the plain protocol?

lr is scaled for the effective batch: official used 0.5 @ batch 1024 (8×128); here 4 GPUs
× 128 = 512 → lr 0.25 (linear scaling). Adjust if you change GPUs/batch.

Run:
    python benchmarks/vbp/generate_v2_experiments.py
    python run_ddp.py --out_dir_name v2_switch_e30
"""
import os
import stat

BASE_OUT = os.environ.get("V2_BASE_OUT", "/algo/NetOptimization/outputs/NORMNET/ResNet50")
REPO = os.environ.get("EXP_REPO", "/home/avrahamra/PycharmProjects/sirc-torch-pruning")
SCRIPT = f"{REPO}/benchmarks/vbp/train_v2.py"
DATA = "/algo/NetOptimization/outputs/VBP/"
NPROC = int(os.environ.get("V2_NPROC", "4"))
EPOCHS = int(os.environ.get("V2_EPOCHS", "100"))
SWITCH_EPOCH = int(os.environ.get("V2_SWITCH_EPOCH", "30"))  # when BN/weights have settled
LR = os.environ.get("V2_LR", "0.25")                         # 0.5 @ batch1024 → 0.25 @ 4×128

# official v2 recipe held identical across both arms (from-scratch: no --checkpoint).
COMMON = (
    f"--model_type cnn --cnn_arch resnet50 --data_path {DATA} --epochs {EPOCHS} "
    f"--lr {LR} --wd 2e-5 --momentum 0.9 --warmup_epochs 5 --label_smoothing 0.1 "
    f"--mixup_alpha 0.2 --cutmix_alpha 1.0 --random_erase 0.1 --ema_decay 0.9998 "
    f"--ra_reps 4 --train_crop 176 --val_resize 232 --train_batch_size 128 "
    f"--reparam_variant bn --norm_bn_momentum 0.01 --calib_batches 50"
)

# v2_baseline kept from the prior run — not regenerated. Switch arm carries the S1 fix.
ARMS = [
    (f"v2_switch_e{SWITCH_EPOCH}", f"--reparam_at_epoch {SWITCH_EPOCH} --switch_precond"),
    # Fast smoke (switch at e1, 3 epochs × 3 batches) — verifies the switch + precond wiring
    # and that val_acc does NOT collapse post-switch. ~1-2 min. argparse last-wins overrides.
    ("v2_smoke_switch_e1",
     "--epochs 3 --warmup_epochs 1 --limit_batches 3 --calib_batches 3 "
     "--reparam_at_epoch 1 --switch_precond"),
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
    print(f"\n{len(made)} experiments (epochs={EPOCHS}, switch@{SWITCH_EPOCH}, lr={LR}). "
          f"Submit each ({NPROC} GPUs):")
    for tag in made:
        print(f"  python run_ddp.py --out_dir_name {tag}")


if __name__ == "__main__":
    main()
