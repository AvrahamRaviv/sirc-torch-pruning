"""
Generate the v2-recipe from-scratch experiments (100 epochs each), run_ddp.py fashion.
v2_baseline is KEPT (already run, EMA 78.1) — not regenerated. Four switch arms test §7:
does moving to normalized coords mid-training beat the plain protocol?

Round-1 (switch@30 + frozen-σ precond) underperformed baseline by ~2.6% — a frozen-σ
LR throttle, NOT a real §7 verdict. Now fixed (live-σ precond + train-calib). The four
arms separate the precond control from the actual §7 test:

  v2_switch_e30_precond  live-σ precond ON. Reverts v_tilde-SGD to plain-W dynamics, so
                         this is the CONTROL: should now TIE baseline (~78). If it still
                         lags, the deficit is EMA/momentum reset, not normalized coords.
  v2_switch_e30_lr025    precond OFF, --switch_lr_scale 0.25. The §7 test: true
                         normalized-coords dynamics, 1/σ² overshoot tamed by one factor.
  v2_switch_e30_lr050    same, scale 0.50 — brackets the normalized-LR sweet spot.
  v2_switch_e15_lr025    earlier switch → more epochs in normalized coords. If §7 helps,
                         an earlier switch should amplify it.

All four: --calib_transform train (σ_cal matches σ_run), bn variant, norm_bn_momentum 0.01.
Each switch dumps a σ sidecar (<tag>_preswitch_e{X}_meta.pt) + pre-switch weights.

Compare per-epoch val_acc (raw + EMA) in <tag>_metrics.jsonl vs v2_baseline. Read raw-vs-raw
to drop the EMA-reset confound.

lr is scaled for the effective batch: official used 0.5 @ batch 1024 (8×128); here 4 GPUs
× 128 = 512 → lr 0.25 (linear scaling). Adjust if you change GPUs/batch.

Run (one at a time, 4 GPUs each):
    python benchmarks/vbp/generate_v2_experiments.py
    python run_ddp.py --out_dir_name v2_switch_e30_lr025
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

# official v2 recipe held identical across all arms (from-scratch: no --checkpoint).
COMMON = (
    f"--model_type cnn --cnn_arch resnet50 --data_path {DATA} --epochs {EPOCHS} "
    f"--lr {LR} --wd 2e-5 --momentum 0.9 --warmup_epochs 5 --label_smoothing 0.1 "
    f"--mixup_alpha 0.2 --cutmix_alpha 1.0 --random_erase 0.1 --ema_decay 0.9998 "
    f"--ra_reps 4 --train_crop 176 --val_resize 232 --train_batch_size 128 "
    f"--reparam_variant bn --norm_bn_momentum 0.01 --calib_batches 50 --calib_transform train"
)

# 4 GPU-budget arms. v2_baseline kept from the prior run.
ARMS = [
    ("v2_switch_e30_precond", "--reparam_at_epoch 30 --switch_precond"),
    ("v2_switch_e30_lr025",   "--reparam_at_epoch 30 --switch_lr_scale 0.25"),
    ("v2_switch_e30_lr050",   "--reparam_at_epoch 30 --switch_lr_scale 0.50"),
    ("v2_switch_e15_lr025",   "--reparam_at_epoch 15 --switch_lr_scale 0.25"),
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
