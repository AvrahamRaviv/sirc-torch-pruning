"""
Generate channel-score DUMP experiment folders + run_ddp.sh, same fashion as
generate_e2_experiments. Each folder gets a run_ddp.sh whose last token is
`--save_dir <out_dir>` so run_ddp.py patches it at submit time:

    python benchmarks/vbp/generate_channel_dumps_experiments.py        # writes the folders
    python run_ddp_ch.py --out_dir_name CD_rn50_glob_mean --arch resnet50   # submit (1 GPU)
    ... (repeat per folder, or loop the printed list)

SINGLE-GPU: submit with run_ddp_ch.py (the -x 1 launcher), NOT run_ddp.py (-x 4). The bash
runs plain `python normnet_standalone.py ...` (no torchrun / DDP). No training (epochs 0) —
calibrate sigma, score, capture prune mask, dump.

Matrix: {resnet50, convnext_tiny} x {global, local} x {mean, none normalizer}. Each run
dumps all 4 criteria (magnitude, nci, rel, nonrel) -> 4 JSON per folder, 32 total. Folders
go under the per-arch root (must match run_ddp_ch.OUT_ROOTS so --arch resolves the path).
Point ExpHandler at the NORMNET root (it globs **/*_channel_scores.json). Scores are
config-independent; only the `kept` mask differs across global/local/normalizer.
"""
import os
import stat

# --- remote paths (cluster). Per-arch roots MUST match run_ddp_ch.py OUT_ROOTS. ---
ROOTS = {
    "resnet50":      os.environ.get("CD_RN50_ROOT", "/algo/NetOptimization/outputs/NORMNET/ResNet50"),
    "convnext_tiny": os.environ.get("CD_CNX_ROOT",  "/algo/NetOptimization/outputs/NORMNET/ConvNeXt-T"),
}
REPO = os.environ.get("EXP_REPO", "/home/avrahamra/PycharmProjects/sirc-torch-pruning")
SCRIPT = f"{REPO}/benchmarks/vbp/normnet_standalone.py"
DATA = "/algo/NetOptimization/outputs/VBP/"        # holds {train,val}_samples.pkl (FastImageNet)

# Plain imagenet1k state_dict per model (loaded via --ckpt, strict=False). resnet50 uses the
# cluster ckpt; convnext falls back to torchvision pretrained (no --ckpt) if none given.
CKPT_FOR = {
    "resnet50": os.environ.get("CD_RN50_CKPT",
                               f"{DATA}ResNet50_TP/resnet50_imagenet1k.pth"),
    "convnext_tiny": os.environ.get("CD_CNX_CKPT", ""),   # "" -> torchvision pretrained
}
SHORT = {"resnet50": "rn50", "convnext_tiny": "cnx"}

# --- experiment matrix ---
MODELS = ["resnet50", "convnext_tiny"]
GLOBALS = [True, False]                 # global cross-layer vs local per-layer ratio
NORMALIZERS = ["mean", "none"]          # per-layer equalizer vs raw cross-layer scale
RATIO = float(os.environ.get("CD_RATIO", "0.5"))
CALIB = int(os.environ.get("CD_CALIB", "50"))
VAL_RESIZE = int(os.environ.get("CD_VAL_RESIZE", "232"))
MODES = "magnitude,nci,rel,nonrel"

COMMON = (
    f"--dataset imagenet --data_path {DATA} --val_resize {VAL_RESIZE} "
    f"--batch_size 128 --epochs 0 --epochs_ft 0 --dump_scores --limit_batches 2 "
    f"--calib_batches {CALIB} --pruning_ratio {RATIO} --modes {MODES}"
)

SH_TEMPLATE = (
    "#!/bin/bash\n"
    "set -e\n"
    f"cd {REPO}\n"
    "python {script} --model {model}{ckpt} {common} {glob} --normalizer {norm} "
    "--tag {tag} --save_dir {out_dir}\n"
)


def main():
    made = []
    for model in MODELS:
        ck = CKPT_FOR.get(model, "")
        ckpt_arg = f" --ckpt {ck}" if ck else ""
        for is_global in GLOBALS:
            for norm in NORMALIZERS:
                gtag = "glob" if is_global else "loc"
                glob_flag = "--global_prune" if is_global else "--local"
                tag = f"CD_{SHORT[model]}_{gtag}_{norm}"
                out_dir = os.path.join(ROOTS[model], tag)        # per-arch root
                os.makedirs(out_dir, exist_ok=True)
                sh_path = os.path.join(out_dir, "run_ddp.sh")
                # out_dir placeholder; run_ddp_ch.py re-patches --save_dir (kept last).
                text = SH_TEMPLATE.format(
                    script=SCRIPT, model=model, ckpt=ckpt_arg, common=COMMON,
                    glob=glob_flag, norm=norm, tag=tag, out_dir=out_dir)
                with open(sh_path, "w") as f:
                    f.write(text)
                os.chmod(sh_path, os.stat(sh_path).st_mode | stat.S_IEXEC)
                made.append((tag, model))
                print(f"wrote {sh_path}")
    print(f"\n{len(made)} dump experiments (ratio={RATIO}, calib={CALIB}). Submit each (1 GPU):")
    for tag, model in made:
        print(f"  python run_ddp_ch.py --out_dir_name {tag} --arch {model}")


if __name__ == "__main__":
    main()
