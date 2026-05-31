"""
Generate E2 prune-and-FT experiment folders + run_ddp.sh templates, same fashion as the
training sweep. Each folder gets a run_ddp.sh whose last token is `--save_dir <out_dir>`
so run_ddp.py patches it in place at submit time:

    python benchmarks/vbp/generate_e2_experiments.py        # writes the folders
    python run_ddp.py --out_dir_name E2_per_layer_r50       # submit one
    ... (repeat per folder, or loop)

E2 is single-process (pruning is a structural edit — no torchrun). run_ddp.py still
submits the bash to a GPU node; the bash just runs `python prune_e2.py ...`.

Scorers share one pruner / reduction / normalizer — only the per-channel base score
differs (magnitude ‖v‖ vs per_layer ‖σ·v‖ vs propagation I^l). Uniform per-layer ratio
(no --global_pruning) so sparsity is matched and the comparison isolates ranking quality.
Add "global" variants by flipping GLOBAL below.
"""
import os
import stat

# --- remote paths (cluster) --- (override BASE_OUT via env for local dry-runs)
BASE_OUT = os.environ.get("E2_BASE_OUT", "/algo/NetOptimization/outputs/NORMNET/ResNet50")
SCRIPT = "/home/avrahamra/PycharmProjects/sirc-torch-pruning/benchmarks/vbp/prune_e2.py"
DATA = "/algo/NetOptimization/outputs/VBP/"
# Dense checkpoint to prune = the λ=1e-4 winner. Use the VNR file: it reloads to
# 0.808 (verified), whereas the merged_biased file reloads to ~0.001 (broken save on
# the cluster — load_normnet_checkpoint auto-detects the vnr format from the sidecar
# and reconstructs weight=v, bias=m−v·μ_x correctly).
CKPT = f"{BASE_OUT}/RN_mean_l1e-4/RN_mean_l1e-4_vnr.pth"

# --- experiment matrix ---
SCORERS = ["magnitude", "per_layer", "propagation"]
RATIOS = [0.50, 0.75]          # pruning_ratio; 0.75 is where ranking separates
GLOBAL = False                 # True → cross-layer global ranking (criterion-4 setting)

# --- shared FT recipe (mirror the λ-sweep: sgd 5e-4 cosine, KD, 10ep) ---
COMMON = (
    f"--model_type cnn --cnn_arch resnet50 --checkpoint {CKPT} --data_path {DATA} "
    f"--epochs_ft 10 --opt sgd --lr 5e-4 --wd 1e-4 --momentum 0.9 --ft_eta_min 1e-6 "
    f"--use_kd --kd_alpha 0.25 --kd_T 4.0 --train_batch_size 128 --val_resize 232 "
    f"--calib_batches 50"
)

SH_TEMPLATE = (
    "#!/bin/bash\n"
    "set -e\n"
    "cd /home/avrahamra/PycharmProjects/sirc-torch-pruning\n"
    "python {script} {common} --scorer {scorer} --pruning_ratio {ratio}{glob} "
    "--save_tag {tag} --save_dir {out_dir}\n"
)


def main():
    glob_flag = " --global_pruning" if GLOBAL else ""
    made = []
    for scorer in SCORERS:
        for ratio in RATIOS:
            r = int(round(ratio * 100))
            tag = f"E2_{scorer}_r{r}" + ("_glob" if GLOBAL else "")
            out_dir = os.path.join(BASE_OUT, tag)
            os.makedirs(out_dir, exist_ok=True)
            sh_path = os.path.join(out_dir, "run_ddp.sh")
            # out_dir written here is a placeholder; run_ddp.py re-patches --save_dir
            # to the resolved path at submit time (kept last for the regex).
            text = SH_TEMPLATE.format(
                script=SCRIPT, common=COMMON, scorer=scorer, ratio=ratio,
                glob=glob_flag, tag=tag, out_dir=out_dir)
            with open(sh_path, "w") as f:
                f.write(text)
            os.chmod(sh_path, os.stat(sh_path).st_mode | stat.S_IEXEC)
            made.append(tag)
            print(f"wrote {sh_path}")
    print(f"\n{len(made)} experiments. Submit each:")
    for tag in made:
        print(f"  python run_ddp.py --out_dir_name {tag}")


if __name__ == "__main__":
    main()
