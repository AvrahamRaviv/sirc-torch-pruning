#!/usr/bin/env python3
"""Generate the 4-arch x 5-scorer FT grid: one <arch_root>/<scorer>/run_ddp.sh per cell.

Run ON THE CLUSTER (creates dirs under /algo/.../NORMNET/<arch>/). Pure stdlib, no torch.

    python3 gen_ft_grid.py            # write all 20 run_ddp.sh
    python3 gen_ft_grid.py --dry_run  # print, write nothing
    python3 gen_ft_grid.py --archs resnet50,mobilenet_v1   # subset

Each cell = prune (scorer) + full FT with the arch's SOTA-isomorphic recipe. Numbers are meant to
compare to the published pruning papers (DepGraph / Isomorphic-Pruning / AMC), NOT to KD-accelerated
short runs. Toggle USE_KD below if you want KD (papers do NOT use it -> off for isomorphic fidelity).

Sources for the recipes: DepGraph + Isomorphic-Pruning official finetune scripts (VainF), AMC
(mit-han-lab). See FT_SCOREBOARD.md.
"""
import argparse
import os

# ------------------------------------------------------------------ launch / global config
REPO = "/home/avrahamra/PycharmProjects/sirc-torch-pruning"
DATA_PATH = "/algo/NetOptimization/outputs/VBP/"
NGPU = 4                       # FT is train-heavy -> 4 GPUs
TRAIN_BS = 64                  # per-GPU; 64 x 4 = 256 effective = the recipe batch (no LR rescale)
CALIB_BATCHES = 200            # scorer cov/var calibration (200 ~= 5000 in rank; 5000 wastes ~1h)
RECALIB_BATCHES = 50           # post-prune BN recalibration (no-grad re-estimation; no-op on LN/convnext)
USE_KD = True                  # KD on (user choice; teacher = the dense --model_name)

# ------------------------------------------------------------------ per-arch config
#   root         = the arch folder you already have (ckpt lives inside)
#   ckpt         = weights file inside root (VERIFY version by sha before running)
#   model_type / cnn_arch / val_resize  = from reproduce_table.ARCHS (matches the retention table)
#   mac          = SOTA-isomorphic MAC target (GMACs)
#   cap          = per-layer --max_prune_ratio (mobilenets only; narrow layers)
#   recipe       = the finetune hyperparams (isomorphic per arch)
ARCHS = {
    "resnet50": dict(
        root="/algo/NetOptimization/outputs/NORMNET/ResNet50",
        ckpt="resnet50_imagenet1k.pth", model_type="cnn", cnn_arch="resnet50",
        val_resize=256, mac=2.00, cap=None,
        recipe=["--opt", "sgd", "--epochs_ft", "100", "--lr_ft", "0.04",
                "--lr_schedule", "step", "--lr_step_size", "30", "--lr_gamma", "0.1",
                "--wd", "1e-4", "--momentum", "0.9"]),   # Isomorphic R50@2.0G: 100ep (DepGraph=90)
    "mobilenet_v2": dict(
        root="/algo/NetOptimization/outputs/NORMNET/MNv2",
        ckpt="mobilenet_v2_weights.pth", model_type="cnn", cnn_arch="mobilenet_v2",
        val_resize=256, mac=0.15, cap="0.8", interior=True,
        train_bs=128, amp=True, kd_off=True,
        # SOTA-MATCH recipe. Isomorphic-Pruning MNv2-0.15G (68.91 top-1) = ONE-SHOT taylor global
        # prune (NO sparsity learning, same class as our scorers) + a PLAIN 300ep finetune. Their
        # exact FT (scripts/finetuning/mobilenet_v2_0.15G.sh):
        #   300ep, bs512x4=2048, lr0.036 cosine, warmup0, wd2e-5, --amp, bilinear, NO KD, NO aug.
        # We match it so the ONLY difference is the scorer (propagation vs taylor) → clean head-to-
        # head at the SAME 0.15G budget + SAME FT. Deltas vs theirs: bs128x4=512 (our GPUs), so lr
        # linear-scaled 0.036*512/2048=0.009; bf16 --amp (accuracy-neutral, ~2x faster); KD OFF
        # (theirs has none — don't let KD confound the comparison). interior_only kept = our
        # method's residual-stream protection (part of what we're benchmarking). Earlier 150ep
        # runs capped ~56 purely because half-budget + step x0.98 dumped lr; NOT a method ceiling
        # — no SL premium exists for this number, so 68.91 is a fair reachable target.
        recipe=["--opt", "sgd", "--epochs_ft", "300", "--lr_ft", "0.009",
                "--lr_schedule", "cosine", "--ft_eta_min", "1e-6",
                "--wd", "2e-5", "--momentum", "0.9"]),
    "convnext_t": dict(
        root="/algo/NetOptimization/outputs/NORMNET/ConvNeXt_tiny",
        ckpt="convnext_tiny_22k_1k_224.pth", model_type="convnext", cnn_arch="convnext_tiny",
        val_resize=232, mac=2.94, cap=None, kd=("0.0", "4.0"),
        # user's validated 20ep convnext FT recipe (AdamW cosine, warmup 3, its own KD alpha/T).
        recipe=["--opt", "adamw", "--epochs_ft", "20", "--lr_ft", "1e-4",
                "--lr_schedule", "cosine", "--ft_warmup_epochs", "3", "--ft_eta_min", "2e-5",
                "--wd", "0"]),
    "mobilenet_v1": dict(
        root="/algo/NetOptimization/outputs/NORMNET/MNv1",
        ckpt="mobilenet_v1.safetensors", model_type="cnn", cnn_arch="mobilenet_v1",
        val_resize=256, mac=0.391, cap="0.8",   # 0.391/0.584 dense = 67% kept (matches retention table)
        # pre-FT already ~0.50 -> finetune (recover), NOT scratch-retrain. Eval-collapse was the BN
        # recalib-momentum bug (fixed 18cfc5f9), NOT lr. First healthy run (lr0.001, cosine90):
        # cov/iter plateau ~0.70 by e12-15 -> 90ep wasteful, lr0.001 crawls early. Now: lr0.008
        # (higher peak = steeper early climb, cosine lands soft), 30ep (3x cheaper; tail still lets
        # slower scorers nci/vbp finish their climb). No warmup needed (recover start is stable).
        recipe=["--opt", "sgd", "--epochs_ft", "30", "--lr_ft", "0.008",
                "--lr_schedule", "cosine", "--ft_eta_min", "1e-6",
                "--wd", "4e-5", "--momentum", "0.9"]),
}

# ------------------------------------------------------------------ scorer -> prune flags
#   (from reproduce_table.BASES; vbp=variance, nci=tp_variance -- trust the mapping that made the
#    retention table). All width normalizer. cov/iter = the variance-covariance propagation family.
SCORERS = {
    "magnitude": ["--scorer", "magnitude"],
    "vbp":       ["--scorer", "variance"],
    "nci":       ["--scorer", "tp_variance"],
    "cov":       ["--scorer", "propagation", "--prop_cov", "--prop_p", "2"],
    "iter":      ["--scorer", "propagation", "--prop_cov", "--prop_p", "2",
                  "--prop_iterative", "--prop_iter_drop", "128", "--prop_iter_max_frac", "0.6"],
}


def core_flags(a):
    """Flags common to every cell: prune protocol B_native (mean-fold, no recalib) + FT scaffolding."""
    f = ["--global_pruning", "--reparam_variant", "mean", "--bias_comp",
         "--recalib_batches", str(RECALIB_BATCHES), "--skip_norm_eval",
         "--calib_batches", str(CALIB_BATCHES),
         "--epochs_train", "0", "--epochs_norm_ft", "0",
         "--imp_normalizer", "width",
         "--val_resize", str(a["val_resize"]), "--mac_target_g", str(a["mac"]),
         "--train_batch_size", str(a.get("train_bs", TRAIN_BS))]  # per-arch batch (mnv2=128)
    if a["cap"]:
        f += ["--max_prune_ratio", a["cap"]]
    if a.get("interior"):
        f += ["--interior_only"]         # protect residual-stream out-channels (arch opt-in)
    if a.get("amp"):
        f += ["--amp"]                   # bf16 FT (mnv2 SOTA-match; accuracy-neutral)
    if USE_KD and not a.get("kd_off"):   # kd_off per arch (mnv2 matches SOTA recipe = no KD)
        alpha, T = a.get("kd", ("0.5", "2.0"))          # per-arch KD; convnext = (0.0, 4.0)
        f += ["--use_kd", "--kd_alpha", alpha, "--kd_T", T]
    return f


def build_sh(arch, scorer):
    a = ARCHS[arch]
    save_dir = os.path.join(a["root"], scorer)
    tag = f"{arch}_ft_{scorer}"
    flags = ["--model_type", a["model_type"], "--cnn_arch", a["cnn_arch"],
             "--model_name", os.path.join(a["root"], a["ckpt"]),
             "--data_path", DATA_PATH,
             "--save_dir", save_dir, "--save_tag", tag]
    flags += core_flags(a) + SCORERS[scorer] + a["recipe"]
    line = (f"python3 -m torch.distributed.launch --nproc_per_node={NGPU} "
            f"{REPO}/benchmarks/vbp/normnet_main.py \\\n    " + " ".join(flags))
    return save_dir, f"#!/bin/bash\nset -e\ncd {REPO}\n{line}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default=",".join(ARCHS), help="comma list; default all 4")
    ap.add_argument("--scorers", default=",".join(SCORERS), help="comma list; default all 5")
    ap.add_argument("--dry_run", action="store_true", help="print, write nothing")
    args = ap.parse_args()

    archs = [x for x in args.archs.split(",") if x]
    scorers = [x for x in args.scorers.split(",") if x]
    made = []
    for arch in archs:
        for scorer in scorers:
            save_dir, sh = build_sh(arch, scorer)
            sh_path = os.path.join(save_dir, "run_ddp.sh")
            if args.dry_run:
                print(f"\n# ===== {arch} / {scorer}  ->  {sh_path} =====\n{sh}")
                continue
            os.makedirs(save_dir, exist_ok=True)
            with open(sh_path, "w") as fh:
                fh.write(sh)
            os.chmod(sh_path, 0o755)
            made.append(sh_path)
    if not args.dry_run:
        print(f"wrote {len(made)} run_ddp.sh:")
        for p in made:
            print("  " + p)
        print("\nsubmit each with your usual wrapper, e.g.:")
        print("  for f in " + " ".join(os.path.dirname(p) for p in made) + "; do "
              "(cd $f && <your_submit> run_ddp.sh); done")


if __name__ == "__main__":
    main()
