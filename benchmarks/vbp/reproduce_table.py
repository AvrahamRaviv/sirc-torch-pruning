"""Reproduce the cross-arch × scorer retention table on the CLUSTER (full calib + validation).

Single wrapper around normnet_main.py: loops archs × scorers, runs each as an isolated
subprocess (real ImageNet, full calibration + full val), parses the pre-FT top-1, and prints
one markdown table at the end. Append-only ledger (results_table.jsonl) → resumable: re-running
skips cells already done (delete a line to force a re-run).

Covers the 4 normnet_main-native archs. mobilenet_v1 is timm-only (BatchNormAct2d fused-ReLU6)
and not buildable by normnet_main — run it via research/harness_mbv1.py separately.

Scorers (8):
  magnitude  vbp(=variance)  prop(=prop_rel_p2)  cov_comp  cov_meas  iter_comp  iter_meas  nci(=tp_variance)

Usage (cluster):
  python reproduce_table.py --data_path /path/to/imagenet --weights_dir ~/.cache/torch/hub/checkpoints
  python reproduce_table.py --data_path ... --archs convnext_t,resnet50 --scorers cov_comp,nci
  python reproduce_table.py --data_path ... --dry_run        # print the commands, run nothing
  python reproduce_table.py --print_only                     # just reprint the table from the ledger

Adjust ARCHS[...]['mac_target_g'] / weights filenames to match your checkpoints if needed.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))      # torch_pruning package root
MAIN = os.path.join(HERE, "normnet_main.py")
LEDGER = os.path.join(HERE, "results_table.jsonl")


def _subenv():
    """Child env with the repo root + benchmarks/vbp on PYTHONPATH so the subprocess imports
    `torch_pruning` and the sibling vbp modules even when the package isn't pip-installed."""
    env = dict(os.environ)
    extra = os.pathsep.join([REPO, HERE])
    env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env

# --------------------------------------------------------------------- per-arch config
# protocol encodes the established per-arch recipe used by the 5×7 table:
#   convnext/deit : LayerNorm, no BN  → no fold, no recalib
#   resnet50      : native BN         → recalib ON post-prune (omit --no_bn_recalib)
#   mobilenet_v2  : fold native BN    → BN-free, no recalib
# mac_target_g ≈ 0.66–0.75 × dense; edit to match the exact budget you want.
ARCHS = OrderedDict([
    ("convnext_t", dict(
        model_type="convnext", cnn_arch="convnext_tiny", model_name="convnext_tiny",
        weights="convnext_tiny_1k_224_ema.pth", mac_target_g=2.94, val_resize=232,
        protocol=[])),
    ("resnet50", dict(
        model_type="cnn", cnn_arch="resnet50", model_name="resnet50",
        weights="resnet50-0676ba61.pth", mac_target_g=2.72, val_resize=256,
        protocol=[])),                                   # native BN → recalib ON (default)
    ("mobilenet_v2", dict(
        model_type="cnn", cnn_arch="mobilenet_v2", model_name="mobilenet_v2",
        weights="mobilenet_v2-7ebf99e0.pth", mac_target_g=0.21, val_resize=256,
        protocol=["--fold_native_bn", "--fold_no_reinsert", "--no_bn_recalib"])),
    ("deit_tiny", dict(
        model_type="vit", cnn_arch="deit_tiny", model_name="facebook/deit-tiny-patch16-224",
        weights=None, mac_target_g=0.95, val_resize=224,
        protocol=[])),
])

# --------------------------------------------------------------------- scorer flag sets
ITER = ["--prop_iterative", "--prop_iter_drop", "128", "--prop_iter_max_frac", "0.6"]
SCORERS = OrderedDict([
    ("magnitude", ["--scorer", "magnitude"]),
    ("vbp",       ["--scorer", "variance"]),
    ("prop",      ["--scorer", "propagation"]),
    ("cov_comp",  ["--scorer", "propagation", "--prop_cov", "--prop_p", "2"]),
    ("cov_meas",  ["--scorer", "propagation", "--prop_cov", "--prop_p", "2", "--prop_measured_var"]),
    ("iter_comp", ["--scorer", "propagation", "--prop_cov", "--prop_p", "2"] + ITER),
    ("iter_meas", ["--scorer", "propagation", "--prop_cov", "--prop_p", "2", "--prop_measured_var"] + ITER),
    ("nci",       ["--scorer", "tp_variance"]),
])

PRE_FT_RE = re.compile(r"pre-FT acc=([0-9.]+).*?([0-9.]+)G\s*\((\d+)%\)")


def common_flags(args):
    return [
        "--data_path", args.data_path, "--global_pruning",
        "--reparam_variant", "mean", "--imp_normalizer", "width", "--bias_comp",
        "--calib_batches", str(args.calib_batches), "--val_batch_size", str(args.val_batch_size),
        "--val_limit", str(args.val_limit), "--num_workers", str(args.num_workers),
        "--epochs_train", "0", "--epochs_ft", "0", "--epochs_norm_ft", "0",
        "--skip_norm_eval", "--disable_ddp",
    ]


def build_argv(arch, scorer, args, save_dir):
    a = ARCHS[arch]
    model_name = a["model_name"]
    if a["weights"] and args.weights_dir:
        wp = os.path.join(os.path.expanduser(args.weights_dir), a["weights"])
        if os.path.exists(wp):
            model_name = wp
    argv = [sys.executable, MAIN,
            "--model_type", a["model_type"], "--cnn_arch", a["cnn_arch"],
            "--model_name", model_name,
            "--mac_target_g", str(a["mac_target_g"]), "--val_resize", str(a["val_resize"]),
            "--save_dir", save_dir, "--save_tag", f"{arch}_{scorer}"]
    return argv + common_flags(args) + a["protocol"] + SCORERS[scorer]


def load_ledger():
    done = {}
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            done[(d["arch"], d["scorer"])] = d
    return done


def append_ledger(rec):
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_cell(arch, scorer, args):
    with tempfile.TemporaryDirectory() as sd:
        argv = build_argv(arch, scorer, args, sd)
        if args.dry_run:
            print("  " + " ".join(argv[1:]))
            return None
        t0 = time.time()
        p = subprocess.run(argv, capture_output=True, text=True, env=_subenv())
        out = p.stdout + "\n" + p.stderr
        accs = [m for m in PRE_FT_RE.finditer(out)]
        if not accs:
            tail = "\n".join(out.strip().splitlines()[-12:])
            return dict(arch=arch, scorer=scorer, acc=None, mac_pct=None,
                        secs=round(time.time() - t0), error=tail[:2000])
        m = accs[-1]                              # last pre-FT line = the applied prune
        return dict(arch=arch, scorer=scorer, acc=float(m.group(1)),
                    mac_g=float(m.group(2)), mac_pct=int(m.group(3)),
                    secs=round(time.time() - t0))


def print_table(done, archs, scorers):
    print("\n" + "=" * 80 + "\nRETENTION TABLE (pre-FT top-1, width norm)\n" + "=" * 80)
    head = "| arch | " + " | ".join(scorers) + " |"
    print(head)
    print("|" + "---|" * (len(scorers) + 1))
    for arch in archs:
        cells = []
        for sc in scorers:
            d = done.get((arch, sc))
            if d is None:
                cells.append("·")
            elif d.get("acc") is None:
                cells.append("ERR")
            else:
                cells.append(f"{d['acc']:.3f}")
        mac = next((done[(arch, s)].get("mac_pct") for s in scorers
                    if done.get((arch, s)) and done[(arch, s)].get("mac_pct")), "?")
        print(f"| {arch} ({mac}%) | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="", help="ImageNet root (train/ + val/)")
    ap.add_argument("--weights_dir", default="~/.cache/torch/hub/checkpoints")
    ap.add_argument("--archs", default="", help="comma subset of " + ",".join(ARCHS))
    ap.add_argument("--scorers", default="", help="comma subset of " + ",".join(SCORERS))
    ap.add_argument("--calib_batches", type=int, default=50)
    ap.add_argument("--val_batch_size", type=int, default=256)
    ap.add_argument("--val_limit", type=int, default=0, help="cap val to first N (0=full)")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true",
                    help="fast end-to-end test: 2 calib batches + 512 val samples per cell "
                         "(overrides --calib_batches/--val_limit unless you set them). Run this "
                         "FIRST on the cluster to verify every arch×scorer cell loads/calibs/"
                         "prunes/evals before the full table. Uses a separate ledger.")
    ap.add_argument("--dry_run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--print_only", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if in the ledger")
    args = ap.parse_args()

    global LEDGER
    if args.smoke:
        if args.calib_batches == 50:
            args.calib_batches = 2
        if args.val_limit == 0:
            args.val_limit = 512
        LEDGER = os.path.join(HERE, "results_table_smoke.jsonl")
        print(f"[SMOKE] calib_batches={args.calib_batches} val_limit={args.val_limit} "
              f"ledger={os.path.basename(LEDGER)}", flush=True)

    archs = [a for a in (args.archs.split(",") if args.archs else ARCHS) if a in ARCHS]
    scorers = [s for s in (args.scorers.split(",") if args.scorers else SCORERS) if s in SCORERS]
    done = load_ledger()

    if args.print_only:
        print_table(done, archs, scorers)
        return
    if not args.data_path and not args.dry_run:
        sys.exit("--data_path required (or use --dry_run)")

    for arch in archs:
        for sc in scorers:
            if not args.force and (arch, sc) in done and done[(arch, sc)].get("acc") is not None:
                print(f"[skip] {arch}/{sc} = {done[(arch, sc)]['acc']:.4f}", flush=True)
                continue
            print(f"\n##### {arch} / {sc} #####", flush=True)
            rec = run_cell(arch, sc, args)
            if rec is None:                      # dry_run
                continue
            if rec.get("acc") is not None:
                print(f"  -> acc={rec['acc']:.4f}  {rec.get('mac_pct')}% MAC  ({rec['secs']}s)",
                      flush=True)
            else:
                print(f"  -> FAILED ({rec['secs']}s)\n{rec.get('error', '')}", flush=True)
            append_ledger(rec)
            done[(arch, sc)] = rec

    if not args.dry_run:
        print_table(done, archs, scorers)


if __name__ == "__main__":
    main()
