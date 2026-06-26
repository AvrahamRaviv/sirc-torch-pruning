"""Reproduce the cross-arch × scorer retention table — 4 native archs × 8 scorers.

Three modes:

  --mode submit   (CLUSTER) write one run_ddp.sh per arch×scorer cell and submit each as an
                  isolated docker/DDP job via /algo/ws/shared/remote-gpu/run_docker_gpu.sh
                  (same envelope as run_ddp.py). Jobs run async on the queue; each writes
                  <out_dir>/<save_tag>_prune.json (holds pre_ft_val_acc + macs_pct).
  --mode collect  (CLUSTER) once jobs finished: scan the out_dirs, read every <save_tag>_prune.json,
                  build the ledger (results_table.jsonl) + print the markdown table. Then plot with
                  plot_table.py.
  --mode local    (LAPTOP) run each cell as a plain subprocess (single GPU/MPS, --disable_ddp), full
                  in-process; for smoke/quick checks only. --smoke = 2 calib batches + 512 val.

The cluster has NO direct-python entrypoint — every run is a .sh submitted to the GPU queue — so the
table is two phases on the cluster: submit all 32 jobs, wait, then collect. Local mode keeps the old
self-contained behavior for laptop smoke.

Scorers (8):
  magnitude  vbp(=variance)  prop(=prop_rel_p2)  cov_comp  cov_meas  iter_comp  iter_meas  nci(=tp_variance)

Cluster usage:
  # 0. dry-run: print the .sh + submission command for every cell, submit nothing
  python reproduce_table.py --mode submit --run_name REPRO_v1 --dry_run
  # 1. submit all 32 jobs (skips cells whose _prune.json already exists)
  python reproduce_table.py --mode submit --run_name REPRO_v1
  # 2. after the queue drains, collect + table
  python reproduce_table.py --mode collect --run_name REPRO_v1
  python plot_table.py --out cluster.png --title "CLUSTER 4x8"

EDIT the CLUSTER dict + ARCHS[...]['cluster_weights'] paths to match your filesystem before submit.

Local usage:
  python reproduce_table.py --mode local --smoke --data_path /tmp/in --weights_dir ~/.cache/torch/hub/checkpoints
  python plot_table.py --ledger results_table_smoke.jsonl --out smoke.png --title "LOCAL smoke"

mobilenet_v1 is timm-only (BatchNormAct2d fused-ReLU6), not buildable by normnet_main — run separately
via research/harness_mbv1.py.
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
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))      # torch_pruning package root (laptop)
MAIN = os.path.join(HERE, "normnet_main.py")
LEDGER = os.path.join(HERE, "results_table.jsonl")

# --------------------------------------------------------------------- CLUSTER config (EDIT THESE)
# Mirrors run_ddp.py. The submit mode writes a run_ddp.sh per cell and submits it with this envelope.
CLUSTER = dict(
    repo="/home/avrahamra/PycharmProjects/sirc-torch-pruning",      # cd target inside the .sh
    data_path="/algo/NetOptimization/outputs/VBP/",                 # ImageNet root on the cluster
    out_root="/algo/NetOptimization/outputs/NORMNET/REPRO_TABLE",   # all cells land under here/<run_name>
    nproc=1,                                                        # GPUs per job (retention = calib+eval, 1 GPU is enough)
    # remote-gpu submission envelope — mirrors the PROVEN-working run_ddp.py command EXACTLY
    # (shell=True flat string). Two things run_docker_gpu.sh requires that earlier versions broke:
    #   * resources = ONE -R blob with a nested ` -R ` joiner: -R 'select[gpu_hm] -R select[...]'
    #     (separate -R flags get mishandled → malformed bsub → job never submitted)
    #   * -E value quoted: -E 'force_python_3=yes'
    submit_sh="/algo/ws/shared/remote-gpu/run_docker_gpu.sh",
    img="gitlab-srv:4567/od-alg/od_next_gen:v1.7.7_tp2",
    queue="gpu_deep_train_low_q",
    mem="50gb", ncpu="10", runlimit="60000", project="VISION",
    mount="/algo/NetOptimization:/algo/NetOptimization",
    resources=["select[gpu_hm]", "select[g_model != RTXA5000]"],     # joined into one -R blob below
    ngpu=1,
)


def submit_cmd(sh_path, desc):
    """Flat shell command for run_docker_gpu.sh — verbatim run_ddp.py shape (shell=True)."""
    c = CLUSTER
    res_blob = " -R ".join(c["resources"])      # 'select[gpu_hm] -R select[g_model != RTXA5000]'
    return (f"{c['submit_sh']} -d {c['img']} -C execute -q {c['queue']} -W working_dir -M {sh_path}"
            f"  -s {c['mem']} -n {c['ncpu']} -o {c['runlimit']} -A '' -p {c['project']}"
            f" -v {c['mount']} -R '{res_blob}' -E 'force_python_3=yes' -x {c['ngpu']} -D '{desc}'")

# --------------------------------------------------------------------- per-arch config
# protocol = the established per-arch recipe (same as the 5×7 table):
#   convnext/deit       : LayerNorm     → no fold
#   resnet50 / mnv2     : native BN      → fold BN into conv (--fold_native_bn --fold_no_reinsert) →
#                                          BN-free, no stale stats. This is the meaningful no-recalib
#                                          path for a BN net (native+no-recalib = stale BN = trivial collapse).
# --no_bn_recalib is applied to EVERY cell via core_flags() — recalib re-fits BN ≈ FT, never in a retention table.
# cluster_weights = path on the CLUSTER (EDIT). weights = laptop filename under --weights_dir (local mode).
ARCHS = OrderedDict([
    ("convnext_t", dict(
        model_type="convnext", cnn_arch="convnext_tiny", model_name="convnext_tiny",
        weights="convnext_tiny_1k_224_ema.pth",
        cluster_weights="/algo/NetOptimization/outputs/NORMNET/ConvNeXt_tiny/convnext_tiny_22k_1k_224.pth",
        mac_target_g=2.94, val_resize=232, protocol=[])),
    ("resnet50", dict(
        model_type="cnn", cnn_arch="resnet50", model_name="resnet50",
        weights="resnet50-0676ba61.pth",
        cluster_weights="/algo/NetOptimization/outputs/NORMNET/ResNet50/resnet50_imagenet1k.pth",
        mac_target_g=2.72, val_resize=256,
        protocol=["--fold_native_bn", "--fold_no_reinsert"])),       # BN-free (fold), NO recalib — like mnv2
    ("mobilenet_v2", dict(
        model_type="cnn", cnn_arch="mobilenet_v2", model_name="mobilenet_v2",
        weights="mobilenet_v2-7ebf99e0.pth",
        cluster_weights="/algo/NetOptimization/outputs/NORMNET/MNv2/mobilenet_v2_weights.pth",
        mac_target_g=0.21, val_resize=232,
        protocol=["--fold_native_bn", "--fold_no_reinsert",
                  "--max_prune_ratio", "0.8"])),   # per-layer cap from the proven mnv2 recipe
    ("deit_tiny", dict(
        model_type="vit", cnn_arch="deit_tiny", model_name="facebook/deit-tiny-patch16-224",
        # local HF dir (config.json + *.safetensors) → from_pretrained(dir, local_files_only=True). EDIT.
        weights=None, cluster_weights="/algo/NetOptimization/outputs/NORMNET/DeiT/deit_tiny",
        mac_target_g=0.95, val_resize=224, protocol=[])),
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


# ===================================================================== shared helpers
def core_flags(args):
    """normnet_main flags common to every cell (calib + eval + no-train, retention-only)."""
    return [
        "--global_pruning", "--reparam_variant", "mean", "--imp_normalizer", "width", "--bias_comp",
        "--calib_batches", str(args.calib_batches), "--no_bn_recalib",   # recalib = re-fit BN ≈ FT → OFF
        "--epochs_train", "0", "--epochs_ft", "0", "--epochs_norm_ft", "0", "--skip_norm_eval",
    ]


def subset(args):
    archs = [a for a in (args.archs.split(",") if args.archs else ARCHS) if a in ARCHS]
    scorers = [s for s in (args.scorers.split(",") if args.scorers else SCORERS) if s in SCORERS]
    return archs, scorers


def save_tag(arch, scorer):
    return f"{arch}_{scorer}"


# ===================================================================== CLUSTER: submit
def build_sh(arch, scorer, args, out_dir):
    """The run_ddp.sh body for one cell: DDP normnet_main, retention-only (epochs_ft 0)."""
    a = ARCHS[arch]
    tag = save_tag(arch, scorer)
    flags = [
        "--model_type", a["model_type"], "--cnn_arch", a["cnn_arch"],
        "--model_name", a["cluster_weights"],
        "--data_path", CLUSTER["data_path"],
        "--mac_target_g", str(a["mac_target_g"]), "--val_resize", str(a["val_resize"]),
        "--save_dir", out_dir, "--save_tag", tag,
    ] + core_flags(args) + a["protocol"] + SCORERS[scorer]
    line = (f"python3 -m torch.distributed.launch --nproc_per_node={CLUSTER['nproc']} "
            f"{CLUSTER['repo']}/benchmarks/vbp/normnet_main.py \\\n    "
            + " ".join(flags))
    return f"#!/bin/bash\nset -e\ncd {CLUSTER['repo']}\n{line}\n"


def submit_cell(arch, scorer, args):
    tag = save_tag(arch, scorer)
    out_dir = os.path.join(CLUSTER["out_root"], args.run_name, tag)
    sh_path = os.path.join(out_dir, "run_ddp.sh")
    cmd = submit_cmd(sh_path, f"REPRO {arch} {scorer}")
    if args.dry_run:                          # pure: render, touch no filesystem (cluster paths RO on laptop)
        print(f"\n# ===== {tag} =====\n# sh: {sh_path}")
        print(build_sh(arch, scorer, args, out_dir))
        print("# submit:\n" + cmd)
        return
    if not args.force and os.path.exists(os.path.join(out_dir, f"{tag}_prune.json")):
        print(f"[skip] {tag} (prune.json exists)")
        return
    os.makedirs(out_dir, exist_ok=True)
    try:
        os.chmod(out_dir, 0o777)
    except OSError:
        pass
    with open(sh_path, "w") as f:
        f.write(build_sh(arch, scorer, args, out_dir))
    os.chmod(sh_path, 0o777)
    print(f"[submit] {tag}")
    subprocess.run(cmd, shell=True)


def mode_submit(args):
    archs, scorers = subset(args)
    if not args.dry_run:
        os.makedirs(os.path.join(CLUSTER["out_root"], args.run_name), exist_ok=True)
    print(f"[SUBMIT] run_name={args.run_name}  {len(archs)}×{len(scorers)} cells  "
          f"out={os.path.join(CLUSTER['out_root'], args.run_name)}")
    for arch in archs:
        for sc in scorers:
            submit_cell(arch, sc, args)
    if not args.dry_run:
        print("\nAll submitted. Wait for the queue, then:\n  "
              f"python reproduce_table.py --mode collect --run_name {args.run_name}")


# ===================================================================== CLUSTER: collect
def collect_cell(arch, scorer, args):
    tag = save_tag(arch, scorer)
    pj = os.path.join(CLUSTER["out_root"], args.run_name, tag, f"{tag}_prune.json")
    if not os.path.exists(pj):
        return dict(arch=arch, scorer=scorer, acc=None, mac_pct=None, error="no prune.json")
    d = json.load(open(pj))
    return dict(arch=arch, scorer=scorer, acc=d.get("pre_ft_val_acc"),
                mac_g=d.get("macs_g"), mac_pct=int(round(d.get("macs_pct", 0))))


def mode_collect(args):
    archs, scorers = subset(args)
    done = {}
    with open(LEDGER, "w") as f:
        for arch in archs:
            for sc in scorers:
                rec = collect_cell(arch, sc, args)
                done[(arch, sc)] = rec
                if rec.get("acc") is not None:
                    f.write(json.dumps(rec) + "\n")
    n = sum(1 for v in done.values() if v.get("acc") is not None)
    print(f"[COLLECT] {n}/{len(done)} cells found → {LEDGER}")
    print_table(done, archs, scorers)
    print(f"\nplot:  python plot_table.py --ledger {os.path.basename(LEDGER)} --out cluster.png")


# ===================================================================== LOCAL: subprocess
def _subenv():
    env = dict(os.environ)
    extra = os.pathsep.join([REPO, HERE])
    env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def build_argv_local(arch, scorer, args, save_dir):
    a = ARCHS[arch]
    model_name = a["model_name"]
    if a["weights"] and args.weights_dir:
        wp = os.path.join(os.path.expanduser(args.weights_dir), a["weights"])
        if os.path.exists(wp):
            model_name = wp
    argv = [sys.executable, MAIN,
            "--model_type", a["model_type"], "--cnn_arch", a["cnn_arch"], "--model_name", model_name,
            "--data_path", args.data_path,
            "--mac_target_g", str(a["mac_target_g"]), "--val_resize", str(a["val_resize"]),
            "--save_dir", save_dir, "--save_tag", save_tag(arch, scorer),
            "--val_batch_size", str(args.val_batch_size), "--val_limit", str(args.val_limit),
            "--num_workers", str(args.num_workers), "--disable_ddp"]
    return argv + core_flags(args) + a["protocol"] + SCORERS[scorer]


def run_cell_local(arch, scorer, args):
    with tempfile.TemporaryDirectory() as sd:
        argv = build_argv_local(arch, scorer, args, sd)
        if args.dry_run:
            print("  " + " ".join(argv[1:]))
            return None
        t0 = time.time()
        p = subprocess.run(argv, capture_output=True, text=True, env=_subenv())
        out = p.stdout + "\n" + p.stderr
        accs = list(PRE_FT_RE.finditer(out))
        if not accs:
            tail = "\n".join(out.strip().splitlines()[-12:])
            return dict(arch=arch, scorer=scorer, acc=None, mac_pct=None,
                        secs=round(time.time() - t0), error=tail[:2000])
        m = accs[-1]
        return dict(arch=arch, scorer=scorer, acc=float(m.group(1)),
                    mac_g=float(m.group(2)), mac_pct=int(m.group(3)), secs=round(time.time() - t0))


def load_ledger():
    done = {}
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            line = line.strip()
            if line:
                d = json.loads(line)
                done[(d["arch"], d["scorer"])] = d
    return done


def mode_local(args):
    global LEDGER
    if args.smoke:
        if args.calib_batches == 50:
            args.calib_batches = 2
        if args.val_limit == 0:
            args.val_limit = 512
        LEDGER = os.path.join(HERE, "results_table_smoke.jsonl")
        print(f"[SMOKE] calib_batches={args.calib_batches} val_limit={args.val_limit} "
              f"ledger={os.path.basename(LEDGER)}", flush=True)
    archs, scorers = subset(args)
    done = load_ledger()
    if not args.data_path and not args.dry_run:
        sys.exit("--data_path required for local mode (or use --dry_run)")
    for arch in archs:
        for sc in scorers:
            if not args.force and (arch, sc) in done and done[(arch, sc)].get("acc") is not None:
                print(f"[skip] {arch}/{sc} = {done[(arch, sc)]['acc']:.4f}", flush=True)
                continue
            print(f"\n##### {arch} / {sc} #####", flush=True)
            rec = run_cell_local(arch, sc, args)
            if rec is None:
                continue
            if rec.get("acc") is not None:
                print(f"  -> acc={rec['acc']:.4f}  {rec.get('mac_pct')}% MAC  ({rec['secs']}s)", flush=True)
            else:
                print(f"  -> FAILED ({rec['secs']}s)\n{rec.get('error', '')}", flush=True)
            with open(LEDGER, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done[(arch, sc)] = rec
    if not args.dry_run:
        print_table(done, archs, scorers)


# ===================================================================== table print
def print_table(done, archs, scorers):
    print("\n" + "=" * 80 + "\nRETENTION TABLE (pre-FT top-1, width norm)\n" + "=" * 80)
    print("| arch | " + " | ".join(scorers) + " |")
    print("|" + "---|" * (len(scorers) + 1))
    for arch in archs:
        cells = []
        for sc in scorers:
            d = done.get((arch, sc))
            cells.append("·" if d is None else ("ERR" if d.get("acc") is None else f"{d['acc']:.3f}"))
        mac = next((done[(arch, s)].get("mac_pct") for s in scorers
                    if done.get((arch, s)) and done[(arch, s)].get("mac_pct")), "?")
        print(f"| {arch} ({mac}%) | " + " | ".join(cells) + " |")


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["submit", "collect", "local"], default="submit")
    ap.add_argument("--archs", default="", help="comma subset of " + ",".join(ARCHS))
    ap.add_argument("--scorers", default="", help="comma subset of " + ",".join(SCORERS))
    ap.add_argument("--calib_batches", type=int, default=50)
    # cluster
    ap.add_argument("--run_name", default="REPRO_v1", help="subdir under CLUSTER out_root")
    ap.add_argument("--force", action="store_true", help="re-submit / re-run even if result exists")
    ap.add_argument("--dry_run", action="store_true", help="print commands/.sh, submit/run nothing")
    # local only
    ap.add_argument("--data_path", default="")
    ap.add_argument("--weights_dir", default="~/.cache/torch/hub/checkpoints")
    ap.add_argument("--val_batch_size", type=int, default=256)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="local: 2 calib batches + 512 val, smoke ledger")
    args = ap.parse_args()

    if args.mode == "submit":
        mode_submit(args)
    elif args.mode == "collect":
        mode_collect(args)
    else:
        mode_local(args)


if __name__ == "__main__":
    main()
