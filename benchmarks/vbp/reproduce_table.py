"""Cross-arch × scorer pruning SWEEP — retention vs compression frontier + targeted ablations.

Scales from the 4×8 table to ~1000 isolated jobs to exploit idle cluster time. Each job = one
normnet_main run (pure pre-FT retention: --no_bn_recalib, epochs_ft 0), submitted as its own .sh to
the GPU queue (run_docker_gpu.sh envelope, single GPU). Resumable: a cell is skipped if its
<tag>_prune.json already exists. collect reads every prune.json → one rich ledger (results_table.jsonl)
with all sweep axes per row → plot_table.py draws the per-arch retention-vs-MAC curves.

Axes (a spec = one point in this grid):
  arch        convnext_t, resnet50, mobilenet_v2, deit_tiny
  scorer base magnitude vbp nci  | prop cov iter   (last three = the variance-propagation family = focus)
  measured    cov/iter only: computed (w̃Σ̂w̃ᵀ) vs measured (÷ σ_out²)
  prop_p      1 vs 2 (mass-conservation: p=2 column-stochastic)
  fold        BN nets only: fold Conv→BN (BN-free, no recalib) vs native
  mac_frac    fraction of dense MACs (the compression curve)
  normalizer  width vs none      (ablation)
  relative    relative vs --prop_non_relative   (ablation)
  iter knobs  --prop_iter_drop / --prop_iter_max_frac   (ablation, iter only)

Blocks:
  curve     : every arch × fold × all 8 scorer variants × the full mac_frac grid (the headline frontier)
  ablation  : at --ref_frac, focus scorers only, vary ONE knob at a time (p=1, normalizer=none,
              non-relative, iter drop/max_frac)

Cluster usage:
  python reproduce_table.py --mode submit  --run_name SWEEP_v1 --list        # print specs + total, submit nothing
  python reproduce_table.py --mode submit  --run_name SWEEP_v1 --dry_run      # print every .sh + cmd
  python reproduce_table.py --mode submit  --run_name SWEEP_v1                # submit all (skips done)
  python reproduce_table.py --mode collect --run_name SWEEP_v1                # prune.json → ledger + summary
  python plot_table.py --curves --out sweep.png                              # per-arch frontier figure

Tune size: --mac_points (curve density), --blocks curve,ablation, --archs, --max_runs.
EDIT the CLUSTER dict + ARCHS[...]['cluster_weights'] before submit.

mobilenet_v1 is timm-only (not buildable by normnet_main) — run via research/harness_mbv1.py separately.
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
CLUSTER = dict(
    repo="/home/avrahamra/PycharmProjects/sirc-torch-pruning",      # cd target inside the .sh
    data_path="/algo/NetOptimization/outputs/VBP/",                 # ImageNet root on the cluster
    out_root="/algo/NetOptimization/outputs/NORMNET/REPRO_TABLE",   # all cells land under here/<run_name>
    # remote-gpu submission envelope — mirrors the PROVEN-working run_ddp.py command EXACTLY
    # (shell=True flat string). run_docker_gpu.sh needs: resources as ONE -R blob with a nested ` -R `
    # joiner (-R 'select[gpu_hm] -R select[...]'), and a quoted -E. Separate -R flags break bsub.
    submit_sh="/algo/ws/shared/remote-gpu/run_docker_gpu.sh",
    img="gitlab-srv:4567/od-alg/od_next_gen:v1.7.7_tp2",
    queue="gpu_deep_train_low_q",
    mem="50gb", ncpu="10", runlimit="60000", project="VISION",
    mount="/algo/NetOptimization:/algo/NetOptimization",
    resources=["select[gpu_hm]", "select[g_model != RTXA5000]"],     # joined into one -R blob below
    ngpu=1,                                                          # 1 GPU/job (retention = calib+eval)
)


def submit_cmd(sh_path, desc):
    """Flat shell command for run_docker_gpu.sh — verbatim run_ddp.py shape (shell=True)."""
    c = CLUSTER
    res_blob = " -R ".join(c["resources"])
    return (f"{c['submit_sh']} -d {c['img']} -C execute -q {c['queue']} -W working_dir -M {sh_path}"
            f"  -s {c['mem']} -n {c['ncpu']} -o {c['runlimit']} -A '' -p {c['project']}"
            f" -v {c['mount']} -R '{res_blob}' -E 'force_python_3=yes' -x {c['ngpu']} -D '{desc}'")


# --------------------------------------------------------------------- per-arch config
# cluster_weights = path on the CLUSTER (EDIT). weights = laptop filename under --weights_dir (local).
# val_resize = eval preprocessing. FOLDABLE/DENSE_MAC drive the fold axis + mac_frac → mac_target_g.
ARCHS = OrderedDict([
    ("convnext_t", dict(
        model_type="convnext", cnn_arch="convnext_tiny", model_name="convnext_tiny",
        weights="convnext_tiny_1k_224_ema.pth",
        cluster_weights="/algo/NetOptimization/outputs/NORMNET/ConvNeXt_tiny/convnext_tiny_22k_1k_224.pth",
        val_resize=232)),
    ("resnet50", dict(
        model_type="cnn", cnn_arch="resnet50", model_name="resnet50",
        weights="resnet50-0676ba61.pth",
        cluster_weights="/algo/NetOptimization/outputs/NORMNET/ResNet50/resnet50_imagenet1k.pth",
        val_resize=256)),
    ("mobilenet_v2", dict(
        model_type="cnn", cnn_arch="mobilenet_v2", model_name="mobilenet_v2",
        weights="mobilenet_v2-7ebf99e0.pth",
        cluster_weights="/algo/NetOptimization/outputs/NORMNET/MNv2/mobilenet_v2_weights.pth",
        val_resize=232)),
    ("deit_tiny", dict(
        model_type="vit", cnn_arch="deit_tiny", model_name="facebook/deit-tiny-patch16-224",
        weights=None, cluster_weights="/algo/NetOptimization/outputs/NORMNET/DeiT/deit_tiny",
        val_resize=224)),
    ("mobilenet_v1", dict(
        # timm-only (torchvision has no v1); load_model builds timm mobilenetv1_100 + un-fuses
        # BatchNormAct2d. cluster_weights="" ⇒ use timm pretrained (HF cache; stage the cache or
        # allow net access on the cluster). BN net → foldable.
        model_type="cnn", cnn_arch="mobilenet_v1", model_name="",
        weights=None,
        cluster_weights="/algo/NetOptimization/outputs/NORMNET/MNv1/mobilenet_v1.safetensors",
        val_resize=256)),
])

DENSE_MAC = {"convnext_t": 4.45, "resnet50": 4.12, "mobilenet_v2": 0.32, "deit_tiny": 1.44,
             "mobilenet_v1": 0.584}  # GMACs @224
FOLDABLE = {"convnext_t": False, "resnet50": True, "mobilenet_v2": True, "deit_tiny": False,
            "mobilenet_v1": True}

# --------------------------------------------------------------------- scorer families (composable)
# base flags WITHOUT prop_p / measured / iter-knobs — those are separate axes added in spec_flags().
BASES = OrderedDict([
    ("magnitude", ["--scorer", "magnitude"]),
    ("vbp",       ["--scorer", "variance"]),
    ("nci",       ["--scorer", "tp_variance"]),
    ("prop",      ["--scorer", "propagation"]),                  # relative independence (+ prop_p)
    ("cov",       ["--scorer", "propagation", "--prop_cov"]),    # covariance (+ prop_p [+ measured])
    ("iter",      ["--scorer", "propagation", "--prop_cov"]),    # iterative cov (+ prop_p [+ measured] + knobs)
])
BASELINE = ["magnitude", "vbp", "nci"]
FOCUS = ["prop", "cov", "iter"]
ITER_DROP_DEFAULT, ITER_FRAC_DEFAULT = 128, 0.6


def scorer_label(base, p, measured):
    if base in BASELINE:
        return base
    return f"{base}{'m' if measured else ''}p{p}"


# --------------------------------------------------------------------- spec generation
def make_spec(arch, base, p, measured, fold, frac, norm="width", nonrel=False,
              drop=ITER_DROP_DEFAULT, ifrac=ITER_FRAC_DEFAULT, block="curve"):
    mac_g = round(DENSE_MAC[arch] * frac, 3)
    tag = (f"{arch}__{scorer_label(base, p, measured)}__f{int(fold)}__n{norm}"
           f"__nr{int(nonrel)}__d{drop}__x{ifrac}__mac{frac:.3f}")
    return dict(arch=arch, base=base, p=p, measured=measured, fold=fold, frac=frac, mac_g=mac_g,
                normalizer=norm, nonrel=nonrel, iter_drop=drop, iter_frac=ifrac, block=block, tag=tag)


def make_bnfold_spec(arch, base, frac, protocol, recalib_k):
    """BN-fold validation cell: scorer fixed at computed p=2, BN handling = protocol(+k)."""
    mac_g = round(DENSE_MAC[arch] * frac, 3)
    sc = scorer_label(base, 2, False)
    tag = f"bnf__{arch}__{sc}__{protocol}__k{recalib_k}__mac{frac:.3f}"
    return dict(arch=arch, base=base, p=2, measured=False, fold=False, frac=frac, mac_g=mac_g,
                normalizer="width", nonrel=False, iter_drop=ITER_DROP_DEFAULT, iter_frac=ITER_FRAC_DEFAULT,
                block="bnfold", protocol=protocol, recalib_k=recalib_k, tag=tag)


def bnfold_specs():
    """~50 cluster runs to validate the BN-fold fix on the native-BN nets (resnet50, mnv2).
    Tests whether a no-grad measure-pass lifts the collapsed cells (resnet50 cov@65%≈0.02) toward
    the recalib ceiling, and whether fold_no_reinsert is dominated. Computed cov/iter only."""
    specs, seen = [], set()
    def add(arch, base, fr, proto, k):
        s = make_bnfold_spec(arch, base, fr, proto, k)
        if s["tag"] not in seen:
            seen.add(s["tag"]); specs.append(s)
    macs = [0.50, 0.65, 0.80]
    for arch in ("resnet50", "mobilenet_v2"):
        # cov: A(collapse) + D(fold+reinsert+measure50) at all MAC; B(native stale) at all MAC;
        #      C(native+measure) k-sweep {1,5,50}@0.65 + k50@{0.50,0.80}
        for fr in macs:
            add(arch, "cov", fr, "A_foldnoreinsert", 0)
            add(arch, "cov", fr, "D_foldreinsert_recal", 50)
            add(arch, "cov", fr, "B_native", 0)
        for k in (1, 5, 50):
            add(arch, "cov", 0.65, "C_native_recal", k)
        add(arch, "cov", 0.50, "C_native_recal", 50)
        add(arch, "cov", 0.80, "C_native_recal", 50)
        # iter: A + D(measure50) + C(native measure50) at all MAC
        for fr in macs:
            add(arch, "iter", fr, "A_foldnoreinsert", 0)
            add(arch, "iter", fr, "D_foldreinsert_recal", 50)
            add(arch, "iter", fr, "C_native_recal", 50)
        # magnitude control (bad scorer: BN fix should NOT rescue it) @0.65
        add(arch, "magnitude", 0.65, "A_foldnoreinsert", 0)
        add(arch, "magnitude", 0.65, "C_native_recal", 50)
    return specs


def bnfix_specs():
    """Champion (measure-pass) validation — BN-free deployable export across ALL scorers.
    Single MAC target (-33% ⇒ keep 0.67), 2 native-BN nets (resnet50, mnv2), all 6 scorers
    (magnitude, vbp, nci, prop, cov, iter). Per cell compare:
      A  fold-no-reinsert + no-recalib             = collapse reference (BN-free, stale)
      C  native + measure-pass(k50)                = champion (keeps BN)
      E  native + measure-pass(k50) + fold         = BN-FREE deploy (folds CALIBRATED BN)
    E≈C ⇒ folding the calibrated BN is loss-free. (var_comp F dropped — checkpoint-fragile, see
    FOLD_FIX.md; champion is measure-pass E.)"""
    specs, seen = [], set()
    def add(arch, base, fr, proto, k):
        s = make_bnfold_spec(arch, base, fr, proto, k)
        if s["tag"] not in seen:
            seen.add(s["tag"]); specs.append(s)
    fr = 0.67                                                # -33% MAC
    SCORERS = ["magnitude", "vbp", "nci", "prop", "cov", "iter"]
    for arch in ("resnet50", "mobilenet_v2"):
        for base in SCORERS:
            add(arch, base, fr, "A_foldnoreinsert", 0)       # collapse reference
            add(arch, base, fr, "C_native_recal", 50)        # champion (BN kept)
            add(arch, base, fr, "E_native_recal_fold", 50)   # BN-free deploy (measure-pass + fold)
    return specs


def bnrecal_specs():
    """Recalib on/off across ALL 6 scorers — the isolated measure-pass lift, native BN, no fold.
    Single MAC target (-33% ⇒ keep 0.67), all 5 archs × 6 scorers × {B,C} = 60 runs. Per cell:
      B  native + no-recalib  (stale BN — recalib OFF)
      C  native + measure-pass(k50)  (recalib ON)
    The 3 BN nets (resnet50, mnv2, mnv1) have a real recalib axis; convnext_t/deit_tiny are BN-free
    (LayerNorm) so the recalib pass is a no-op ⇒ B==C (the grid is filled for symmetry).
    mnv1 = timm-only, built by load_model via _unfuse_bn_act (see vbp_common). Cells carry
    block='bnfold' so bn_flags + summarize_bnfold render them (B vs C:native+m50)."""
    specs, seen = [], set()
    def add(arch, base, fr, proto, k):
        s = make_bnfold_spec(arch, base, fr, proto, k)
        if s["tag"] not in seen:
            seen.add(s["tag"]); specs.append(s)
    fr = 0.67                                                # -33% MAC
    SCORERS = ["magnitude", "vbp", "nci", "prop", "cov", "iter"]
    for arch in ("resnet50", "mobilenet_v2", "mobilenet_v1", "convnext_t", "deit_tiny"):
        for base in SCORERS:
            add(arch, base, fr, "B_native", 0)               # recalib OFF (stale)
            add(arch, base, fr, "C_native_recal", 50)        # recalib ON  (measure-pass k50)
    return specs


def gen_specs(args):
    archs = [a for a in (args.archs.split(",") if args.archs else ARCHS) if a in ARCHS]
    blocks = args.blocks.split(",")
    n = max(args.mac_points, 2)
    fracs = [round(0.30 + (0.90 - 0.30) * i / (n - 1), 3) for i in range(n)]
    specs, seen = [], set()

    def add(s):
        if s["tag"] not in seen:
            seen.add(s["tag"])
            specs.append(s)

    def scorer_variants(arch, fold, frac, **kw):
        """The 8 canonical variants: 3 baselines + prop + cov{comp,meas} + iter{comp,meas}, all p=2."""
        for b in BASELINE:
            add(make_spec(arch, b, 2, False, fold, frac, **kw))
        add(make_spec(arch, "prop", 2, False, fold, frac, **kw))
        for b in ("cov", "iter"):
            add(make_spec(arch, b, 2, False, fold, frac, **kw))
            add(make_spec(arch, b, 2, True, fold, frac, **kw))

    def focus_variants(arch, fold, frac, **kw):
        """Focus family only (prop + cov{comp,meas} + iter{comp,meas}), p=2 — for ablations."""
        add(make_spec(arch, "prop", 2, False, fold, frac, **kw))
        for b in ("cov", "iter"):
            add(make_spec(arch, b, 2, False, fold, frac, **kw))
            add(make_spec(arch, b, 2, True, fold, frac, **kw))

    for arch in archs:
        folds = [True, False] if FOLDABLE[arch] else [False]
        for fold in folds:
            if "curve" in blocks:                                    # headline retention-vs-MAC frontier
                for fr in fracs:
                    scorer_variants(arch, fold, fr, block="curve")
            if "ablation" in blocks:                                 # one-knob-at-a-time at ref_frac
                fr = args.ref_frac
                # p = 1 (vs the default p=2) — mass-conservation ablation
                add(make_spec(arch, "prop", 1, False, fold, fr, block="abl_p1"))
                for b in ("cov", "iter"):
                    add(make_spec(arch, b, 1, False, fold, fr, block="abl_p1"))
                    add(make_spec(arch, b, 1, True, fold, fr, block="abl_p1"))
                focus_variants(arch, fold, fr, norm="none", block="abl_norm")     # normalizer none vs width
                focus_variants(arch, fold, fr, nonrel=True, block="abl_nonrel")   # non-relative vs relative
                for meas in (False, True):                                        # iter hyperparams
                    for drop in (64, 256):
                        add(make_spec(arch, "iter", 2, meas, fold, fr, drop=drop, block="abl_iter"))
                    for ifr in (0.4, 0.8):
                        add(make_spec(arch, "iter", 2, meas, fold, fr, ifrac=ifr, block="abl_iter"))

    if "bnfold" in blocks:
        for s in bnfold_specs():
            if s["arch"] in archs and s["tag"] not in seen:
                seen.add(s["tag"]); specs.append(s)

    if "bnfix" in blocks:                                    # var_comp form (b): BN-free deploy @ -33%
        for s in bnfix_specs():
            if s["arch"] in archs and s["tag"] not in seen:
                seen.add(s["tag"]); specs.append(s)

    if "bnrecal" in blocks:                                  # recalib on/off × 6 scorers @ -33%
        for s in bnrecal_specs():
            if s["arch"] in archs and s["tag"] not in seen:
                seen.add(s["tag"]); specs.append(s)

    if args.max_runs and len(specs) > args.max_runs:
        specs = specs[:args.max_runs]
    return specs


# --------------------------------------------------------------------- flag assembly
def core_flags(args):
    """normnet_main flags common to every cell (BN/fold handling lives in bn_flags)."""
    f = [
        "--global_pruning", "--reparam_variant", "mean", "--bias_comp",
        "--calib_batches", str(args.calib_batches),
        "--epochs_train", "0", "--epochs_ft", "0", "--epochs_norm_ft", "0", "--skip_norm_eval",
    ]
    if getattr(args, "calib_split", "train") != "train":
        f += ["--calib_split", args.calib_split]            # val-calib (matches the harness mask)
    return f


def bn_flags(spec):
    """BN/fold/recalib handling. curve+ablation = pure pre-FT, no recalib. bnfold block = per
    protocol (the 50-run BN-fold validation): A fold-no-reinsert+no-recalib (collapse baseline),
    B native+no-recalib (stale guard), C native+measure-pass(k), D fold+reinsert+measure-pass(k).
    Measure-pass = no-grad BN re-estimation (k≈1 suffices ⇒ calibration, not FT)."""
    if spec.get("block") == "bnfold":
        proto, k = spec["protocol"], spec.get("recalib_k", 0)
        if proto == "A_foldnoreinsert":
            return ["--fold_native_bn", "--fold_no_reinsert", "--no_bn_recalib"]
        if proto == "B_native":
            return ["--no_bn_recalib"]
        if proto == "C_native_recal":
            return ["--recalib_batches", str(k)]
        if proto == "D_foldreinsert_recal":
            return ["--fold_native_bn", "--recalib_batches", str(k)]
        if proto == "E_native_recal_fold":                  # var_comp form (b): C + fold-after-recalib
            return ["--recalib_batches", str(k), "--fold_after_recalib"]
        if proto == "F_varcomp":                            # analytic var_comp: BN-free, ZERO forward
            return ["--fold_native_bn", "--fold_no_reinsert", "--var_comp", "--no_bn_recalib"]
        raise ValueError(f"unknown protocol {proto!r}")
    f = ["--no_bn_recalib"]                                  # default: pure pre-FT, recalib OFF
    if spec.get("fold"):
        f += ["--fold_native_bn", "--fold_no_reinsert"]
    return f


def spec_flags(spec):
    """The per-spec normnet_main flags (mac target, normalizer, scorer + all axes). Fold/BN in bn_flags."""
    arch, base = spec["arch"], spec["base"]
    a = ARCHS[arch]
    f = ["--mac_target_g", str(spec["mac_g"]), "--val_resize", str(a["val_resize"]),
         "--imp_normalizer", spec["normalizer"]]
    if arch in ("mobilenet_v2", "mobilenet_v1"):
        f += ["--max_prune_ratio", "0.8"]                  # per-layer cap (narrow mobilenet layers)
    f += list(BASES[base])
    if base in ("prop", "cov", "iter"):
        f += ["--prop_p", str(spec["p"])]
    if base == "iter":
        f += ["--prop_iterative", "--prop_iter_drop", str(spec["iter_drop"]),
              "--prop_iter_max_frac", str(spec["iter_frac"])]
    if spec["measured"] and base in ("cov", "iter"):
        f += ["--prop_measured_var"]
    if spec["nonrel"]:
        f += ["--prop_non_relative"]
    return f


# ===================================================================== CLUSTER: submit
def build_sh(spec, args, out_dir):
    a = ARCHS[spec["arch"]]
    flags = ["--model_type", a["model_type"], "--cnn_arch", a["cnn_arch"],
             "--model_name", a["cluster_weights"], "--data_path", CLUSTER["data_path"],
             "--save_dir", out_dir, "--save_tag", spec["tag"]] + core_flags(args) + bn_flags(spec) + spec_flags(spec)
    line = (f"python3 -m torch.distributed.launch --nproc_per_node={CLUSTER['ngpu']} "
            f"{CLUSTER['repo']}/benchmarks/vbp/normnet_main.py \\\n    " + " ".join(flags))
    return f"#!/bin/bash\nset -e\ncd {CLUSTER['repo']}\n{line}\n"


def submit_spec(spec, args):
    tag = spec["tag"]
    out_dir = os.path.join(CLUSTER["out_root"], args.run_name, tag)
    sh_path = os.path.join(out_dir, "run_ddp.sh")
    cmd = submit_cmd(sh_path, f"SWEEP {tag}")
    if args.dry_run:
        print(f"\n# ===== {tag} =====")
        print(build_sh(spec, args, out_dir))
        print("# submit:\n" + cmd)
        return
    if not args.force and os.path.exists(os.path.join(out_dir, f"{tag}_prune.json")):
        return "skip"
    os.makedirs(out_dir, exist_ok=True)
    try:
        os.chmod(out_dir, 0o777)
    except OSError:
        pass
    with open(sh_path, "w") as fp:
        fp.write(build_sh(spec, args, out_dir))
    os.chmod(sh_path, 0o777)
    subprocess.run(cmd, shell=True)
    return "submit"


def mode_submit(args):
    specs = gen_specs(args)
    if args.list:
        for s in specs:
            print(f"{s['block']:11s} {s['tag']}")
        print(f"\nTOTAL: {len(specs)} specs")
        return
    if not args.dry_run:
        os.makedirs(os.path.join(CLUSTER["out_root"], args.run_name), exist_ok=True)
    print(f"[SUBMIT] run_name={args.run_name}  {len(specs)} specs  "
          f"out={os.path.join(CLUSTER['out_root'], args.run_name)}", flush=True)
    n_sub = n_skip = 0
    for s in specs:
        r = submit_spec(s, args)
        if r == "submit":
            n_sub += 1
            print(f"[submit {n_sub}] {s['tag']}", flush=True)
        elif r == "skip":
            n_skip += 1
    if not args.dry_run:
        print(f"\nsubmitted {n_sub}, skipped {n_skip} (already done). Collect when the queue drains:\n  "
              f"python reproduce_table.py --mode collect --run_name {args.run_name}")


# ===================================================================== CLUSTER: collect
def collect(args):
    specs = gen_specs(args)
    rows, n_found = [], 0
    for s in specs:
        pj = os.path.join(CLUSTER["out_root"], args.run_name, s["tag"], f"{s['tag']}_prune.json")
        if not os.path.exists(pj):
            continue
        try:
            d = json.load(open(pj))
        except Exception:
            continue
        n_found += 1
        rows.append(dict(arch=s["arch"], scorer=scorer_label(s["base"], s["p"], s["measured"]),
                         base=s["base"], p=s["p"], measured=s["measured"], fold=s["fold"],
                         frac=s["frac"], block=s["block"], normalizer=s["normalizer"],
                         nonrel=s["nonrel"], iter_drop=s["iter_drop"], iter_frac=s["iter_frac"],
                         protocol=s.get("protocol"), recalib_k=s.get("recalib_k"),
                         acc=d.get("pre_ft_val_acc"), mac_g=d.get("macs_g"),
                         mac_pct=d.get("macs_pct"), tag=s["tag"]))
    with open(LEDGER, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[COLLECT] {n_found}/{len(specs)} cells found → {LEDGER}")
    summarize(rows)


SUMMARY_SCORERS = ["magnitude", "vbp", "propp2", "covp2", "covmp2", "iterp2", "itermp2", "nci"]


def summarize(rows):
    """Per-arch retention-vs-MAC table (ALL mac_frac points) + best-per-arch + BN-fold block."""
    if not rows:
        print("no results yet.")
        return
    canon = [r for r in rows if r["block"] == "curve" and r["normalizer"] == "width"
             and not r["nonrel"] and r["p"] == 2]
    for arch in ARCHS:
        ar = [r for r in canon if r["arch"] == arch and r["fold"] == FOLDABLE[arch]]
        if not ar:
            continue
        fracs = sorted({r["frac"] for r in ar})
        print(f"\n{'='*100}\n{arch}  (fold={int(FOLDABLE[arch])}, width, p2) — retention vs MAC "
              f"[{len(fracs)} points]\n{'='*100}")
        print("| mac_frac | mac% | " + " | ".join(SUMMARY_SCORERS) + " |")
        print("|" + "---|" * (len(SUMMARY_SCORERS) + 2))
        for fr in fracs:
            fa = [r for r in ar if r["frac"] == fr]
            mac = next((r["mac_pct"] for r in fa if r.get("mac_pct") is not None), "?")
            cells = []
            for sc in SUMMARY_SCORERS:
                m = [r for r in fa if r["scorer"] == sc]
                cells.append(f"{m[0]['acc']:.3f}" if m and m[0]["acc"] is not None else "·")
            print(f"| {fr:.3f} | {mac} | " + " | ".join(cells) + " |")
    print("\nBest scorer per arch (any axis):")
    for arch in ARCHS:
        ar = [r for r in rows if r["arch"] == arch and r["acc"] is not None]
        if ar:
            b = max(ar, key=lambda r: r["acc"])
            print(f"  {arch:14s} {b['acc']:.3f}  ({b['tag']})")
    summarize_bnfold(rows)
    print(f"\ncurves figure:  python plot_table.py --curves --out sweep.png")


def summarize_bnfold(rows):
    """BN-fold validation: for each arch×scorer×MAC, the protocol/measure-pass columns side by side."""
    br = [r for r in rows if r.get("block") == "bnfold" and r["acc"] is not None]
    if not br:
        return
    cols = [("A_foldnoreinsert", 0, "A:fold-noreinsert"), ("B_native", 0, "B:native-stale"),
            ("C_native_recal", 1, "C:native+m1"), ("C_native_recal", 5, "C:native+m5"),
            ("C_native_recal", 50, "C:native+m50"), ("D_foldreinsert_recal", 50, "D:fold+reins+m50"),
            ("E_native_recal_fold", 50, "E:native+m50+fold"), ("F_varcomp", 0, "F:varcomp-analytic")]
    print(f"\n{'='*100}\nBN-FOLD VALIDATION  (pre-FT top-1; measure-pass = no-grad BN re-estimation, "
          f"k batches)\n{'='*100}")
    print("| arch | scorer | mac% | " + " | ".join(c[2] for c in cols) + " |")
    print("|" + "---|" * (len(cols) + 3))
    keys = sorted({(r["arch"], r["scorer"], r["frac"]) for r in br})
    for arch, sc, fr in keys:
        grp = [r for r in br if r["arch"] == arch and r["scorer"] == sc and r["frac"] == fr]
        mac = next((r["mac_pct"] for r in grp if r.get("mac_pct") is not None), "?")
        cells = []
        for proto, k, _ in cols:
            m = [r for r in grp if r["protocol"] == proto and (r.get("recalib_k") or 0) == k]
            cells.append(f"{m[0]['acc']:.3f}" if m else "·")
        print(f"| {arch} | {sc} | {mac} | " + " | ".join(cells) + " |")


# ===================================================================== LOCAL: subprocess
def _subenv():
    env = dict(os.environ)
    extra = os.pathsep.join([REPO, HERE])
    env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


PRE_FT_RE = re.compile(r"pre-FT acc=([0-9.]+).*?([0-9.]+)G\s*\((\d+)%\)")


def mode_local(args):
    global LEDGER
    if args.smoke:
        if args.calib_batches == 50:
            args.calib_batches = 2
        if args.val_limit == 0:
            args.val_limit = 512
        LEDGER = os.path.join(HERE, "results_table_smoke.jsonl")
    specs = gen_specs(args)
    if not args.data_path and not args.dry_run:
        sys.exit("--data_path required for local mode")
    done = set()
    if os.path.exists(LEDGER):
        done = {json.loads(l)["tag"] for l in open(LEDGER) if l.strip()}
    print(f"[LOCAL] {len(specs)} specs, {len(done)} already in ledger", flush=True)
    for s in specs:
        if not args.force and s["tag"] in done:
            continue
        a = ARCHS[s["arch"]]
        model_name = a["model_name"]
        if a["weights"] and args.weights_dir:
            wp = os.path.join(os.path.expanduser(args.weights_dir), a["weights"])
            if os.path.exists(wp):
                model_name = wp
        with tempfile.TemporaryDirectory() as sd:
            argv = [sys.executable, MAIN, "--model_type", a["model_type"], "--cnn_arch", a["cnn_arch"],
                    "--model_name", model_name, "--data_path", args.data_path,
                    "--save_dir", sd, "--save_tag", s["tag"], "--disable_ddp",
                    "--val_batch_size", str(args.val_batch_size), "--val_limit", str(args.val_limit),
                    "--num_workers", str(args.num_workers)] + core_flags(args) + bn_flags(s) + spec_flags(s)
            if args.dry_run:
                print("  " + " ".join(argv[1:]))
                continue
            t0 = time.time()
            p = subprocess.run(argv, capture_output=True, text=True, env=_subenv())
            out = p.stdout + "\n" + p.stderr
            m = list(PRE_FT_RE.finditer(out))
            acc = float(m[-1].group(1)) if m else None
            rec = dict(arch=s["arch"], scorer=scorer_label(s["base"], s["p"], s["measured"]),
                       fold=s["fold"], frac=s["frac"], block=s["block"], acc=acc,
                       mac_pct=int(m[-1].group(3)) if m else None, tag=s["tag"],
                       secs=round(time.time() - t0))
            print(f"  {s['tag']}: acc={acc}  ({rec['secs']}s)", flush=True)
            with open(LEDGER, "a") as f:
                f.write(json.dumps(rec) + "\n")


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["submit", "collect", "local"], default="submit")
    ap.add_argument("--archs", default="", help="comma subset of " + ",".join(ARCHS))
    ap.add_argument("--blocks", default="curve,ablation",
                    help="curve, ablation, bnfold (~50-run BN-fold validation), bnfix (BN-free deploy "
                         "@ -33%), bnrecal (recalib on/off × 6 scorers @ -33%, r50+mnv2)")
    ap.add_argument("--mac_points", type=int, default=18, help="number of mac_frac points (curve density)")
    ap.add_argument("--ref_frac", type=float, default=0.65, help="mac_frac for the ablation block")
    ap.add_argument("--max_runs", type=int, default=0, help="cap total specs (0=all)")
    ap.add_argument("--calib_batches", type=int, default=50)
    ap.add_argument("--calib_split", default="train", choices=["train", "val"],
                    help="calib split for reparam σ/μ + covariance. 'val' matches the local research "
                         "harness mask (var_comp is sensitive to the cov source).")
    # cluster
    ap.add_argument("--run_name", default="SWEEP_v1")
    ap.add_argument("--force", action="store_true", help="re-submit / re-run even if result exists")
    ap.add_argument("--dry_run", action="store_true", help="print .sh/commands, run nothing")
    ap.add_argument("--list", action="store_true", help="print specs + total count, submit nothing")
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
        collect(args)
    else:
        mode_local(args)


if __name__ == "__main__":
    main()
