"""
Summarize normalize_net.py runs into a compact, analyzable table.

Reads the machine artifacts each run writes (<tag>_run.json + <tag>_metrics.jsonl)
and prints (1) a one-row-per-run summary and (2) a per-epoch val_acc matrix.
Accepts run dirs, parent dirs (recurses one level), or explicit *_run.json paths.

Usage:
    python benchmarks/vbp/parse_normnet.py /path/to/NORMNET/ResNet50
    python benchmarks/vbp/parse_normnet.py v1_dir v2_dir v3_dir
    python benchmarks/vbp/parse_normnet.py --pct out/*/  # accuracies as %
"""
import argparse
import glob
import json
import os
import sys


def _find_run_jsons(paths):
    found = []
    for p in paths:
        if os.path.isfile(p) and p.endswith("_run.json"):
            found.append(p)
        elif os.path.isdir(p):
            hits = glob.glob(os.path.join(p, "*_run.json"))
            hits += glob.glob(os.path.join(p, "*", "*_run.json"))  # one level down
            found.extend(hits)
        else:  # treat as glob
            found.extend(g for g in glob.glob(p) if g.endswith("_run.json"))
    # de-dup, stable order
    seen, out = set(), []
    for f in sorted(found):
        rp = os.path.realpath(f)
        if rp not in seen:
            seen.add(rp); out.append(f)
    return out


def _sched_label(cfg):
    """One-token description of the FT schedule from the config."""
    lr = cfg.get("lr"); eta = cfg.get("ft_eta_min"); wu = cfg.get("ft_warmup_epochs", 0) or 0
    if lr is not None and eta is not None and abs(eta - lr) / max(lr, 1e-12) < 1e-6:
        return f"flat@{lr:.0e}"
    base = f"cosine->{eta:.0e}" if eta is not None else "cosine"
    return (f"wu{int(wu)}+" + base) if wu else base


def _load(run_json):
    with open(run_json) as f:
        run = json.load(f)
    tag = os.path.basename(run_json)[:-len("_run.json")]
    metrics_path = os.path.join(os.path.dirname(run_json), f"{tag}_metrics.jsonl")
    epochs = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("val_acc") is not None:
                    epochs[r["epoch"]] = r["val_acc"]
    return tag, run, epochs


def main(argv):
    ap = argparse.ArgumentParser(description="Summarize normalize_net runs")
    ap.add_argument("paths", nargs="+", help="run dirs / parent dirs / *_run.json")
    ap.add_argument("--setups", default=None,
                    help="Comma-separated whitelist: only include runs whose run-dir name OR "
                         "tag is listed (e.g. --setups v1,v2,v3). Mirrors parse_logs.py --setups.")
    ap.add_argument("--pct", action="store_true", help="show accuracies as percent")
    args = ap.parse_args(argv[1:])

    run_jsons = _find_run_jsons(args.paths)
    if args.setups:
        sel = {s.strip() for s in args.setups.split(",") if s.strip()}
        run_jsons = [rj for rj in run_jsons
                     if os.path.basename(os.path.dirname(rj)) in sel
                     or os.path.basename(rj)[:-len("_run.json")] in sel]
    if not run_jsons:
        print("No *_run.json found under:", args.paths,
              ("(setups=" + args.setups + ")") if args.setups else ""); return 1

    runs = [_load(rj) for rj in run_jsons]
    scale = 100.0 if args.pct else 1.0
    fmt = "{:.2f}" if args.pct else "{:.4f}"

    def f(x):
        return fmt.format(x * scale) if isinstance(x, (int, float)) else "  -  "

    # ---- Summary table ----
    hdr = ["tag", "arm", "opt", "lr", "sched", "kd", "pre", "init", "best", "Δinit", "Δpre", "st"]
    rows = []
    for tag, run, epochs in runs:
        cfg = run.get("config", {})
        pre = run.get("pre_train_val_acc")
        init = epochs.get(0)
        best = run.get("best_val_acc")
        if best is None and epochs:
            best = max(v for k, v in epochs.items() if k >= 1) if any(k >= 1 for k in epochs) else None
        d_init = (best - init) if (best is not None and init is not None) else None
        d_pre = (best - pre) if (best is not None and pre is not None) else None
        kd = "y" if cfg.get("use_kd") else "n"
        lr = cfg.get("lr")
        rows.append([
            tag, run.get("arm", "?"), cfg.get("opt", "?"),
            f"{lr:.1e}" if lr is not None else "-", _sched_label(cfg), kd,
            f(pre), f(init), f(best),
            ("+" + f(d_init)) if (d_init is not None and d_init >= 0) else f(d_init),
            ("+" + f(d_pre)) if (d_pre is not None and d_pre >= 0) else f(d_pre),
            run.get("status", "?")[:4],
        ])

    widths = [max(len(str(r[i])) for r in ([hdr] + rows)) for i in range(len(hdr))]
    def line(cells):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))
    print("=== RUN SUMMARY" + (" (%)" if args.pct else "") + " ===")
    print(line(hdr))
    for r in rows:
        print(line(r))

    # ---- Per-epoch val_acc matrix (e0 = post-reparam init) ----
    max_e = max((max(e) for _, _, e in runs if e), default=0)
    print("\n=== VAL_ACC PER EPOCH (e0=init) ===")
    cols = ["tag"] + [f"e{i}" for i in range(0, max_e + 1)]
    mat = []
    for tag, _, epochs in runs:
        mat.append([tag] + [f(epochs.get(i)) for i in range(0, max_e + 1)])
    w = [max(len(str(m[i])) for m in ([cols] + mat)) for i in range(len(cols))]
    print("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
    for m in mat:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(m)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
