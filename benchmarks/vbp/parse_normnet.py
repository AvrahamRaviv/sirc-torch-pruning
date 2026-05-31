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
    epochs = {}     # epoch → val_acc (back-compat convenience)
    records = {}    # epoch → full record dict (loss, reg, sparsity, …)
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                ep = r.get("epoch")
                if ep is not None:
                    records[ep] = r
                if r.get("val_acc") is not None:
                    epochs[ep] = r["val_acc"]
    return tag, run, epochs, records


def _last_real_epoch(records):
    """Return the record of the highest epoch ≥1 (skips e0 post-reparam init)."""
    keys = [k for k in records if k >= 1]
    return records[max(keys)] if keys else None


def main(argv):
    ap = argparse.ArgumentParser(description="Summarize normalize_net runs")
    ap.add_argument("paths", nargs="+", help="run dirs / parent dirs / *_run.json")
    ap.add_argument("--setups", default=None,
                    help="Comma-separated whitelist: only include runs whose run-dir name OR "
                         "tag is listed (e.g. --setups v1,v2,v3). Mirrors parse_logs.py --setups.")
    ap.add_argument("--pct", action="store_true", help="show accuracies as percent")
    ap.add_argument("--show", default="acc",
                    help="comma list of per-epoch matrices to print: "
                         "acc,reg,loss,spars (default: acc). 'all' = every matrix.")
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

    def g(x, nd=4):  # generic float formatter (no acc-scaling); "-" for missing
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "  -  "

    # ---- Summary table ----
    # λ / ema = reparam regularization knobs; reg/ce = final-epoch loss split;
    # <.1 = fraction of channels with ‖σ·v‖<0.1 (induced sparsity, the E4 signal).
    hdr = ["tag", "arm", "lr", "sched", "kd", "λ", "ema", "pre", "best", "Δpre",
           "trn", "reg", "ce", "<.1", "vmin", "st"]
    rows = []
    for tag, run, epochs, records in runs:
        cfg = run.get("config", {})
        pre = run.get("pre_train_val_acc")
        best = run.get("best_val_acc")
        if best is None and epochs:
            best = max(v for k, v in epochs.items() if k >= 1) if any(k >= 1 for k in epochs) else None
        d_pre = (best - pre) if (best is not None and pre is not None) else None
        last = _last_real_epoch(records) or {}
        lam = cfg.get("reparam_lambda")
        ema = cfg.get("mu_ema_momentum")
        lr = cfg.get("lr")
        rows.append([
            tag, run.get("arm", "?"),
            f"{lr:.1e}" if lr is not None else "-", _sched_label(cfg),
            "y" if cfg.get("use_kd") else "n",
            f"{lam:g}" if lam is not None else "-",
            f"{ema:g}" if ema is not None else "-",
            f(pre), f(best),
            ("+" + f(d_pre)) if (d_pre is not None and d_pre >= 0) else f(d_pre),
            g(last.get("train_loss"), 3), g(last.get("reg_loss"), 4),
            g(last.get("ce_kd_loss"), 3),
            g(last.get("frac_below_0.1"), 3), g(last.get("vnorm_min"), 4),
            run.get("status", "?")[:4],
        ])

    widths = [max(len(str(r[i])) for r in ([hdr] + rows)) for i in range(len(hdr))]
    def line(cells):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))
    print("=== RUN SUMMARY" + (" (%)" if args.pct else "") + " ===")
    print(line(hdr))
    for r in rows:
        print(line(r))

    # ---- Per-epoch matrices (selectable via --show) ----
    # spec: key → (title, record-field, formatter, start-epoch)
    matrices = {
        "acc":   ("VAL_ACC PER EPOCH (e0=init)", "val_acc", f, 0),
        "reg":   ("REG_LOSS (λ‖σ·v‖) PER EPOCH", "reg_loss", lambda x: g(x, 4), 1),
        "loss":  ("TRAIN_LOSS PER EPOCH", "train_loss", lambda x: g(x, 3), 1),
        "spars": ("FRAC ‖σ·v‖<0.1 PER EPOCH", "frac_below_0.1", lambda x: g(x, 3), 1),
    }
    want = list(matrices.keys()) if args.show.strip() == "all" \
        else [s.strip() for s in args.show.split(",") if s.strip() in matrices]
    max_e = max((max(r) for _, _, _, r in runs if r), default=0)
    for key in want:
        title, field, ffmt, e0 = matrices[key]
        # val_acc lives in the epochs dict; others in the full records.
        def cell(records, epochs, i, field=field, ffmt=ffmt):
            if field == "val_acc":
                return ffmt(epochs.get(i))
            rec = records.get(i)
            return ffmt(rec.get(field)) if rec else "  -  "
        print(f"\n=== {title} ===")
        cols = ["tag"] + [f"e{i}" for i in range(e0, max_e + 1)]
        mat = [[tag] + [cell(records, epochs, i) for i in range(e0, max_e + 1)]
               for tag, _, epochs, records in runs]
        w = [max(len(str(m[i])) for m in ([cols] + mat)) for i in range(len(cols))]
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
        for m in mat:
            print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(m)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
