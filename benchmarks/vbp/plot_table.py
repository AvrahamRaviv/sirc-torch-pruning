"""Plot the reproduce_table sweep ledger.

  --curves  : per-arch retention-vs-MAC frontier (one line per scorer; canonical axes: curve block,
              width norm, p=2, relative, fold=arch-default). The headline figure for the 1000-run sweep.
  (default) : grouped bar chart at the reference fraction (back-compat with the small 4×8 table).

  python plot_table.py --curves --out sweep.png
  python plot_table.py --ledger results_table_smoke.jsonl --out smoke.png --title "LOCAL smoke"
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCORER_ORDER = ["magnitude", "vbp", "nci", "prop", "covp2", "covmp2", "iterp2", "itermp2",
                "cov_comp", "cov_meas", "iter_comp", "iter_meas"]   # last 4 = old 4×8 tag names
FOLDABLE = {"convnext_t": False, "resnet50": True, "mobilenet_v2": True, "deit_tiny": False}


def load(ledger):
    if not os.path.exists(ledger):
        raise SystemExit(f"no ledger: {ledger}")
    rows = [json.loads(l) for l in open(ledger) if l.strip()]
    return [r for r in rows if r.get("acc") is not None]


def canonical(rows):
    """curve block, width norm, p=2, relative, fold=arch-default — the comparable frontier set."""
    out = []
    for r in rows:
        if r.get("block", "curve") != "curve":
            continue
        if r.get("normalizer", "width") != "width" or r.get("nonrel", False) or r.get("p", 2) != 2:
            continue
        if r.get("fold") is not None and r["fold"] != FOLDABLE.get(r["arch"], False):
            continue
        out.append(r)
    return out


def plot_curves(rows, out, title):
    rows = canonical(rows)
    archs = sorted({r["arch"] for r in rows})
    if not archs:
        raise SystemExit("no curve-block rows (run the sweep first, or use the bar mode)")
    ncol = min(2, len(archs))
    nrow = (len(archs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4.5 * nrow), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ai, arch in enumerate(archs):
        ax = axes[ai // ncol][ai % ncol]
        ar = [r for r in rows if r["arch"] == arch]
        scorers = [s for s in SCORER_ORDER if any(r["scorer"] == s for r in ar)]
        for i, sc in enumerate(scorers):
            pts = sorted([(r["mac_pct"], r["acc"]) for r in ar if r["scorer"] == sc])
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "-o", ms=3, color=cmap(i % 10), label=sc)
        ax.set_title(arch)
        ax.set_xlabel("MAC %")
        ax.set_ylabel("pre-FT top-1")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    for j in range(len(archs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


def plot_bars(rows, out, title):
    by = defaultdict(dict)
    macs = {}
    for r in rows:
        by[r["arch"]][r["scorer"]] = r["acc"]
        if r.get("mac_pct"):
            macs[r["arch"]] = r["mac_pct"]
    archs = list(by.keys())
    scorers = [s for s in SCORER_ORDER if any(s in by[a] for a in archs)]
    x = np.arange(len(archs))
    w = 0.8 / max(len(scorers), 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(archs) + 4, 5))
    cmap = plt.get_cmap("tab10")
    for i, sc in enumerate(scorers):
        vals = [by[a].get(sc, 0.0) for a in archs]
        ax.bar(x + i * w, vals, w, label=sc, color=cmap(i % 10))
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([f"{a}\n({macs.get(a,'?')}%)" for a in archs])
    ax.set_ylabel("pre-FT top-1")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="results_table.jsonl")
    ap.add_argument("--out", default="sweep.png")
    ap.add_argument("--title", default="retention vs MAC (pre-FT, no recalib)")
    ap.add_argument("--curves", action="store_true", help="per-arch retention-vs-MAC frontier")
    args = ap.parse_args()
    rows = load(args.ledger)
    if args.curves:
        plot_curves(rows, args.out, args.title)
    else:
        plot_bars(rows, args.out, args.title)


if __name__ == "__main__":
    main()
