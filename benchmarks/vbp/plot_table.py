"""Plot the reproduce_table ledger as a grouped bar chart (arch × scorer pre-FT top-1).

Reads results_table.jsonl (or --ledger), draws one grouped bar chart so a smoke/local run can be
visually compared against the cluster run side by side.

  python plot_table.py                                  # results_table.jsonl  -> retention_table.png
  python plot_table.py --ledger results_table_smoke.jsonl --out smoke.png --title "LOCAL smoke"
"""
import argparse
import json
import os
from collections import OrderedDict, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCORER_ORDER = ["magnitude", "vbp", "prop", "cov_comp", "cov_meas",
                "iter_comp", "iter_meas", "nci"]


def load(ledger):
    rows = defaultdict(dict)          # arch -> scorer -> acc
    macs = {}
    if not os.path.exists(ledger):
        raise SystemExit(f"no ledger: {ledger}")
    for line in open(ledger):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("acc") is not None:
            rows[d["arch"]][d["scorer"]] = d["acc"]
            if d.get("mac_pct"):
                macs[d["arch"]] = d["mac_pct"]
    return rows, macs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="results_table.jsonl")
    ap.add_argument("--out", default="retention_table.png")
    ap.add_argument("--title", default="retention (pre-FT top-1, width norm)")
    args = ap.parse_args()

    rows, macs = load(args.ledger)
    archs = list(rows.keys())
    scorers = [s for s in SCORER_ORDER if any(s in rows[a] for a in archs)]

    x = np.arange(len(archs))
    w = 0.8 / max(len(scorers), 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(archs) + 3, 5))
    cmap = plt.get_cmap("tab10")
    for i, sc in enumerate(scorers):
        vals = [rows[a].get(sc, 0.0) for a in archs]
        bars = ax.bar(x + i * w, vals, w, label=sc, color=cmap(i % 10))
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([f"{a}\n({macs.get(a,'?')}% MAC)" for a in archs])
    ax.set_ylabel("pre-FT top-1")
    ax.set_title(args.title)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("wrote", args.out)

    # also dump the markdown table for quick eyeball
    print("\n| arch | " + " | ".join(scorers) + " |")
    print("|" + "---|" * (len(scorers) + 1))
    for a in archs:
        cells = [f"{rows[a].get(s, float('nan')):.3f}" if s in rows[a] else "·" for s in scorers]
        print(f"| {a} ({macs.get(a,'?')}%) | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
