"""Plot VBP experiment results for DeiT-T.

Scans subfolders for vbp_imagenet.log files, parses retention/best accuracy
and params, then plots comparison against paper Table 10.

Expected folder structure:
    <root_dir>/
        <setup_name>/          # e.g., global_kd, global_no_kd
            kr_0.95/
                vbp_imagenet.log
            kr_0.90/
                vbp_imagenet.log
            ...

Usage:
    python benchmarks/vbp/plot_results.py --root /algo/NetOptimization/outputs/VBP/DeiT_tiny
    python benchmarks/vbp/plot_results.py --root /algo/.../DeiT_tiny --save plot.png
"""

import argparse
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Paper Table 10, DeiT-T (arxiv 2507.12988)
PAPER_DEIT_T = {
    # prune_pct: (retention_acc, macs_G, params_M)
    0:  (72.02, 1.26, 5.72),
    5:  (71.67, 1.22, 5.54),
    10: (70.95, 1.19, 5.36),
    15: (70.05, 1.15, 5.18),
    20: (68.87, 1.12, 5.01),
    25: (67.37, 1.08, 4.83),
    30: (64.76, 1.05, 4.65),
    35: (61.12, 1.01, 4.48),
    40: (55.64, 0.98, 4.30),
    45: (49.77, 0.94, 4.12),
    50: (39.58, 0.91, 3.94),
}


def parse_log(log_path):
    """Parse a vbp_imagenet.log file for key metrics.

    Returns dict with keys: keep_ratio, retention_acc, best_acc, final_acc,
    pruned_params_M, pruned_macs_G, base_params_M, original_acc.
    Returns None if log is incomplete.
    """
    result = {}

    with open(log_path, "r") as f:
        text = f.read()

    # keep_ratio from pruning line
    m = re.search(r"keep_ratio=([\d.]+)", text)
    if m:
        result["keep_ratio"] = float(m.group(1))

    # Original accuracy
    m = re.search(r"Original Acc(?:uracy)?:\s*([\d.]+)", text)
    if m:
        result["original_acc"] = float(m.group(1))
        # If value < 1, it's a fraction (0.7202), convert to %
        if result["original_acc"] < 1.0:
            result["original_acc"] *= 100

    # Retention accuracy
    m = re.search(r"Retention Acc(?:uracy)?:\s*([\d.]+)", text)
    if m:
        result["retention_acc"] = float(m.group(1))
        if result["retention_acc"] < 1.0:
            result["retention_acc"] *= 100

    # Best accuracy (after fine-tuning)
    m = re.search(r"Best Acc(?:uracy)?:\s*([\d.]+)", text)
    if m:
        result["best_acc"] = float(m.group(1))
        if result["best_acc"] < 1.0:
            result["best_acc"] *= 100

    # Final accuracy
    m = re.search(r"Final Acc(?:uracy)?:\s*([\d.]+)", text)
    if m:
        result["final_acc"] = float(m.group(1))
        if result["final_acc"] < 1.0:
            result["final_acc"] *= 100

    # Pruned params from summary line: "Base Params:  5.72M -> Pruned: 5.18M (90.6%)"
    m = re.search(r"Base Params:\s*([\d.]+)M\s*->\s*Pruned:\s*([\d.]+)M", text)
    if m:
        result["base_params_M"] = float(m.group(1))
        result["pruned_params_M"] = float(m.group(2))

    # Pruned MACs
    m = re.search(r"Base MACs:\s*([\d.]+)G\s*->\s*Pruned:\s*([\d.]+)G", text)
    if m:
        result["base_macs_G"] = float(m.group(1))
        result["pruned_macs_G"] = float(m.group(2))

    # Fallback: parse from earlier "Pruned: X.XXG MACs, X.XXM params" line
    if "pruned_params_M" not in result:
        m = re.search(r"Pruned:\s*([\d.]+)G MACs,\s*([\d.]+)M params", text)
        if m:
            result["pruned_macs_G"] = float(m.group(1))
            result["pruned_params_M"] = float(m.group(2))

    # Compute prune_pct from keep_ratio
    if "keep_ratio" in result:
        result["prune_pct"] = round((1 - result["keep_ratio"]) * 100)

    # Also try to infer from folder name if not in log
    if "keep_ratio" not in result:
        folder = os.path.basename(os.path.dirname(log_path))
        m = re.match(r"kr_([\d.]+)", folder)
        if m:
            result["keep_ratio"] = float(m.group(1))
            result["prune_pct"] = round((1 - result["keep_ratio"]) * 100)

    # Minimum required fields
    if "pruned_params_M" not in result or "keep_ratio" not in result:
        return None

    return result


def scan_root(root_dir):
    """Scan root_dir for experiment results.

    Returns: {setup_name: [result_dict, ...]} sorted by prune_pct.
    """
    setups = defaultdict(list)

    for setup_name in sorted(os.listdir(root_dir)):
        setup_dir = os.path.join(root_dir, setup_name)
        if not os.path.isdir(setup_dir):
            continue

        for kr_folder in sorted(os.listdir(setup_dir)):
            kr_dir = os.path.join(setup_dir, kr_folder)
            if not os.path.isdir(kr_dir):
                continue

            log_path = os.path.join(kr_dir, "vbp_imagenet.log")
            if not os.path.exists(log_path):
                continue

            result = parse_log(log_path)
            if result is None:
                print(f"  [SKIP] Incomplete log: {log_path}")
                continue

            result["setup"] = setup_name
            setups[setup_name].append(result)

    # Sort each setup by prune_pct
    for setup_name in setups:
        setups[setup_name].sort(key=lambda r: r.get("prune_pct", 0))

    return dict(setups)


def print_table(setups):
    """Print a comparison table for all setups."""
    print("\n" + "=" * 100)
    print("VBP DeiT-T Results Summary")
    print("=" * 100)

    for setup_name, results in setups.items():
        print(f"\n--- {setup_name} ---")
        header = (f"{'Prune%':>7} | {'Keep':>5} | {'Ret%':>7} | {'Best%':>7} | "
                  f"{'Paper Ret%':>10} | {'Δ Ret':>6} | "
                  f"{'Params(M)':>9} | {'Paper':>6}")
        print(header)
        print("-" * len(header))

        for r in results:
            pp = r.get("prune_pct", "?")
            kr = r.get("keep_ratio", "?")
            ret = r.get("retention_acc", None)
            best = r.get("best_acc", None)
            params = r.get("pruned_params_M", None)

            paper = PAPER_DEIT_T.get(pp)
            paper_ret = paper[0] if paper else None

            ret_str = f"{ret:.2f}" if ret is not None else "  —"
            best_str = f"{best:.2f}" if best is not None else "  —"
            paper_str = f"{paper_ret:.2f}" if paper_ret is not None else "  —"
            delta = f"{ret - paper_ret:+.2f}" if (ret is not None and paper_ret is not None) else "  —"
            params_str = f"{params:.2f}" if params is not None else "  —"
            paper_params = f"{paper[2]:.2f}" if paper else "  —"

            print(f"  {pp:>5}% | {kr:>.2f} | {ret_str:>6}% | {best_str:>6}% | "
                  f"{paper_str:>9}% | {delta:>6} | "
                  f"{params_str:>8}M | {paper_params:>5}M")

    print("=" * 100)


def plot_results(setups, save_path):
    """Plot retention and best accuracy vs params, compared to paper."""
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Paper reference (both plots) ---
    paper_params = [PAPER_DEIT_T[p][2] for p in sorted(PAPER_DEIT_T.keys()) if p > 0]
    paper_ret = [PAPER_DEIT_T[p][0] for p in sorted(PAPER_DEIT_T.keys()) if p > 0]

    for ax in axes:
        ax.plot(paper_params, paper_ret, "s--", color="gray", linewidth=2,
                markersize=7, label="Paper Table 10 (retention)", zorder=2, alpha=0.7)
        ax.axhline(y=PAPER_DEIT_T[0][0], color="lightgray", linestyle=":",
                    linewidth=1, label=f"Baseline ({PAPER_DEIT_T[0][0]:.1f}%)")

    # --- Plot 1: Retention accuracy vs Params ---
    ax = axes[0]
    for i, (setup_name, results) in enumerate(setups.items()):
        params = [r["pruned_params_M"] for r in results if "retention_acc" in r]
        rets = [r["retention_acc"] for r in results if "retention_acc" in r]
        if params:
            ax.plot(params, rets, "o-", color=colors[i % len(colors)],
                    linewidth=2, markersize=6, label=setup_name, zorder=3)

    ax.set_xlabel("Parameters (M)", fontsize=13)
    ax.set_ylabel("Retention Accuracy (%)", fontsize=13)
    ax.set_title("Retention Accuracy vs Parameters", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # --- Plot 2: Best accuracy (after FT) vs Params ---
    ax = axes[1]
    for i, (setup_name, results) in enumerate(setups.items()):
        params = [r["pruned_params_M"] for r in results if "best_acc" in r]
        bests = [r["best_acc"] for r in results if "best_acc" in r]
        if params:
            ax.plot(params, bests, "o-", color=colors[i % len(colors)],
                    linewidth=2, markersize=6, label=setup_name, zorder=3)

    ax.set_xlabel("Parameters (M)", fontsize=13)
    ax.set_ylabel("Best Accuracy after FT (%)", fontsize=13)
    ax.set_title("Best Accuracy (post Fine-Tuning) vs Parameters", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    fig.suptitle("VBP — DeiT-T", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot VBP experiment results for DeiT-T",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True,
                        help="Root dir containing setup subfolders (e.g., .../DeiT_tiny)")
    parser.add_argument("--save", default=None,
                        help="Output plot path (default: <root>/vbp_results.png)")
    args = parser.parse_args()

    if args.save is None:
        args.save = os.path.join(args.root, "vbp_results.png")

    setups = scan_root(args.root)

    if not setups:
        print(f"No results found under {args.root}")
        print(f"Expected structure: {args.root}/<setup_name>/kr_<ratio>/vbp_imagenet.log")
        return

    total_runs = sum(len(v) for v in setups.values())
    print(f"Found {len(setups)} setup(s), {total_runs} total runs")

    print_table(setups)
    plot_results(setups, args.save)


if __name__ == "__main__":
    main()
