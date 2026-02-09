"""Plot VBP experiment results for DeiT-T.

Scans subfolders for vbp_imagenet.log files, parses retention/best accuracy
and params, then plots comparison against paper Table 10.

Handles partial/in-progress experiments gracefully:
- Runs with only retention accuracy (no fine-tuning yet) are included
- Missing prune rates are shown as gaps in the table
- Plots use whatever data is available

Expected folder structure:
    <root_dir>/
        <setup_name>/          # e.g., global_kd, global_no_kd
            kr_0.95/
                vbp_imagenet.log
            kr_0.90/
                vbp_imagenet.log
            ...

Usage:
    cd /algo/NetOptimization/outputs/VBP/DeiT_tiny && python plot_results.py
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
    # prune_pct: (retention_acc, finetuned_acc, macs_G, params_M)
    0:  (72.02, 72.02, 1.26, 5.72),
    5:  (71.67, 72.05, 1.22, 5.54),
    10: (70.95, 71.92, 1.19, 5.36),
    15: (70.05, 71.76, 1.15, 5.18),
    20: (68.87, 71.60, 1.12, 5.01),
    25: (67.37, 71.44, 1.08, 4.83),
    30: (64.76, 71.20, 1.05, 4.65),
    35: (61.12, 70.86, 1.01, 4.48),
    40: (55.64, 70.55, 0.98, 4.30),
    45: (49.77, 70.08, 0.94, 4.12),
    50: (39.58, 69.70, 0.91, 3.94),
}


def parse_log(log_path):
    """Parse a vbp_imagenet.log file for key metrics.

    Returns dict with whatever fields are available. Only requires keep_ratio
    (from log or folder name) to be considered valid. Missing fields are simply
    absent from the dict.
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

    # Early pruned stats: "Pruned: 1.04G MACs, 5.54M params" (appears right after retention)
    m = re.search(r"Pruned:\s*([\d.]+)G MACs,\s*([\d.]+)M params", text)
    if m:
        result["pruned_macs_G"] = float(m.group(1))
        result["pruned_params_M"] = float(m.group(2))

    # Summary line (overrides with same values, but also captures base stats)
    m = re.search(r"Base Params:\s*([\d.]+)M\s*->\s*Pruned:\s*([\d.]+)M", text)
    if m:
        result["base_params_M"] = float(m.group(1))
        result["pruned_params_M"] = float(m.group(2))

    m = re.search(r"Base MACs:\s*([\d.]+)G\s*->\s*Pruned:\s*([\d.]+)G", text)
    if m:
        result["base_macs_G"] = float(m.group(1))
        result["pruned_macs_G"] = float(m.group(2))

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

    # If we still don't have params, estimate from paper table by prune_pct
    if "pruned_params_M" not in result and "prune_pct" in result:
        paper = PAPER_DEIT_T.get(result["prune_pct"])
        if paper:
            result["pruned_params_M"] = paper[3]
            result["params_estimated"] = True

    # Only need keep_ratio to be valid
    if "keep_ratio" not in result:
        return None

    # Determine status
    if "best_acc" in result or "final_acc" in result:
        result["status"] = "done"
    elif "retention_acc" in result:
        result["status"] = "retention_only"
    else:
        result["status"] = "pruned_only"

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
                print(f"  [SKIP] Cannot parse: {log_path}")
                continue

            result["setup"] = setup_name
            result["log_path"] = log_path
            setups[setup_name].append(result)

    # Sort each setup by prune_pct
    for setup_name in setups:
        setups[setup_name].sort(key=lambda r: r.get("prune_pct", 0))

    return dict(setups)


def print_table(setups):
    """Print a comparison table for all setups."""
    print("\n" + "=" * 110)
    print("VBP DeiT-T Results Summary")
    print("=" * 110)

    for setup_name, results in setups.items():
        n_done = sum(1 for r in results if r.get("status") == "done")
        n_ret = sum(1 for r in results if r.get("status") == "retention_only")
        n_total = len(results)
        status_str = f"{n_done} done, {n_ret} retention-only" if n_ret > 0 else f"{n_done} done"
        print(f"\n--- {setup_name} ({n_total} runs: {status_str}) ---")
        header = (f"{'Prune%':>7} | {'Keep':>5} | {'Ret%':>7} | {'Best%':>7} | "
                  f"{'Paper Ret%':>10} | {'Paper FT%':>9} | {'Δ Ret':>6} | "
                  f"{'Params(M)':>9} | {'Status':>8}")
        print(header)
        print("-" * len(header))

        for r in results:
            pp = r.get("prune_pct", "?")
            kr = r.get("keep_ratio", "?")
            ret = r.get("retention_acc", None)
            best = r.get("best_acc", None)
            params = r.get("pruned_params_M", None)
            status = r.get("status", "?")

            paper = PAPER_DEIT_T.get(pp)
            paper_ret = paper[0] if paper else None
            paper_ft = paper[1] if paper else None

            ret_str = f"{ret:.2f}" if ret is not None else "  —"
            best_str = f"{best:.2f}" if best is not None else "  —"
            paper_ret_str = f"{paper_ret:.2f}" if paper_ret is not None else "  —"
            paper_ft_str = f"{paper_ft:.2f}" if paper_ft is not None else "  —"
            delta = f"{ret - paper_ret:+.2f}" if (ret is not None and paper_ret is not None) else "  —"
            params_str = f"{params:.2f}" if params is not None else "  —"
            if r.get("params_estimated"):
                params_str += "*"
            status_icon = {"done": "OK", "retention_only": "FT...", "pruned_only": "PRUNE"}.get(status, "?")

            print(f"  {pp:>5}% | {kr:>.2f} | {ret_str:>6}% | {best_str:>6}% | "
                  f"{paper_ret_str:>9}% | {paper_ft_str:>8}% | {delta:>6} | "
                  f"{params_str:>8}M | {status_icon:>8}")

    print("=" * 110)
    print("  * = params estimated from paper table (log incomplete)")
    print("  FT... = fine-tuning in progress (retention available, no best/final yet)")


def plot_results(setups, save_path):
    """Plot retention and best accuracy vs params, compared to paper.

    Handles partial data: plots retention for all runs that have it,
    plots best_acc only for completed runs.
    """
    colors = plt.cm.tab10.colors
    has_any_best = any("best_acc" in r for results in setups.values() for r in results)
    n_plots = 2 if has_any_best else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7))
    if n_plots == 1:
        axes = [axes]

    # --- Paper data ---
    paper_pcts = sorted(k for k in PAPER_DEIT_T.keys() if k > 0)
    paper_params = [PAPER_DEIT_T[p][3] for p in paper_pcts]
    paper_ret = [PAPER_DEIT_T[p][0] for p in paper_pcts]
    paper_ft = [PAPER_DEIT_T[p][1] for p in paper_pcts]

    # --- Baseline point ---
    base_params = PAPER_DEIT_T[0][3]
    base_acc = PAPER_DEIT_T[0][0]

    # --- Plot 1: Retention accuracy vs Params ---
    ax = axes[0]
    ax.plot(paper_params, paper_ret, "s--", color="gray", linewidth=2,
            markersize=7, label="Paper (retention)", zorder=2, alpha=0.7)
    ax.plot(base_params, base_acc, "o", color="black", markersize=10,
            zorder=5, label=f"Baseline ({base_acc:.1f}%, {base_params:.2f}M)")

    for i, (setup_name, results) in enumerate(setups.items()):
        valid = [r for r in results if "retention_acc" in r and "pruned_params_M" in r]
        if valid:
            params = [r["pruned_params_M"] for r in valid]
            rets = [r["retention_acc"] for r in valid]
            n_total = len(results)
            n_valid = len(valid)
            label = f"{setup_name} ({n_valid}/{n_total})" if n_valid < n_total else setup_name
            ax.plot(params, rets, "o-", color=colors[i % len(colors)],
                    linewidth=2, markersize=6, label=label, zorder=3)

    ax.set_xlabel("Parameters (M)", fontsize=13)
    ax.set_ylabel("Retention Accuracy (%)", fontsize=13)
    ax.set_title("Retention Accuracy vs Parameters", fontsize=14)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Best accuracy (after FT) vs Params (only if any data) ---
    if has_any_best:
        ax = axes[1]
        ax.plot(paper_params, paper_ft, "s--", color="gray", linewidth=2,
                markersize=7, label="Paper (fine-tuned)", zorder=2, alpha=0.7)
        ax.plot(base_params, base_acc, "o", color="black", markersize=10,
                zorder=5, label=f"Baseline ({base_acc:.1f}%, {base_params:.2f}M)")

        for i, (setup_name, results) in enumerate(setups.items()):
            valid = [r for r in results if "best_acc" in r and "pruned_params_M" in r]
            if valid:
                params = [r["pruned_params_M"] for r in valid]
                bests = [r["best_acc"] for r in valid]
                n_total = len(results)
                n_valid = len(valid)
                label = f"{setup_name} ({n_valid}/{n_total})" if n_valid < n_total else setup_name
                ax.plot(params, bests, "o-", color=colors[i % len(colors)],
                        linewidth=2, markersize=6, label=label, zorder=3)

        ax.set_xlabel("Parameters (M)", fontsize=13)
        ax.set_ylabel("Best Accuracy after FT (%)", fontsize=13)
        ax.set_title("Best Accuracy (post Fine-Tuning) vs Parameters", fontsize=14)
        ax.legend(fontsize=9, loc="lower left")
        ax.grid(True, alpha=0.3)

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
    parser.add_argument("--root", default="",
                        help="Root dir containing setup subfolders (e.g., .../DeiT_tiny)")
    args = parser.parse_args()
    args.root = os.getcwd()
    args.save = os.path.join(args.root, "vbp_results.png")

    setups = scan_root(args.root)

    if not setups:
        print(f"No results found under {args.root}")
        print(f"Expected structure: {args.root}/<setup_name>/kr_<ratio>/vbp_imagenet.log")
        return

    total_runs = sum(len(v) for v in setups.values())
    n_done = sum(1 for results in setups.values() for r in results if r.get("status") == "done")
    n_partial = total_runs - n_done
    print(f"Found {len(setups)} setup(s), {total_runs} total runs ({n_done} complete, {n_partial} in-progress)")

    print_table(setups)
    plot_results(setups, args.save)


if __name__ == "__main__":
    main()
