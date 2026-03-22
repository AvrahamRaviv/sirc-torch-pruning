"""Extract key metrics from VBP experiment logs into JSON/CSV.

Parses hyperparams, step retention, per-epoch val_acc/train_loss,
and final summary from vbp_imagenet.log files.

Expected folder structure:
    <root_dir>/
        <setup_name>/
            kr_0.95/
                vbp_imagenet.log
            kr_0.90/
                vbp_imagenet.log

Usage:
    # All setups, all keep ratios → JSON
    python parse_logs.py --root /path/to/experiments

    # Specific setups and keep ratios
    python parse_logs.py --root /path/to/experiments \
        --setups global_kd_lr1e-4_1010_wvnr_bn \
        --keep_ratios 0.95 0.90 0.85

    # CSV output
    python parse_logs.py --root /path/to/experiments --format csv

    # Print to stdout (no file save)
    python parse_logs.py --root /path/to/experiments --stdout
"""

import argparse
import csv
import json
import os
import re
import sys


def parse_hyperparams(text):
    """Extract hyperparams from the args dump block at the top of the log."""
    params = {}
    # Args are logged as "  key: value" after the "Unified Pipeline" or "=" header
    in_args = False
    for line in text.split("\n"):
        if "Unified Pipeline" in line or "VBP Pruning" in line:
            in_args = True
            continue
        if in_args:
            m = re.search(r"INFO \|   (\w+): (.*)", line)
            if m:
                key, val = m.group(1), m.group(2).strip()
                # Convert types
                if val in ("True", "False"):
                    val = val == "True"
                elif val == "None":
                    val = None
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                params[key] = val
            elif params:
                # First non-matching line after we started → done
                break
    return params


def parse_epochs(text):
    """Extract per-epoch metrics: phase, train_loss, val_acc, MACs, aux losses."""
    epochs = []
    # Pattern: [FT] Epoch 1/10: train_loss=0.9960, val_acc=0.6963, MACs=1.04G
    # Also: [Sparse] Epoch 1/5: ... and [PAT] Epoch 3/15: ...
    for m in re.finditer(
        r"\[(\w+)\]\s*Epoch\s+(\d+)/(\d+):\s*train_loss=([\d.]+),\s*val_acc=([\d.]+),\s*MACs=([\d.]+)G"
        r"(?:\s*\|\s*(.*?))?$",
        text, re.MULTILINE
    ):
        entry = {
            "phase": m.group(1),
            "epoch": int(m.group(2)),
            "total_epochs": int(m.group(3)),
            "train_loss": float(m.group(4)),
            "val_acc": float(m.group(5)),
            "macs_G": float(m.group(6)),
        }
        # Parse auxiliary losses (e.g. "reg=0.0012 ent=0.0034")
        if m.group(7):
            for aux in re.finditer(r"(\w+)=([\d.]+)", m.group(7)):
                entry[aux.group(1)] = float(aux.group(2))
        epochs.append(entry)
    return epochs


def parse_step_retentions(text):
    """Extract step retention metrics (after each pruning step)."""
    retentions = []
    for m in re.finditer(
        r"(?:Step\s+)?[Rr]etention.*?acc=([\d.]+),\s*loss=([\d.]+),\s*MACs=([\d.]+)G",
        text
    ):
        retentions.append({
            "acc": float(m.group(1)),
            "loss": float(m.group(2)),
            "macs_G": float(m.group(3)),
        })
    return retentions


def parse_pruning_channels(text):
    """Extract per-layer pruning channel counts from physical Prune lines.

    Only parses lines after "Last pruning step: switching to physical channel removal"
    to avoid counting earlier Mask steps. Matches both "Prune" and "Prune+comp".
    Returns list of dicts: {layer, pruned, total, pruned_pct}.
    """
    # Find the last physical pruning block
    anchor = text.rfind("Last pruning step: switching to physical channel removal")
    if anchor == -1:
        # Fallback: no anchor found, scan entire text for Prune (not Mask) lines
        region = text
    else:
        region = text[anchor:]

    layers = []
    for m in re.finditer(r"Prune(?:\+comp)?\s+(\d+)/(\d+)\s+channels?\s+on\s+(\S+)", region):
        pruned, total = int(m.group(1)), int(m.group(2))
        layers.append({
            "layer": m.group(3).rstrip("."),
            "pruned": pruned,
            "total": total,
            "pruned_pct": 100.0 * pruned / total if total > 0 else 0.0,
        })
    return layers


def parse_summary(text):
    """Extract final summary block."""
    summary = {}

    m = re.search(r"Original [Aa]cc(?:uracy)?:\s*([\d.]+)", text)
    if m:
        summary["original_acc"] = float(m.group(1))

    m = re.search(r"Final [Aa]cc(?:uracy)?:\s*([\d.]+)", text)
    if m:
        summary["final_acc"] = float(m.group(1))

    # Use last match — first may be from sparse phase (before pruning)
    best_matches = list(re.finditer(r"Best [Aa]cc(?:uracy)?:\s*([\d.]+)", text))
    if best_matches:
        summary["best_acc"] = float(best_matches[-1].group(1))

    m = re.search(r"Baseline:\s*([\d.]+)G MACs,\s*([\d.]+)M params", text)
    if m:
        summary["base_macs_G"] = float(m.group(1))
        summary["base_params_M"] = float(m.group(2))

    m = re.search(r"Base MACs:\s*([\d.]+)G\s*->\s*Pruned:\s*([\d.]+)G", text)
    if m:
        summary["base_macs_G"] = float(m.group(1))
        summary["pruned_macs_G"] = float(m.group(2))

    m = re.search(r"Base Params:\s*([\d.]+)M\s*->\s*Pruned:\s*([\d.]+)M", text)
    if m:
        summary["base_params_M"] = float(m.group(1))
        summary["pruned_params_M"] = float(m.group(2))

    return summary


def parse_log(log_path):
    """Parse a single log file into structured data."""
    with open(log_path, "r") as f:
        text = f.read()

    # Key hyperparams to surface (subset of all args)
    hp_keys = [
        "model_type", "model_name", "checkpoint", "cnn_arch",
        "criterion", "keep_ratio", "global_pruning", "mac_target",
        "pat_steps", "pat_epochs_per_step", "epochs_ft", "epochs_sparse",
        "lr", "opt", "wd", "ft_eta_min", "ft_warmup_epochs",
        "sparse_mode", "reparam_lambda", "reparam_normalize",
        "reparam_during_pat", "reparam_target",
        "use_kd", "kd_alpha", "kd_T",
        "no_compensation", "norm_per_layer", "similarity_discount",
        "pruning_schedule", "importance_mode",
        "var_loss_weight",
    ]

    all_params = parse_hyperparams(text)
    hyperparams = {k: all_params[k] for k in hp_keys if k in all_params}

    result = {
        "hyperparams": hyperparams,
        "step_retentions": parse_step_retentions(text),
        "epochs": parse_epochs(text),
        "summary": parse_summary(text),
        "pruning_channels": parse_pruning_channels(text),
    }

    # Infer keep_ratio from folder name if not in log
    if "keep_ratio" not in hyperparams:
        folder = os.path.basename(os.path.dirname(log_path))
        m = re.match(r"kr_([\d.]+)", folder)
        if m:
            hyperparams["keep_ratio"] = float(m.group(1))

    return result


def scan_experiments(root_dir, setups=None, keep_ratios=None):
    """Scan root_dir, optionally filtering by setup name and keep ratio."""
    results = {}

    for setup_name in sorted(os.listdir(root_dir)):
        setup_dir = os.path.join(root_dir, setup_name)
        if not os.path.isdir(setup_dir):
            continue
        if setups and setup_name not in setups:
            continue

        for kr_folder in sorted(os.listdir(setup_dir)):
            kr_dir = os.path.join(setup_dir, kr_folder)
            if not os.path.isdir(kr_dir):
                continue

            # Filter by keep ratio if specified
            if keep_ratios:
                m = re.match(r"kr_([\d.]+)", kr_folder)
                if m and float(m.group(1)) not in keep_ratios:
                    continue

            log_path = os.path.join(kr_dir, "vbp_imagenet.log")
            if not os.path.exists(log_path):
                continue

            key = f"{setup_name}/{kr_folder}"
            results[key] = parse_log(log_path)
            results[key]["log_path"] = log_path

    return results


def flatten_for_csv(results):
    """Flatten nested results into flat rows for CSV output."""
    rows = []
    for key, data in results.items():
        setup, kr_folder = key.split("/", 1)
        hp = data["hyperparams"]
        summary = data["summary"]
        epochs = data["epochs"]
        retentions = data["step_retentions"]

        row = {"setup": setup, "kr_folder": kr_folder}
        row.update({f"hp_{k}": v for k, v in hp.items()})
        row.update({f"sum_{k}": v for k, v in summary.items()})

        # Last step retention
        if retentions:
            last_ret = retentions[-1]
            row["retention_acc"] = last_ret["acc"]
            row["retention_loss"] = last_ret["loss"]
            row["retention_macs_G"] = last_ret["macs_G"]

        # Per-epoch val_acc as columns
        for ep in epochs:
            phase = ep["phase"]
            row[f"{phase}_ep{ep['epoch']}_val_acc"] = ep["val_acc"]
            row[f"{phase}_ep{ep['epoch']}_train_loss"] = ep["train_loss"]

        # Best/final from epochs
        ft_epochs = [e for e in epochs if e["phase"] == "FT"]
        if ft_epochs:
            row["ft_best_val_acc"] = max(e["val_acc"] for e in ft_epochs)
            row["ft_final_val_acc"] = ft_epochs[-1]["val_acc"]

        rows.append(row)
    return rows


def print_compact(results, prune_channels=False, epoch_interval=20):
    """Print compact summary optimized for LLM analysis.

    Layout: one config block per setup, then a table with one row per KR
    showing retention → val_acc every N epochs → best/final.
    Minimal tokens, maximum information density.
    """
    # Group by setup
    by_setup = {}
    for key, data in results.items():
        setup, kr_folder = key.split("/", 1)
        by_setup.setdefault(setup, []).append((kr_folder, data))

    for setup, entries in by_setup.items():
        # Print config from first entry (shared across KRs in same setup)
        hp = entries[0][1]["hyperparams"]
        config_keys = [
            "criterion", "lr", "ft_eta_min", "ft_warmup_epochs",
            "epochs_ft", "epochs_sparse", "sparse_mode", "pat_steps",
            "pat_epochs_per_step", "use_kd", "opt", "wd",
            "reparam_lambda", "reparam_normalize", "reparam_during_pat",
            "pruning_schedule", "importance_mode", "no_compensation",
        ]
        parts = [f"{k}={hp[k]}" for k in config_keys
                 if k in hp and hp[k] is not None and hp[k] is not False
                 and hp[k] != "none" and hp[k] != 0]
        print(f"\n=== {setup} ===")
        print(f"Config: {', '.join(parts)}")

        # Sparse stats sub-table (only if any KR has sparse epochs)
        max_sp_epochs = max(
            (len([e for e in data["epochs"] if e["phase"] == "Sparse"])
             for _, data in entries),
            default=0
        )
        if max_sp_epochs > 0:
            sp_ep_hdr = "".join(f" {'S'+str(i+1):>13}" for i in range(max_sp_epochs))
            print(f"  Sparse (acc|reg):  {sp_ep_hdr}")
            for kr_folder, data in sorted(entries, key=lambda x: -x[1]["hyperparams"].get("keep_ratio", 0)):
                kr = data["hyperparams"].get("keep_ratio", "?")
                sp_eps = [e for e in data["epochs"] if e["phase"] == "Sparse"]
                cells = ""
                for e in sp_eps:
                    reg = e.get("reg")
                    reg_str = f"{reg:.4f}" if reg is not None else "     -"
                    cells += f" {e['val_acc']:.4f}|{reg_str}"
                # Pad missing epochs
                cells += "".join(f" {'':>13}" for _ in range(max_sp_epochs - len(sp_eps)))
                print(f"  {kr:>5}{cells}")

        # Collect all training epoch counts (PAT + FT) to build sampled header
        max_train_epochs = 0
        for _, data in entries:
            train_eps = [e for e in data["epochs"] if e["phase"] in ("FT", "PAT")]
            max_train_epochs = max(max_train_epochs, len(train_eps))

        # Sampled epoch indices (every Nth, 0-based)
        sample_indices = list(range(epoch_interval - 1, max_train_epochs, epoch_interval))

        # Header
        ep_cols = "".join(f" {'E'+str(i+1):>6}" for i in sample_indices)
        print(f"  {'KR':>5} {'SpBest':>7} {'Ret':>7} {ep_cols} {'Best':>7} {'MACs':>6}")

        # One row per keep ratio
        for kr_folder, data in sorted(entries, key=lambda x: -x[1]["hyperparams"].get("keep_ratio", 0)):
            kr = data["hyperparams"].get("keep_ratio", "?")
            retentions = data["step_retentions"]
            epochs = data["epochs"]
            summary = data["summary"]

            # Best val_acc during sparse phase
            sp_eps = [e for e in epochs if e["phase"] == "Sparse"]
            sp_best = max((e["val_acc"] for e in sp_eps), default=None)
            sp_str = f"{sp_best:.4f}" if sp_best is not None else "     -"

            # Retention (last step)
            ret_str = f"{retentions[-1]['acc']:.4f}" if retentions else "  -"

            # Per-epoch val_acc (PAT + FT combined, sampled every 20 epochs)
            train_eps = [e for e in epochs if e["phase"] in ("FT", "PAT")]
            ep_vals = ""
            for idx in sample_indices:
                if idx < len(train_eps):
                    ep_vals += f" {train_eps[idx]['val_acc']:>6.4f}"
                else:
                    ep_vals += "      -"

            # Best acc after physical pruning (FT phase only)
            ft_eps = [e for e in epochs if e["phase"] == "FT"]
            best = summary.get("best_acc")
            if best is None and ft_eps:
                best = max(e["val_acc"] for e in ft_eps)
            best_str = f"{best:.4f}" if best else "  -"

            # MACs
            macs = retentions[-1]["macs_G"] if retentions else summary.get("pruned_macs_G")
            macs_str = f"{macs:.2f}G" if macs else "  -"

            print(f"  {kr:>5} {sp_str:>7} {ret_str:>7} {ep_vals} {best_str:>7} {macs_str:>6}")

        # Original accuracy (once per setup)
        orig = entries[0][1]["summary"].get("original_acc")
        if orig:
            print(f"  Original: {orig:.4f}")

    # Per-layer pruning channels: one table per setup, KRs as columns
    has_pruning = any(data.get("pruning_channels") for data in results.values())
    if has_pruning and prune_channels:
        print(f"\n{'='*70}")
        print("Per-layer pruning (pruned%)")
        print(f"{'='*70}")
        for setup, entries in by_setup.items():
            # Filter to entries with pruning data, sorted by KR descending
            pruned_entries = [(kr, d) for kr, d in entries if d.get("pruning_channels")]
            if not pruned_entries:
                continue
            pruned_entries.sort(key=lambda x: -x[1]["hyperparams"].get("keep_ratio", 0))

            # Collect all layer names (preserve order from first KR that has them)
            all_layers = []
            seen = set()
            for _, d in pruned_entries:
                for ch in d["pruning_channels"]:
                    if ch["layer"] not in seen:
                        all_layers.append(ch["layer"])
                        seen.add(ch["layer"])

            # Build per-KR lookup: layer → {pruned, total, pruned_pct}
            kr_labels = []
            kr_lookups = []
            for kr_folder, d in pruned_entries:
                kr = d["hyperparams"].get("keep_ratio", kr_folder)
                kr_labels.append(str(kr))
                lookup = {ch["layer"]: ch for ch in d["pruning_channels"]}
                kr_lookups.append(lookup)

            col_w = 7
            kr_hdr = "".join(f" {k:>{col_w}}" for k in kr_labels)
            print(f"\n--- {setup} ---")
            print(f"  {'Layer':<40} {'Tot':>4}{kr_hdr}")
            for layer in all_layers:
                # Total from first KR that has this layer
                total = next((lk[layer]["total"] for lk in kr_lookups if layer in lk), "?")
                cells = ""
                for lk in kr_lookups:
                    if layer in lk:
                        cells += f" {lk[layer]['pruned_pct']:>{col_w}.1f}"
                    else:
                        cells += f" {'-':>{col_w}}"
                print(f"  {layer:<40} {total:>4}{cells}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract key metrics from VBP experiment logs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default=".",
                        help="Root dir containing setup subfolders")
    parser.add_argument("--setups", nargs="+", default=None,
                        help="Filter to these setup names only")
    parser.add_argument("--keep_ratios", nargs="+", type=float, default=None,
                        help="Filter to these keep ratios only")
    parser.add_argument("--format", choices=["json", "csv"], default="json",
                        help="Output format")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: <root>/experiment_summary.<fmt>)")
    parser.add_argument("--stdout", action="store_true",
                        help="Print compact summary to stdout instead of saving file")
    parser.add_argument("--epoch_interval", type=int, default=20,
                        help="Show val_acc every N epochs in compact output")
    parser.add_argument("--prune_channels", action="store_true",
                        help="Show per-layer pruning channel breakdown")
    args = parser.parse_args()

    results = scan_experiments(args.root, args.setups, args.keep_ratios)

    if not results:
        print(f"No logs found under {args.root}")
        print(f"Expected: <root>/<setup>/<kr_folder>/vbp_imagenet.log")
        sys.exit(1)

    print(f"Parsed {len(results)} experiment(s)")

    if args.stdout:
        print_compact(results, prune_channels=args.prune_channels,
                      epoch_interval=args.epoch_interval)
        return

    out_path = args.output or os.path.join(
        args.root, f"experiment_summary.{args.format}")

    if args.format == "json":
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
    else:
        rows = flatten_for_csv(results)
        if not rows:
            print("No data to write")
            return
        all_keys = []
        for row in rows:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
