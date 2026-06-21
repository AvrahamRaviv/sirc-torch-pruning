"""Compare two --dump_artifacts dirs (e.g. local vs cluster) layer-by-layer.

Localizes WHERE the cross-platform pipeline first diverges: stats (mu_x/sigma_x/sigma_out_x) →
input_cov → scores. For each artifact and layer prints relative max-abs-diff + cosine; for
scores also Spearman rank-corr + top-K prune-set Jaccard (what the pruner actually consumes).
The EARLIEST artifact (stats < cov < scores) that diverges is the stage to drill.

Usage:
  python probe_compare.py --a /tmp/art_local --b /tmp/art_cluster [--topk_frac 0.4]
"""
import argparse
import os

import torch


def _rel(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    denom = a.abs().max().clamp_min(1e-12)
    max_abs = (a - b).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a[None], b[None]).item()
    return max_abs / denom.item(), cos


def _spearman(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    n = min(a.numel(), b.numel())
    ra = a[:n].argsort().argsort().float()
    rb = b[:n].argsort().argsort().float()
    return torch.nn.functional.cosine_similarity(
        (ra - ra.mean())[None], (rb - rb.mean())[None]).item()


def _jaccard_bottom(a, b, frac):
    a, b = a.float().flatten(), b.float().flatten()
    n = min(a.numel(), b.numel())
    k = max(1, int(frac * n))
    sa = set(a[:n].argsort()[:k].tolist())     # bottom-k = pruned-first
    sb = set(b[:n].argsort()[:k].tolist())
    return len(sa & sb) / len(sa | sb)


def _load(d, name):
    p = os.path.join(d, name)
    return torch.load(p, map_location="cpu", weights_only=False) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="artifact dir A (e.g. local)")
    ap.add_argument("--b", required=True, help="artifact dir B (e.g. cluster)")
    ap.add_argument("--topk_frac", type=float, default=0.4, help="prune-set Jaccard fraction")
    ap.add_argument("--worst", type=int, default=12, help="show N worst-diverging layers")
    args = ap.parse_args()

    # ---- stats (per layer: dict of mu_x/sigma_x/sigma_out_x) ----
    sa, sb = _load(args.a, "stats.pt"), _load(args.b, "stats.pt")
    if sa and sb:
        print("\n=== STATS (rel-maxdiff / cosine), worst layers per stat ===")
        for stat in ("mu_x", "sigma_x", "sigma_out_x"):
            rows = []
            for n in sa:
                if n in sb and stat in sa[n] and stat in sb[n]:
                    rd, cos = _rel(sa[n][stat], sb[n][stat])
                    rows.append((rd, cos, n))
            rows.sort(reverse=True)
            print(f"  [{stat}] worst:")
            for rd, cos, n in rows[:args.worst]:
                print(f"    {n:28s} reldiff={rd:.3e} cos={cos:.5f}")

    # ---- cov (per layer correlation matrix) ----
    ca, cb = _load(args.a, "cov.pt"), _load(args.b, "cov.pt")
    if ca and cb:
        print("\n=== INPUT_COV (rel-maxdiff / cosine), worst layers ===")
        rows = []
        for n in ca:
            if n in cb and ca[n].shape == cb[n].shape:
                rd, cos = _rel(ca[n], cb[n])
                rows.append((rd, cos, n))
        rows.sort(reverse=True)
        for rd, cos, n in rows[:args.worst]:
            print(f"    {n:28s} reldiff={rd:.3e} cos={cos:.5f}")

    # ---- scores (per layer: what the ranker consumes) ----
    xa, xb = _load(args.a, "scores.pt"), _load(args.b, "scores.pt")
    if xa and xb:
        print(f"\n=== SCORES (cosine / spearman / bottom-{args.topk_frac:.0%} Jaccard), worst ===")
        rows = []
        for n in xa:
            if n in xb and xa[n].numel() == xb[n].numel() and xa[n].numel() > 1:
                _, cos = _rel(xa[n], xb[n])
                sp = _spearman(xa[n], xb[n])
                jac = _jaccard_bottom(xa[n], xb[n], args.topk_frac)
                rows.append((sp, cos, jac, n))
        rows.sort()                                     # lowest spearman = worst reordering
        for sp, cos, jac, n in rows[:args.worst]:
            print(f"    {n:28s} spearman={sp:.4f} cos={cos:.4f} jaccard={jac:.3f}")

    # ---- mean_dict (bias_comp) ----
    ma, mb = _load(args.a, "mean_dict.pt"), _load(args.b, "mean_dict.pt")
    if ma and mb:
        print("\n=== MEAN_DICT / bias_comp (rel-maxdiff / cosine), worst ===")
        rows = []
        for n in ma:
            if n in mb and ma[n].numel() == mb[n].numel():
                rd, cos = _rel(ma[n], mb[n])
                rows.append((rd, cos, n))
        rows.sort(reverse=True)
        for rd, cos, n in rows[:args.worst]:
            print(f"    {n:28s} reldiff={rd:.3e} cos={cos:.5f}")


if __name__ == "__main__":
    main()
