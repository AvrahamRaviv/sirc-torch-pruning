"""Fill the −25% MAC (75%) DeiT-tiny table: run the MISSING iterative specs at the paper's
operating point (Table 10, 45% rate = −25.2% MAC, VBP Ret 49.77). Baseline prop/cov/variance
already cached in results_deit_75.jsonl; this adds iter (cov, clean) at width+mean so the
prop/cov/iter/vbp comparison is complete at one matched MAC.

Reuses harness_deit Ctx/run_spec verbatim; only points RESULTS/LEADERBOARD at the _75 files and
sets mac_target = 0.75×dense. No edit to the immutable harness.
"""
import os
import sys
import json
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness_deit as HD

# redirect output to the −25% ledger (key doesn't encode MAC → separate file per MAC point)
HD.RESULTS = os.path.join(HERE, "results_deit_75.jsonl")
HD.LEADERBOARD = os.path.join(HERE, "LEADERBOARD_deit_75.md")

SPECS = [
    dict(name="iter_cov_width", kind="propagation_iterative",
         params=dict(normalizer="width", cov=True, join_cov=False,
                     iter_drop=128, iter_max_frac=0.6), note="iterative cov @ width"),
    dict(name="iter_cov_mean", kind="propagation_iterative",
         params=dict(normalizer="mean", cov=True, join_cov=False,
                     iter_drop=128, iter_max_frac=0.6), note="iterative cov @ mean"),
]


def main():
    args = SimpleNamespace(eval_device="mps", batch_size=64, calib_batches=50,
                           val_limit=5000, mac_target_g=0.0, pruning_ratio=0.5,
                           normalizer="width", bias_comp=True)
    ctx = HD.Ctx(args)
    args.mac_target_g = round(0.75 * ctx.dense_macs, 2)   # −25% MAC = paper 45%-rate point
    print(f"[run75] MAC target = {args.mac_target_g}G (0.75×dense = −25% MAC)", flush=True)
    done = HD.load_done()
    print(f"[run75] dense {ctx.dense_macs:.2f}G; {len(SPECS)} specs; {len(done)} cached", flush=True)
    for spec in SPECS:
        k = HD._key(spec, args.val_limit, args.bias_comp)
        if k in done:
            print(f"  skip (done): {spec['name']} acc={done[k]['res']['acc']:.4f}", flush=True)
            continue
        t0 = time.time()
        try:
            res = HD.run_spec(ctx, spec, args.val_limit)
        except Exception as e:
            import traceback
            print(f"  FAIL {spec['name']}: {e}", flush=True)
            traceback.print_exc()
            continue
        rec = dict(key=k, spec=spec, res=res, limit=args.val_limit, sec=round(time.time() - t0))
        HD.append_result(rec)
        HD.rewrite_leaderboard()
        print(f"  [{spec['name']}] top-1={res['acc']:.4f}  {res['mac_pct']:.0f}% MAC  ({rec['sec']}s)",
              flush=True)
    print("[run75] done", flush=True)


if __name__ == "__main__":
    main()
