"""Convert official published training logs into our ExpHandler format (parse_normnet.py reader).

Emits the <tag>_run.json + <tag>_metrics.jsonl pair that parse_normnet._load() reads, so an
official reference curve sits alongside our own from-scratch runs and can be diffed per-epoch.

Currently: ResNet50 mmpretrain `resnet50_8xb32_in1k` (100ep, SGD lr0.1, step x0.1 @30/60/90,
bs256, wd1e-4 -> 76.55% top-1). Its log is JSONL with one {"mode":"val","epoch":N,
"accuracy_top-1":<percent>,"lr":...} line per epoch. We store val_acc as a FRACTION (0-1) to
match our convention (mmpretrain logs percent).

MobileNetV2: NO per-epoch VAL curve is published anywhere (mmpretrain log is train-only), so no
reference jsonl is generated for it -- verify MNv2 control on the FINAL number only (71.86-72.0).

Usage:
    # download once:
    curl -sSL https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.json -o r50_mmpretrain_raw.json
    python make_reference_curves.py r50_mmpretrain_raw.json
"""
import argparse
import json
import os
import sys

R50_SOURCE = ("https://download.openmmlab.com/mmclassification/v0/resnet/"
              "resnet50_8xb32_in1k_20210831-ea4938fc.json")
TAG = "r50_official_mmpretrain"


def convert_r50(raw_path, out_dir):
    val_recs = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("mode") != "val" or "accuracy_top-1" not in r:
                continue
            val_recs.append(r)
    val_recs.sort(key=lambda r: r["epoch"])
    n = len(val_recs)
    assert n > 0, "no val lines found in raw log"

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"{TAG}_metrics.jsonl")
    best = 0.0
    with open(metrics_path, "w") as mf:
        for r in val_recs:
            val_acc = r["accuracy_top-1"] / 100.0          # percent -> fraction (our convention)
            best = max(best, val_acc)
            rec = {
                "arm": "reference",
                "epoch": int(r["epoch"]),
                "epochs": n,
                "val_acc": round(val_acc, 6),
                "val_top5": round(r.get("accuracy_top-5", 0.0) / 100.0, 6),
                "best_val_acc": round(best, 6),
                "lr": r.get("lr"),
            }
            mf.write(json.dumps(rec) + "\n")

    run_path = os.path.join(out_dir, f"{TAG}_run.json")
    run = {
        "arm": "reference",
        "status": "reference",
        "source": R50_SOURCE,
        "pre_train_val_acc": None,
        "best_val_acc": round(best, 6),
        "config": {
            "model": "resnet50",
            "dataset": "imagenet1k",
            "lr": 0.1,
            "schedule": "step x0.1 @ 30/60/90",
            "lr_milestones": [30, 60, 90],
            "lr_gamma": 0.1,
            "optimizer": "sgd",
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "epochs": n,
        },
    }
    with open(run_path, "w") as rf:
        json.dump(run, rf, indent=2)

    print(f"wrote {metrics_path} ({n} epochs)")
    print(f"wrote {run_path}  best_val_acc={best:.4f}")
    print(f"  final epoch {val_recs[-1]['epoch']}: {val_recs[-1]['accuracy_top-1']:.3f}% "
          f"-> val_acc={val_recs[-1]['accuracy_top-1']/100:.4f}")


def main(argv):
    ap = argparse.ArgumentParser(description="official log -> ExpHandler reference pair")
    ap.add_argument("raw_json", help="downloaded mmpretrain resnet50_8xb32_in1k_*.json")
    ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args(argv[1:])
    convert_r50(a.raw_json, a.out_dir)


if __name__ == "__main__":
    main(sys.argv)
