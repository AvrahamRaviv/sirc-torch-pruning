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
from collections import defaultdict

R50_SOURCE = ("https://download.openmmlab.com/mmclassification/v0/resnet/"
              "resnet50_8xb32_in1k_20210831-ea4938fc.json")
TAG = "r50_official_mmpretrain"

MNV2_SOURCE = ("https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/"
               "mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.json")
MNV2_TAG = "mnv2_official_mmcls_train"


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


def convert_mnv2(raw_path, out_dir):
    """MobileNetV2 mmcls log is TRAIN-ONLY (no val lines published anywhere — verified). We emit a
    per-epoch TRAIN reference (mean train_loss + mean train top-1) so a from-scratch control can be
    shape-checked on train_loss (the only metric shared with our runs). val_acc is null; verify the
    final val on the known anchor ~71.86-72.0% separately."""
    loss, acc, lr_last = defaultdict(list), defaultdict(list), {}
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("mode") != "train" or "loss" not in r:
                continue
            e = r["epoch"]
            loss[e].append(r["loss"])
            # mmcls switches the train-acc field name from "acc" (early) to "top-1" (later)
            a1 = r.get("top-1", r.get("acc"))
            if a1 is not None:
                acc[e].append(a1)
            if "lr" in r:
                lr_last[e] = r["lr"]
    epochs = sorted(loss)
    n = len(epochs)
    assert n > 0, "no train lines found"

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"{MNV2_TAG}_metrics.jsonl")
    best_train = 0.0
    with open(metrics_path, "w") as mf:
        for e in epochs:
            mean_loss = sum(loss[e]) / len(loss[e])
            ta = (sum(acc[e]) / len(acc[e]) / 100.0) if acc[e] else None    # percent -> fraction
            if ta is not None:
                best_train = max(best_train, ta)
            rec = {
                "arm": "reference_train",
                "epoch": int(e),
                "epochs": n,
                "val_acc": None,                       # not published for MNv2
                "train_loss": round(mean_loss, 6),
                "train_acc": round(ta, 6) if ta is not None else None,
                "lr": lr_last.get(e),
            }
            mf.write(json.dumps(rec) + "\n")

    run_path = os.path.join(out_dir, f"{MNV2_TAG}_run.json")
    run = {
        "arm": "reference_train",
        "status": "reference_train_only",
        "source": MNV2_SOURCE,
        "note": "mmcls log is TRAIN-ONLY (no per-epoch val published). Use train_loss for shape "
                "match; anchor final val on ~71.86-72.0%.",
        "pre_train_val_acc": None,
        "best_val_acc": None,
        "final_val_anchor": 0.7186,
        "best_train_acc": round(best_train, 6),
        "config": {
            "model": "mobilenet_v2",
            "dataset": "imagenet1k",
            "lr": 0.045,
            "schedule": "step x0.98 every epoch",
            "lr_step_size": 1,
            "lr_gamma": 0.98,
            "optimizer": "sgd",
            "momentum": 0.9,
            "weight_decay": 4e-5,
            "batch_size": 256,
            "epochs": n,
        },
    }
    with open(run_path, "w") as rf:
        json.dump(run, rf, indent=2)
    print(f"wrote {metrics_path} ({n} epochs, TRAIN-only)")
    print(f"wrote {run_path}  best_train_acc={best_train:.4f}")
    print(f"  ep1 train_loss={sum(loss[epochs[0]])/len(loss[epochs[0]]):.4f}  "
          f"ep{epochs[-1]} train_loss={sum(loss[epochs[-1]])/len(loss[epochs[-1]]):.4f}")


def main(argv):
    ap = argparse.ArgumentParser(description="official log -> ExpHandler reference pair")
    ap.add_argument("raw_json", help="downloaded mmcls/mmpretrain log json")
    ap.add_argument("--model", default="r50", choices=["r50", "mnv2"],
                    help="r50 = per-epoch VAL (mmpretrain); mnv2 = per-epoch TRAIN-only (mmcls)")
    ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args(argv[1:])
    if a.model == "r50":
        convert_r50(a.raw_json, a.out_dir)
    else:
        convert_mnv2(a.raw_json, a.out_dir)


if __name__ == "__main__":
    main(sys.argv)
