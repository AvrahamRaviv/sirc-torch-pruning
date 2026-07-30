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

CONVNEXT_SOURCE = ("https://download.openmmlab.com/mmclassification/v0/convnext/"
                   "convnext-tiny_32xb128_in1k_20221207-998cf3e9.json")
CONVNEXT_TAG = "convnext_t_official_mmpretrain"

MNV1_IMGCLSMOB_SOURCE = ("https://github.com/osmr/imgclsmob/releases/download/v0.0.155/"
                         "mobilenet_w1-0865-eafd91e9.params.log")
MNV1_IMGCLSMOB_TAG = "mnv1_imgclsmob_train"


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


def convert_convnext(raw_path, out_dir):
    """ConvNeXt-T mmpretrain log (mmengine format, 2022-12): train lines
    {"lr","loss","epoch","step"} every 100 steps + ONE val line {"accuracy/top1","accuracy/top5",
    "step"} per epoch where "step" = epoch number. CAVEAT baked into the note: the published
    per-epoch val is EMA-evaluated — near-zero until ~e40, crosses 50% ~e70; a raw-model curve
    climbs much faster early. Don't panic-compare early epochs; shape + final (82.16) are the
    anchors."""
    val = {}                                  # epoch -> (top1, top5)
    loss, lr_last = defaultdict(list), {}
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "accuracy/top1" in r:
                val[int(r["step"])] = (r["accuracy/top1"], r.get("accuracy/top5"))
            elif "loss" in r and "epoch" in r:
                e = int(r["epoch"])
                loss[e].append(r["loss"])
                lr_last[e] = r.get("lr")
    epochs = sorted(val)
    n = len(epochs)
    assert n > 0, "no accuracy/top1 lines found"

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"{CONVNEXT_TAG}_metrics.jsonl")
    best = 0.0
    with open(metrics_path, "w") as mf:
        for e in epochs:
            top1, top5 = val[e]
            val_acc = top1 / 100.0
            best = max(best, val_acc)
            rec = {
                "arm": "reference",
                "epoch": e,
                "epochs": n,
                "val_acc": round(val_acc, 6),        # NOTE: EMA-evaluated (see run.json note)
                "val_top5": round(top5 / 100.0, 6) if top5 is not None else None,
                "best_val_acc": round(best, 6),
                "lr": lr_last.get(e),
            }
            if loss.get(e):
                rec["train_loss"] = round(sum(loss[e]) / len(loss[e]), 6)
            mf.write(json.dumps(rec) + "\n")

    run_path = os.path.join(out_dir, f"{CONVNEXT_TAG}_run.json")
    run = {
        "arm": "reference",
        "status": "reference",
        "source": CONVNEXT_SOURCE,
        "note": "per-epoch val is EMA-evaluated: near-zero until ~e40 (e30 0.6%, e50 9.5%), "
                "e100 73.3%, e200 80.9%, final 82.16%. A raw-model (non-EMA) curve climbs much "
                "faster early — compare SHAPE mid/late + final, not early epochs. train_loss is "
                "the raw-model signal and IS comparable per-epoch.",
        "pre_train_val_acc": None,
        "best_val_acc": round(best, 6),
        "config": {
            "model": "convnext_tiny",
            "dataset": "imagenet1k",
            "lr": 4e-3,
            "schedule": "cosine, 20ep linear warmup",
            "optimizer": "adamw",
            "weight_decay": 0.05,
            "batch_size": 4096,
            "epochs": n,
            "extras_not_in_our_trainer": ["mixup 0.8", "cutmix 1.0", "randaug(9,0.5)",
                                          "label_smoothing 0.1", "EMA 0.9999",
                                          "layer-wise LR decay: none", "drop_path 0.1 (we HAVE "
                                          "--drop_path)"],
        },
    }
    with open(run_path, "w") as rf:
        json.dump(run, rf, indent=2)
    print(f"wrote {metrics_path} ({n} epochs)")
    print(f"wrote {run_path}  best_val_acc={best:.4f} (EMA-evaluated)")


def convert_mnv1_imgclsmob(raw_path, out_dir):
    """imgclsmob (osmr) MobileNet-w1 Gluon training log — the ONLY published per-epoch MNv1
    curve found anywhere. NOT the official recipe: NAG SGD lr 0.5 cosine 210ep, effective
    bs 1120 (224 x batch-size-scale 5), label smoothing, 5ep warmup, no WD on bn/bias ->
    73.57% top-1 (official paper recipe lands 70.6-70.9). Use as a SHAPE/sanity reference
    for a stronger recipe, not as the target of our MNv2-style repro run.

    Log format (text): per-epoch triplet
        [Epoch N] speed: ...
        [Epoch N] training: err-top1=X loss=Y
        [Epoch N] validation: err-top1=X err-top5=Y
    plus batch lines "Epoch[N] Batch [k] ... lr=Z" (we keep the LAST lr seen per epoch)."""
    import re
    re_train = re.compile(r"\[Epoch (\d+)\] training: err-top1=([\d.]+)\s+loss=([\d.]+)")
    re_val = re.compile(r"\[Epoch (\d+)\] validation: err-top1=([\d.]+)\s+err-top5=([\d.]+)")
    re_lr = re.compile(r"Epoch\[(\d+)\].*\blr=([\d.eE+-]+)")
    train, val, lr_last = {}, {}, {}
    with open(raw_path, errors="replace") as f:
        for line in f:
            m = re_train.search(line)
            if m:
                train[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
                continue
            m = re_val.search(line)
            if m:
                val[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
                continue
            m = re_lr.search(line)
            if m:
                lr_last[int(m.group(1))] = float(m.group(2))
    epochs = sorted(val)
    n = len(epochs)
    assert n > 0, "no validation lines found"

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"{MNV1_IMGCLSMOB_TAG}_metrics.jsonl")
    best = 0.0
    with open(metrics_path, "w") as mf:
        for e in epochs:
            err1, err5 = val[e]
            val_acc = 1.0 - err1
            best = max(best, val_acc)
            rec = {
                "arm": "reference_thirdparty",
                "epoch": e,
                "epochs": n,
                "val_acc": round(val_acc, 6),
                "val_top5": round(1.0 - err5, 6),
                "best_val_acc": round(best, 6),
                "lr": lr_last.get(e),
            }
            if e in train:
                rec["train_acc"] = round(1.0 - train[e][0], 6)
                rec["train_loss"] = round(train[e][1], 6)
            mf.write(json.dumps(rec) + "\n")

    run_path = os.path.join(out_dir, f"{MNV1_IMGCLSMOB_TAG}_run.json")
    run = {
        "arm": "reference_thirdparty",
        "status": "reference_nonofficial_recipe",
        "source": MNV1_IMGCLSMOB_SOURCE,
        "note": "ONLY published per-epoch MNv1 curve anywhere (imgclsmob, Gluon/MXNet, 2018). "
                "STRONGER recipe than official paper -> 73.57% vs official 70.6-70.9. Our "
                "MNv2-style repro run targets the official anchor (mnv1_official_anchor_run"
                ".json); use this curve for shape/sanity only (e.g. what epoch crosses 60/70%).",
        "pre_train_val_acc": None,
        "best_val_acc": round(best, 6),
        "config": {
            "model": "mobilenet_w1",
            "dataset": "imagenet1k",
            "lr": 0.5,
            "schedule": "cosine, 5ep warmup",
            "optimizer": "nag (nesterov sgd)",
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "wd_excludes": "bn gamma/beta + bias",
            "label_smoothing": True,
            "batch_size": 1120,
            "epochs": n,
        },
    }
    with open(run_path, "w") as rf:
        json.dump(run, rf, indent=2)
    print(f"wrote {metrics_path} ({n} epochs)")
    print(f"wrote {run_path}  best_val_acc={best:.4f} (third-party recipe)")


def write_mnv1_anchor(out_dir):
    """MobileNetV1: NO official per-epoch log exists anywhere (paper=TF internal, TF-slim
    publishes final ckpt only 70.9%, mmpretrain has no V1, timm publishes no logs). Emit a
    run.json anchor (no metrics.jsonl) so the comparison target is recorded: final-only,
    like MNv2 val."""
    run_path = os.path.join(out_dir, "mnv1_official_anchor_run.json")
    run = {
        "arm": "reference",
        "status": "reference_final_only",
        "source": "paper arXiv:1704.04861 (70.6%); TF-slim ckpt mobilenet_v1_1.0_224 (70.9%)",
        "note": "NO per-epoch curve published anywhere for MNv1 — verify from-scratch control "
                "on the FINAL number only (70.6-70.9). Recipe per paper = 'RMSprop, like "
                "Inception V3, less regularization'; the concrete MobileNet-family recipe "
                "(lr 0.045, x0.98/epoch, wd 4e-5, momentum 0.9) is the V2-paper restatement "
                "of the same setup — our SGD port of it matches the MNv2 baseline_repro run.",
        "final_val_anchor": 0.709,
        "config": {
            "model": "mobilenet_v1_1.0_224",
            "dataset": "imagenet1k",
            "lr": 0.045,
            "schedule": "step x0.98 every epoch",
            "lr_step_size": 1,
            "lr_gamma": 0.98,
            "optimizer": "rmsprop (ours: sgd)",
            "momentum": 0.9,
            "weight_decay": 4e-5,
            "batch_size": 256,
            "epochs": 300,
        },
    }
    with open(run_path, "w") as rf:
        json.dump(run, rf, indent=2)
    print(f"wrote {run_path} (final-only anchor 0.709, no metrics.jsonl — none published)")


def main(argv):
    ap = argparse.ArgumentParser(description="official log -> ExpHandler reference pair")
    ap.add_argument("raw_json", nargs="?", default=None,
                    help="downloaded mmcls/mmpretrain log json (not needed for mnv1_anchor)")
    ap.add_argument("--model", default="r50",
                    choices=["r50", "mnv2", "convnext", "mnv1_anchor", "mnv1_imgclsmob"],
                    help="r50/convnext = per-epoch VAL (mmpretrain); mnv2 = per-epoch TRAIN-only "
                         "(mmcls); mnv1_anchor = run.json only (no official log exists); "
                         "mnv1_imgclsmob = third-party per-epoch curve (stronger recipe, 73.57%)")
    ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args(argv[1:])
    if a.model == "mnv1_anchor":
        write_mnv1_anchor(a.out_dir)
        return
    assert a.raw_json, "raw_json required for this model"
    if a.model == "r50":
        convert_r50(a.raw_json, a.out_dir)
    elif a.model == "convnext":
        convert_convnext(a.raw_json, a.out_dir)
    elif a.model == "mnv1_imgclsmob":
        convert_mnv1_imgclsmob(a.raw_json, a.out_dir)
    else:
        convert_mnv2(a.raw_json, a.out_dir)


if __name__ == "__main__":
    main(sys.argv)
