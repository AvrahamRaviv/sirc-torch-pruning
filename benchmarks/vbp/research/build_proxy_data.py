"""Rebuild the LOCAL MNv2 retention-proxy dataset in a PERSISTENT dir (/tmp wipes overnight).

Source: HF imagenet-1k validation parquets (label = torchvision class index). Decodes a small
disjoint subset -> train_samples.pkl (calib) + val_samples.pkl (eval), each a list of
(abs_path, int_label) consumed by vbp_common.FastImageNet.

  python build_proxy_data.py            # writes ~/nm_data/in1k/{train,val}_images + pkls
"""
import os, glob, pickle
import pyarrow.parquet as pq

SRC = "/Users/avrahamraviv/PycharmProjects/imagenet-1k-test/data"
OUT = os.path.expanduser("~/nm_data/in1k")
N_VAL = 1600          # eval subset (val_limit 1500 takes first 1500)
N_TRAIN = 5000        # calib subset (16 batches * 256 = 4096 needed)

def main():
    os.makedirs(os.path.join(OUT, "val_images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "train_images"), exist_ok=True)
    shards = sorted(glob.glob(os.path.join(SRC, "validation-*.parquet")))
    need = N_VAL + N_TRAIN
    rows = []
    for sh in shards:
        for batch in pq.ParquetFile(sh).iter_batches(batch_size=512):
            rows.extend(batch.to_pylist())
            if len(rows) >= need:
                break
        if len(rows) >= need:
            break
    print(f"collected {len(rows)} rows from {shards[0]}...")
    val_samples, train_samples = [], []
    for i, r in enumerate(rows[:need]):
        split = "val_images" if i < N_VAL else "train_images"
        b = r["image"]["bytes"]
        path = os.path.join(OUT, split, f"{i:06d}.jpg")
        with open(path, "wb") as f:
            f.write(b)
        (val_samples if i < N_VAL else train_samples).append((path, int(r["label"])))
    with open(os.path.join(OUT, "val_samples.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(OUT, "train_samples.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    print(f"wrote val={len(val_samples)} train={len(train_samples)} to {OUT}")
    print("val label span:", len({l for _, l in val_samples}), "classes")

if __name__ == "__main__":
    main()
