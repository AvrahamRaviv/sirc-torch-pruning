#!/usr/bin/env python3
"""Replicate convnext-small experiment dir → convnext-base.

- Copies run_ddp.py (top-level), replacing 'small'→'base'.
- For each subfolder in small/: mkdir same name in base/, copy run_ddp.sh,
  replace 'small'→'base' AND scale LR/batch for ConvNeXt-Base.
"""
import shutil
from pathlib import Path

# === EDIT THESE ===
SRC = Path("/path/to/convnext-small")   # parent of run_ddp.py + 3 subfolders
DST = Path("/path/to/convnext-base")
# ==================

# small → base scaling (matches recipe: lr 5e-5→3e-5, eta_min 1e-5→6e-6, bs 128→64)
SCHED_REPLACEMENTS = {
    "--lr 5e-5":              "--lr 3e-5",
    "--ft_eta_min 1e-5":      "--ft_eta_min 6e-6",
    "--train_batch_size 128": "--train_batch_size 64",
}


def rewrite(text: str) -> str:
    text = text.replace("small", "base")
    for old, new in SCHED_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def copy_file_with_rewrite(src_file: Path, dst_file: Path) -> None:
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    dst_file.write_text(rewrite(src_file.read_text()))
    shutil.copymode(src_file, dst_file)
    print(f"  wrote {dst_file}")


def main() -> None:
    assert SRC.is_dir(), f"missing src: {SRC}"
    DST.mkdir(parents=True, exist_ok=True)

    # 1. top-level run_ddp.py
    src_py = SRC / "run_ddp.py"
    if src_py.is_file():
        copy_file_with_rewrite(src_py, DST / "run_ddp.py")
    else:
        print(f"WARN: {src_py} not found")

    # 2. each subfolder → mkdir + copy run_ddp.sh with rewrites
    for sub in sorted(p for p in SRC.iterdir() if p.is_dir()):
        new_name = sub.name.replace("small", "base")
        dst_sub = DST / new_name
        dst_sub.mkdir(exist_ok=True)
        print(f"[dir] {sub.name} -> {dst_sub}")

        src_sh = sub / "run_ddp.sh"
        if src_sh.is_file():
            copy_file_with_rewrite(src_sh, dst_sub / "run_ddp.sh")
        else:
            print(f"  WARN: {src_sh} not found")


if __name__ == "__main__":
    main()
