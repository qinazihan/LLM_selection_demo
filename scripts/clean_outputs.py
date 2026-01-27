#!/usr/bin/env python3
"""
Remove everything under outputs/ except:
- outputs/model_checks.jsonl
- outputs/dataset_simple/ (kept entirely)

Usage:
    python scripts/clean_outputs.py
"""
from __future__ import annotations

import shutil
from pathlib import Path


ALLOWED = {
    Path("outputs/model_checks.jsonl"),
    Path("outputs/dataset_simple"),
}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    outputs_dir = root / "outputs"
    if not outputs_dir.exists():
        print("outputs/ does not exist; nothing to clean.")
        return

    for entry in outputs_dir.iterdir():
        rel = entry.relative_to(root)
        if rel in ALLOWED or any(rel == p or rel.is_relative_to(p) for p in ALLOWED):
            print(f"Keep: {rel}")
            continue

        if entry.is_dir():
            print(f"Delete dir: {rel}")
            shutil.rmtree(entry)
        else:
            print(f"Delete file: {rel}")
            entry.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
