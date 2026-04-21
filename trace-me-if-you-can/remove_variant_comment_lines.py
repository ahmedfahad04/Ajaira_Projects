#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse


def should_remove_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("#") and "Variant" in stripped


def clean_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    filtered = [line for line in lines if not should_remove_line(line)]
    updated = "".join(filtered)

    if updated == original:
        return False

    path.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove comment lines that contain 'Variant' from Python files."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="extracted_code",
        help="Root folder to scan (default: extracted_code)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    changed = 0
    scanned = 0
    for py_file in root.rglob("*.py"):
        scanned += 1
        if clean_file(py_file):
            changed += 1

    print(f"Scanned {scanned} Python files")
    print(f"Updated {changed} files")


if __name__ == "__main__":
    main()
