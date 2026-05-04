import pandas as pd
from pathlib import Path
import re
import argparse

SCRIPT_DIR = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser(description="Extract base_code and variant_1..5 into .py files per task_id")
    parser.add_argument(
        "--csv",
        type=str,
        default="class_eval/classEval_variants_claude_default.csv",
        help="CSV path (absolute or relative to output/)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output root folder (absolute or relative to output/). If omitted, auto-derived from CSV name.",
    )
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


args = parse_args()
csv_path = resolve_path(args.csv)

if args.out:
    out_root = resolve_path(args.out)
else:
    stem = csv_path.stem
    if stem.startswith("classEval_variants_"):
        eval_type = "class_eval"
        model_name = stem.replace("classEval_variants_", "", 1)
    elif stem.startswith("human_eval_variants_"):
        eval_type = "human_eval"
        model_name = stem.replace("human_eval_variants_", "", 1)
    else:
        eval_type = "extracted"
        model_name = stem

    out_root = SCRIPT_DIR / "extracted_code" / f"{eval_type}_{model_name}"

if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found at: {csv_path}")

df = pd.read_csv(csv_path)

code_cols = ["base_code"] + [f"variant_{i}" for i in range(1, 6)]

def clean_code(text):
    if pd.isna(text):
        return ""
    s = str(text).strip()
    s = re.sub(r"^```(?:python)?\s*\n", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n```$", "", s)
    return s.strip() + "\n"

for _, row in df.iterrows():
    task_id = int(row["task_id"])
    task_dir = out_root / f"task_{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    for col in code_cols:
        code = clean_code(row.get(col, ""))
        filename = "base.py" if col == "base_code" else f"{col}.py"
        (task_dir / filename).write_text(code, encoding="utf-8")

print(f"Done. Extracted files to: {out_root}")