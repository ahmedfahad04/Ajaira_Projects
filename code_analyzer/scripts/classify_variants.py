#!/usr/bin/env python3
"""
Classify generated code variants by comparing with base implementation.
Uses LLM to determine how each variant differs from the original.

Supported datasets: classEval, human_eval_164, rwpb
Supported providers: claude, gemini, nvidia, ollama, groq, etc.

Usage:
    # Using NVIDIA NIM (recommended - free endpoint)
    export NVIDIA_API_KEY="your-key-from-build.nvidia.com"
    python scripts/classify_variants.py --provider nvidia --model deepseek-ai/deepseek-v4-flash \\
        --input output/final/classEval_gpt-5-mini_generation_complete/classEval \\
        --dataset-name classEval --output output/labels_classEval.csv

    # Using other providers
    python scripts/classify_variants.py --provider ollama --model llama3.1 \\
        --input /path/to/variants --dataset-name classEval
"""

import argparse
import csv
import sys
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from variant_generator.llm_providers import LLMProviderFactory


def create_classification_prompt(original_code: str, variant_code: str) -> str:
    return f"""<instructions>
Analyze how this refactored code differs from the original. 
Assign labels describing the transformation techniques used.

LABEL TAXONOMY (prefer existing labels, invent new ones only when nothing fits):

Naming:
  variable renaming | function renaming | method renaming | class renaming
  parameter renaming | constant renaming | string formatting change
  literal rewrite | import reorganization | formatting only

Control Flow:
  early return | guard clause | nested conditional | flattened conditional
  combined condition | split condition | reversed condition | conditional expression
  loop rewrite | for loop | while loop | enumerate loop | index-based loop
  recursion | helper function extraction | helper function inlining
  exception handling | try-except wrapper | branch reordering | sentinel flag

Data Structure:
  accumulator list | accumulator string | accumulator dict | accumulator set
  temporary variable | state variable | tuple unpacking | dictionary lookup
  set conversion | list conversion | generator expression | list comprehension
  dict comprehension | set comprehension | slicing | in-place mutation | copy before mutation

Built-ins/Libraries:
  builtin function | standard library helper | itertools usage | functools reduce
  map/filter usage | lambda expression | sort key | regex usage
  membership test | any/all usage | zip usage | join/split usage | f-string

Algorithmic:
  brute force | two pointer | sliding window | stack-based | queue-based
  heap-based | greedy strategy | dynamic programming | memoization
  recursion with base case | divide and conquer | mathematical formula
  sorting-based approach | frequency counting | prefix/suffix computation
  graph traversal | tree traversal | binary search

Robustness:
  input validation | empty input handling | boundary case handling
  type conversion | type check | default value handling
  error suppression | behavior change

RULES:
- Pick labels at highest useful granularity
- Separate multiple labels with semicolons
- Invent new lowercase labels (2-4 words, no verbs) only when nothing fits
- Focus on WHAT changed algorithmically, not surface syntax
</instructions>

Original Code:
```python
{original_code}
```

Refactored Variant:
```python
{variant_code}
```

Output ONLY a single line of semicolon-separated labels. No explanation."""


def get_base_code(dataset_name: str, task_id: int, dataset_path: Path) -> Optional[str]:
    """Load base code from dataset CSV or JSON."""
    if dataset_name == "rwpb":
        return get_rwpb_base_code(task_id, dataset_path)
    
    csv_file = dataset_path / f"{dataset_name}.csv"
    
    if not csv_file.exists():
        return None
    
    try:
        df = pd.read_csv(csv_file)
        row = df[df['task_id'] == task_id]
        
        if row.empty:
            return None
        
        if 'solution_code' in row.columns:
            return str(row['solution_code'].iloc[0])
        return None
    except Exception as e:
        print(f"Error loading base code for task {task_id}: {e}", file=sys.stderr)
        return None


def get_rwpb_base_code(task_id: int, dataset_path: Path) -> Optional[str]:
    """Load base code from rwpb JSON."""
    json_path = dataset_path / "RWPB" / "rwpb.json"
    
    if not json_path.exists():
        return None
    
    try:
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            item_task_id = str(item.get('task_id', '')).split('/')[-1]
            try:
                if int(item_task_id) == task_id:
                    return item.get('canonical_solution', '') or item.get('solution', '')
            except ValueError:
                pass
        return None
    except Exception as e:
        print(f"Error loading rwpb base code for task {task_id}: {e}", file=sys.stderr)
        return None


def extract_task_id_and_variant(filename: str) -> tuple:
    """Extract task_id and variant number from filename like '0_variant1.py' or 'RWPB_3_variant5.py'."""
    name = Path(filename).stem
    parts = name.rsplit('_variant', 1)
    
    if len(parts) == 2:
        try:
            variant_num = int(parts[1])
            task_id_str = parts[0]
            if task_id_str.startswith('RWPB_'):
                task_id = int(task_id_str.split('_')[1])
            else:
                task_id = int(task_id_str)
            return task_id, variant_num
        except (ValueError, IndexError):
            pass
    
    return None, None


def _save_intermediate_results(results: list, output_csv: Path, dataset_name: str):
    """Save intermediate results incrementally to prevent data loss."""
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    if df.empty:
        return
    
    pivot_df = df.pivot(index='task_id', columns='variant', values='labels')
    pivot_df.columns = [f'variant_{col}_label' for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    
    for i in range(1, 6):
        if f'variant_{i}_label' not in pivot_df.columns:
            pivot_df[f'variant_{i}_label'] = ""
    
    variant_cols = sorted([c for c in pivot_df.columns if c.startswith('variant_')],
                          key=lambda x: int(x.split('_')[1]))
    final_cols = ['task_id'] + variant_cols
    pivot_df = pivot_df[[c for c in final_cols if c in pivot_df.columns]]
    
    pivot_df.to_csv(output_csv, index=False)


def classify_folder(
    folder_path: Path,
    provider,
    dataset_name: str = "classEval",
    dataset_path: Path = None,
    output_csv: Path = None,
    verbose: bool = False,
    limit: int = None,
    start: int = None
) -> pd.DataFrame:
    """Classify all variants in a folder."""
    
    if dataset_path is None:
        base_path = Path(__file__).parent.parent
        dataset_path = base_path / "dataset"
        csv_path = dataset_path / f"{dataset_name}.csv"
        if not csv_path.exists():
            alt_path = base_path / "scripts" / "dataset"
            if (alt_path / f"{dataset_name}.csv").exists() or (alt_path / "RWPB").exists():
                dataset_path = alt_path
    
    py_files = list(folder_path.glob("*.py"))
    
    if not py_files:
        print(f"No .py files found in {folder_path}", file=sys.stderr)
        return pd.DataFrame()
    
    task_to_files = {}
    for py_file in py_files:
        task_id, variant_num = extract_task_id_and_variant(py_file.name)
        if task_id is None:
            print(f"Skipping {py_file.name}: could not extract task_id", file=sys.stderr)
            continue
        if task_id not in task_to_files:
            task_to_files[task_id] = []
        task_to_files[task_id].append((variant_num, py_file))
    
    sorted_task_ids = sorted(task_to_files.keys())
    
    if output_csv:
        empty_df = pd.DataFrame(columns=['task_id'] + [f'variant_{i}_label' for i in range(1, 6)])
        empty_df.to_csv(output_csv, index=False)
        print(f"Initialized output file: {output_csv}")
    
    if start is not None:
        sorted_task_ids = [tid for tid in sorted_task_ids if tid >= start]
    
    if limit:
        sorted_task_ids = sorted_task_ids[:limit]
    
    print(f"Total tasks found: {len(task_to_files)}, processing: {len(sorted_task_ids)}")
    print(f"Task IDs to process (sequential): {sorted_task_ids}")
    
    results = []
    
    for task_id in sorted_task_ids:
        files = sorted(task_to_files[task_id], key=lambda x: x[0])
        
        base_code = get_base_code(dataset_name, task_id, dataset_path)
        
        if not base_code or base_code == 'nan':
            print(f"Warning: No base code found for task {task_id}", file=sys.stderr)
            for variant_num, _ in files:
                results.append({
                    'task_id': task_id,
                    'variant': variant_num,
                    'labels': "no_base_code"
                })
            continue
        
        for variant_num, py_file in files:
            if verbose:
                print(f"Classifying task {task_id} variant {variant_num}...")
            
            try:
                variant_code = py_file.read_text()
                prompt = create_classification_prompt(base_code, variant_code)
                response = provider.generate(prompt, max_tokens=1024, temperature=0.1)
                labels = response.text.strip()
                print(f"  -> Task {task_id} variant {variant_num}: {labels[:50]}...")
            except Exception as e:
                print(f"Error classifying {py_file.name}: {e}", file=sys.stderr)
                labels = "error"
            
            results.append({
                'task_id': task_id,
                'variant': variant_num,
                'labels': labels
            })

        if output_csv:
            _save_intermediate_results(results, output_csv, dataset_name)
        print("\n --- \n")
    
    df = pd.DataFrame(results)
    
    if df.empty:
        return df
    
    pivot_df = df.pivot(index='task_id', columns='variant', values='labels')
    pivot_df.columns = [f'variant_{col}_label' for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    
    for i in range(1, 6):
        if f'variant_{i}_label' not in pivot_df.columns:
            pivot_df[f'variant_{i}_label'] = ""
    
    variant_cols = sorted([c for c in pivot_df.columns if c.startswith('variant_')],
                          key=lambda x: int(x.split('_')[1]))
    final_cols = ['task_id'] + variant_cols
    pivot_df = pivot_df[[c for c in final_cols if c in pivot_df.columns]]
    
    if output_csv:
        pivot_df.to_csv(output_csv, index=False)
        print(f"Saved classification results to {output_csv}")
    
    return pivot_df


def classify_from_zip(
    zip_path: Path,
    provider,
    dataset_name: str = "classEval",
    dataset_path: Path = None,
    verbose: bool = False,
    limit: int = None,
    start: int = None
) -> pd.DataFrame:
    """Extract and classify from a zip file."""
    
    if dataset_path is None:
        base_path = Path(__file__).parent.parent
        dataset_path = base_path / "dataset"
        csv_path = dataset_path / f"{dataset_name}.csv"
        if not csv_path.exists():
            alt_path = base_path / "scripts" / "dataset"
            if (alt_path / f"{dataset_name}.csv").exists() or (alt_path / "RWPB").exists():
                dataset_path = alt_path
    
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)
        
        folders = list(Path(tmpdir).iterdir())
        
        for folder in folders:
            if folder.is_dir():
                py_files = list(folder.glob("*.py"))
                if py_files:
                    output_csv = zip_path.parent / f"variants_labels_{folder.name}.csv"
                    return classify_folder(folder, provider, dataset_name, dataset_path, output_csv, verbose, limit, start)
    
    return pd.DataFrame()


def determine_dataset_name(folder_or_zip: Path) -> str:
    """Infer dataset name from folder/zip filename."""
    name = folder_or_zip.name.lower()
    
    if 'class_eval' in name or 'classeval' in name:
        return "classEval"
    elif 'human_eval' in name:
        return "human_eval_164"
    elif 'rwpb' in name:
        return "rwpb"
    
    return "classEval"


def main():
    parser = argparse.ArgumentParser(description="Classify code variants by comparing with base")
    parser.add_argument("--provider", default="claude",
                        help="LLM provider (claude, gemini, groq, cerebras, ollama, etc.)")
    parser.add_argument("--model", default=None,
                        help="Model name (provider-specific)")
    parser.add_argument("--input", required=True,
                        help="Input folder/zip with variant .py files")
    parser.add_argument("--input-folder", dest="input", # Alias for backward compatibility
                        help="Input folder with .py variant files (alias for --input)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (optional)")
    parser.add_argument("--dataset-name", default=None,
                        help="Dataset name (classEval, human_eval_164, rwpb)")
    parser.add_argument("--dataset-path", default=None, type=Path,
                        help="Path to dataset folder with CSVs")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed logs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of tasks to process (for testing)")
    parser.add_argument("--start", type=int, default=None,
                        help="Starting task ID to begin processing from")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    provider = LLMProviderFactory.create(args.provider, model=args.model)
    print(f"Using provider: {args.provider}, model: {args.model or 'default'}")
    
    dataset_name = args.dataset_name or determine_dataset_name(input_path)
    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    
    output_csv = Path(args.output) if args.output else None
    
    if input_path.is_file() and input_path.suffix == '.zip':
        df = classify_from_zip(input_path, provider, dataset_name, dataset_path, args.verbose, args.limit, args.start)
    else:
        df = classify_folder(input_path, provider, dataset_name, dataset_path, output_csv, args.verbose, args.limit, args.start)
    
    print(f"Classified {len(df)} tasks" if not df.empty else "No results")


if __name__ == "__main__":
    main()