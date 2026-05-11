#!/usr/bin/env python3

'''
Usage:
    # Ollama (default - llama3.1:8b or qwen2.5-coder:7b, gemma4:e4b, deepseek-r1:8b, starcoder2:7b recommended for best results)
    python generate_class_eval_variants.py --provider ollama --model deepseek-r1:7b-qwen-distill-q4_K_M --start-id 0 --total 3

    # Claude (process from ID 5, all remaining samples)
    python generate_class_eval_variants.py --provider claude --start-id 5

    # Gemini (process first 10 samples only)
    python generate_class_eval_variants.py --provider gemini --model gemini-2.0-flash --total 10
    
    # Groq (process from ID 0, 50 samples)
    python generate_class_eval_variants.py --provider groq --model meta-llama/llama-3.3-70b-instruct --start-id 0 --total 50
'''

import argparse
import pandas as pd
import re
import sys
import unittest
import io
import textwrap
from pathlib import Path
from typing import List, Dict, Tuple

from llm_providers import LLMProviderFactory

DATASET_PATH = Path(__file__).parent.parent / "dataset"
OUTPUT_PATH = Path(__file__).parent.parent / "output"
OUTPUT_PATH.mkdir(exist_ok=True)


# NEW IMPLEMENTATION FOR CLASSIFICATION-ENABLED VARIANT GENERATION (no test feedback loop, just generate + classify in one pass)
def create_vs_generation_prompt(problem_statement: str, class_skeleton: str, class_name: str) -> str:
    return f"""<instructions>
You are generating a probability-weighted distribution of Python solutions to solve the given problem.

Given the problem description and class skeleton below, generate 5 independent Python solutions that implement the class.
Each solution must be correct and executable. Explore different algorithmic approaches - consider different implementations, data structures, etc.

For each solution, assign a probability (0.0 to 1.0) representing how likely this
approach would appear across the full distribution of valid solutions to this problem.
Probabilities do not need to sum to 1.0 across your 5 samples.

Output each variant within <response> tags containing:
- <probability>: float between 0.0 and 1.0
- <code>: complete, executable Python code only - implement the class based on the skeleton

Do not explain. Do not add comments describing what changed. Do not add any main function or test cases.
Output only the 5 <response> blocks.
</instructions>

Problem Description:
```python
{problem_statement}
```

Class Skeleton (implement this):
```python
{class_skeleton}
```

Generate 5 probability-weighted independent solutions that solve this problem:"""


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


def parse_variants(response: str) -> list[dict]:
    variants = []
    for block in re.findall(r'<response>(.*?)</response>', response, re.DOTALL):
        prob_match = re.search(r'<probability>(.*?)</probability>', block, re.DOTALL)
        code_match = re.search(r'<code>(.*?)</code>', block, re.DOTALL)
        
        if prob_match and code_match:
            code = code_match.group(1).strip()
        else:
            code_match = re.search(r'```(?:python)?\n(.*?)```', block, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                continue
        
        if prob_match:
            probability = float(prob_match.group(1).strip())
        else:
            probability = 0.0
        
        if code:
            variants.append({
                'probability': probability,
                'code': code
            })
    return variants


def generate_labeled_variants(code: str, llm_call) -> list[dict]:
    # Step 1: Generate diverse variants with VS
    gen_prompt = create_vs_generation_prompt(code)
    # print("Generating variants with prompt:\n", gen_prompt)
    gen_response = llm_call(gen_prompt)
    variants = parse_variants(gen_response)
    
    # Step 2: Classify each variant independently
    for variant in variants:
        cls_prompt = create_classification_prompt(code, variant['code'])
        variant['labels'] = llm_call(cls_prompt).strip()
    
    return variants


# OLD IMPELENTATION BELOW (for classEval dataset with test cases and pass/fail feedback loop)
def extract_variants_from_response(response_text: str) -> List[Dict[str, str]]:
    variants = []

    # Try XML <response>/<label>/<code> format first
    response_blocks = re.findall(r'<response>(.*?)</response>', response_text, re.DOTALL)
    for block in response_blocks:
        label_match = re.search(r'<label>(.*?)</label>', block, re.DOTALL)
        code_match = re.search(r'<code>(.*?)</code>', block, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            if code.startswith('```'):
                code = code[code.find('\n')+1:]
            if code.endswith('```'):
                code = code[:code.rfind('```')]
            code = code.strip()
            label = label_match.group(1).strip() if label_match else "unknown"
            if code:
                variants.append({'code': code, 'label': label})

    if variants:
        return variants[:5]

    # Fallback: parse numbered markdown format produced by most local models
    # Two sub-formats observed:
    #   "1. **label:** foo; bar\n```python\ncode\n```"  (label after bold+colon)
    #   "1. **early return; nested conditional**\n```python\ncode\n```"  (label IS the bold text)
    blocks = re.split(r'\n(?=\d+\.\s)', '\n' + response_text)
    for block in blocks:
        if not block.strip():
            continue
        label = "unknown"
        # Unified bold label parser covering two formats:
        #   Format A: "**label:** foo; bar"  or  "**Label**: foo; bar"  → text after bold is label
        #   Format B: "**early return; nested conditional**"            → text inside bold is label
        bold_m = re.search(r'\*\*([^*]+)\*\*\s*:?\s*([^\n]*)', block)
        if bold_m:
            inside_bold = bold_m.group(1).strip().rstrip(':., ')
            after_bold = bold_m.group(2).strip().rstrip('*:., ')
            # Use after-bold text only when it's present and not a code fence
            if after_bold and not after_bold.startswith('`'):
                label = after_bold
            else:
                label = inside_bold
        code_match = re.search(r'```(?:python)?\n(.*?)```', block, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            if code:
                variants.append({'code': code, 'label': label})

    return variants[:5]


def generate_variants(problem_statement: str, class_skeleton: str, class_name: str,
                      code: str, provider, test_code: str, task_id: str,
                      max_attempts: int = 5, model_name: str = "unknown",
                      verbose: bool = False, dataset_name: str = "classEval",
                      tracking_tag: str = None) -> List[Dict]:
    """
    Two-step VS pipeline:
      Step 1 – one LLM call to generate 5 probability-weighted code variants.
      Step 2 – one LLM call per variant to classify it independently.
    No test-feedback loop; test results are still recorded for analysis.
    """

    def vlog(msg: str):
        if verbose:
            print(f"  [VERBOSE] {msg}", flush=True)

    def llm_call(prompt: str) -> str:
        response = provider.generate(prompt, max_tokens=8192, temperature=0.8)
        return response.text

    # ── Step 1: Generate variants ──────────────────────────────────────────
    gen_prompt = create_vs_generation_prompt(problem_statement, class_skeleton, class_name)
    # print("Generating variants with prompt:\n", gen_prompt)
    raw_variants: list[dict] = []

    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    tracking_suffix = f"_{tracking_tag}" if tracking_tag else ""
    raw_responses_dir = OUTPUT_PATH / "raw_llm_responses" / f"{dataset_name}_{safe_model_name}{tracking_suffix}"
    raw_responses_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_attempts):
        vlog(f"Generation attempt {attempt+1}/{max_attempts}")
        try:
            raw = llm_call(gen_prompt)
            vlog(f"Response received ({len(raw)} chars)")
            if verbose:
                print(f"  [VERBOSE] Raw response:\n{'─'*60}\n{raw}\n{'─'*60}", flush=True)

            response_file = raw_responses_dir / f"{task_id}_gen_attempt{attempt+1}.txt"
            response_file.write_text(raw)

            new_variants = parse_variants(raw)
            vlog(f"Parsed {len(new_variants)} variants")
            raw_variants.extend(new_variants)

            if len(raw_variants) >= 5:
                break

        except Exception as e:
            print(f"  Generation attempt {attempt+1} failed: {e}", file=sys.stderr)
            continue

    raw_variants = raw_variants[:5]

    # ── Step 2: Classify + test each variant ──────────────────────────────
    variants: List[Dict] = []

    for i, var in enumerate(raw_variants[:5]):
        variant_code = var['code']
        probability  = var.get('probability', 0.0)

        # Classification
        vlog(f"Classifying variant {i+1}")
        try:
            cls_prompt = create_classification_prompt(code, variant_code)
            cls_raw = llm_call(cls_prompt)
            labels = cls_raw.strip()
            vlog(f"  → Labels: {labels}")

            cls_response_file = raw_responses_dir / f"{task_id}_variant{i+1}_classification.txt"
            cls_response_file.write_text(f"{variant_code}\n\n# Classification response:\n# {cls_raw.replace(chr(10), chr(10) + '# ')}")
        except Exception as e:
            labels = "unknown"
            vlog(f"  → Classification failed: {e}")

        # Test execution (recorded but does NOT gate inclusion)
        passed, msg = test_variant(variant_code, test_code, task_id, class_name)
        vlog(f"  → Test {'PASSED' if passed else 'FAILED'}: {msg}")

        variants.append({
            'code':        variant_code,
            'label':       labels,
            'probability': probability,
            'test_passed': passed,
            'test_msg':    msg,
        })

    # Pad with original if model returned fewer than 5
    while len(variants) < 5:
        vlog("Padding with original code (model returned fewer than 5 variants)")
        variants.append({
            'code':        code,
            'label':       'original',
            'probability': 0.0,
            'test_passed': True,
            'test_msg':    'fallback',
        })

    return variants[:5]


def extract_class_name_from_test(test_code: str) -> str:
    matches = re.findall(r'(\w+)\s*\(\s*\)', test_code)
    for match in matches:
        if match not in ['assertTrue', 'assertFalse', 'assertEqual', 'assertIn', 'unittest']:
            return match
    return None


def ensure_class_name_in_code(variant_code: str, expected_class_name: str) -> str:
    class_match = re.search(r'^\s*class\s+(\w+)', variant_code, re.MULTILINE)
    if class_match:
        actual_class_name = class_match.group(1)
        if actual_class_name != expected_class_name:
            variant_code = re.sub(
                r'^\s*class\s+' + re.escape(actual_class_name),
                f'class {expected_class_name}',
                variant_code,
                flags=re.MULTILINE
            )
    return variant_code


def test_variant(variant_code: str, test_code: str, task_id: str, class_name: str = None) -> Tuple[bool, str]:
    try:
        variant_code = textwrap.dedent(variant_code)
        test_code = textwrap.dedent(test_code)

        if class_name is None:
            class_name = extract_class_name_from_test(test_code)

        if class_name:
            variant_code = ensure_class_name_in_code(variant_code, class_name)

        full_code = variant_code + "\n\n" + test_code

        namespace = {}
        exec(full_code, namespace)

        loader = unittest.TestLoader()
        test_suite = unittest.TestSuite()
        for name, obj in namespace.items():
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:
                suite = loader.loadTestsFromTestCase(obj)
                test_suite.addTests(suite)

        runner = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0)
        result = runner.run(test_suite)

        passed = result.wasSuccessful()
        message = f"Tests: {result.testsRun}, Failures: {len(result.failures)}, Errors: {len(result.errors)}"

        return passed, message
    except Exception as e:
        return False, f"Exception: {str(e)[:100]}"


def process_class_eval_dataset(provider, model_name: str,
                                output_path: Path = None, verbose: bool = False,
                                start_id: int = None, total: int = None,
                                dataset_name: str = "classEval",
                                tracking_tag: str = None) -> None:
    if output_path is None:
        output_path = OUTPUT_PATH

    output_path.mkdir(parents=True, exist_ok=True)

    class_eval = pd.read_csv(DATASET_PATH / f"{dataset_name}.csv")
    code_results = []
    label_results = []

    total_rows = len(class_eval)

    if start_id is not None:
        class_eval = class_eval[class_eval['task_id'] >= start_id]
        range_info = f" from ID {start_id}"
    else:
        range_info = " from start"

    if total is not None:
        class_eval = class_eval.iloc[:total]
        range_info += f", total {total} samples"
    else:
        range_info += f", all remaining ({len(class_eval)} samples)"

    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    provider_name = provider.__class__.__name__.replace("Provider", "").lower()
    tracking_suffix = f"_{tracking_tag}" if tracking_tag else ""
    file_prefix = f"{dataset_name}_{provider_name}_{safe_model_name}{tracking_suffix}"

    print(f"\n{'='*70}")
    print(f"Processing {dataset_name}.csv with provider: {provider_name}{range_info}")
    print(f"Total dataset rows: {total_rows}")
    print(f"Output stored at: {output_path}")
    print(f"Output file prefix: {file_prefix}")
    if tracking_tag:
        print(f"Tracking tag: {tracking_tag}")
    if verbose:
        print(f"Verbose mode: ON")
    print(f"{'='*70}\n")

    processed = 0
    for idx, row in class_eval.iterrows():
        task_id = row['task_id']

        if 'solution_code' not in row.index:
            continue

        base_code = str(row['solution_code'])
        test_code = str(row['test']).strip() if 'test' in row.index else ""
        skeleton = str(row['skeleton']).strip() if 'skeleton' in row.index and pd.notna(row.get('skeleton')) else ""
        class_desc = str(row['class_description']).strip() if 'class_description' in row.index and pd.notna(row.get('class_description')) else ""

        if not base_code or base_code == 'nan':
            continue

        problem_statement = class_desc if class_desc and class_desc != 'nan' else ""
        class_skeleton = skeleton if skeleton and skeleton != 'nan' else ""

        processed += 1
        print(f"[{processed}/{len(class_eval)}] Processing task {task_id}...")

        try:
            class_name = str(row['class_name']).strip() if 'class_name' in row.index else None

            variants = generate_variants(
                problem_statement, class_skeleton, class_name,
                base_code, provider, test_code, task_id,
                model_name=model_name, verbose=verbose,
                dataset_name=dataset_name, tracking_tag=tracking_tag
            )

            base_passed, base_msg = test_variant(base_code, test_code, task_id, class_name)

            code_row = {
                'task_id': task_id,
                'base_code': base_code,
                'base_code_test_passed': base_passed,
                'base_code_test_msg': base_msg
            }

            label_row = {
                'task_id': task_id,
                'base_code_label': 'original'
            }

            passed_count = 0
            for i, variant_data in enumerate(variants):
                code_row[f'variant_{i+1}']             = variant_data['code']
                code_row[f'variant_{i+1}_test_passed'] = variant_data['test_passed']
                code_row[f'variant_{i+1}_test_msg']    = variant_data['test_msg']
                label_row[f'variant_{i+1}_label']       = variant_data['label']
                label_row[f'variant_{i+1}_probability'] = variant_data.get('probability', 0.0)
                if variant_data['test_passed']:
                    passed_count += 1

            code_results.append(code_row)
            label_results.append(label_row)
            print(f"  ✓ Generated {len(variants)} variants, {passed_count}/{len(variants)} passed tests")

        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
            continue

    code_df = pd.DataFrame(code_results)
    code_file = output_path / f"variants_code_{file_prefix}.csv"
    code_df.to_csv(code_file, index=False)

    label_df = pd.DataFrame(label_results)
    label_file = output_path / f"variants_labels_{file_prefix}.csv"
    label_df.to_csv(label_file, index=False)

    print(f"\n{'='*70}")
    print(f"✓ Saved {len(code_df)} tasks")
    print(f"  Code: {code_file}")
    print(f"  Labels: {label_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate classEval code variants")
    parser.add_argument("--provider", default="ollama",
                        help="LLM provider (ollama, claude, gemini, groq, cerebras, aisuite)")
    parser.add_argument("--model", default="llama3",
                        help="Model name (for ollama) or provider-specific model id")
    parser.add_argument("--start-id", type=int, default=None,
                         help="Starting task ID number (default: process from start)")
    parser.add_argument("--total", type=int, default=None,
                         help="Total number of samples to process (default: all)")
    parser.add_argument("--dataset-name", type=str, default="classEval",
                         help="Dataset CSV filename without extension (default: classEval)")
    parser.add_argument("--tracking-tag", type=str, default=None,
                         help="Tracking tag to append to output folder name (e.g., 'one' or 'two')")
    parser.add_argument("output_path", nargs="?", default=None,
                         help="Output directory path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed per-step logs (requests, responses, test results)")
    args = parser.parse_args()

    provider = LLMProviderFactory.create(args.provider, model=args.model)
    print(f"Using provider: {args.provider}, model: {args.model}")

    out = Path(args.output_path) if args.output_path else OUTPUT_PATH
    process_class_eval_dataset(provider, args.model, out, verbose=args.verbose,
                               start_id=args.start_id, total=args.total,
                               dataset_name=args.dataset_name, tracking_tag=args.tracking_tag)