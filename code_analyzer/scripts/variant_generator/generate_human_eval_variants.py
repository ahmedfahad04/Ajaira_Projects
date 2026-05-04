#!/usr/bin/env python3

import argparse
import pandas as pd
import ollama
import re
import sys
import io
import textwrap
from pathlib import Path
from typing import List, Dict, Tuple

DATASET_PATH = Path(__file__).parent.parent / "dataset"
OUTPUT_PATH = Path(__file__).parent.parent / "output"
OUTPUT_PATH.mkdir(exist_ok=True)


def create_refactoring_prompt(code: str) -> str:
    return f"""<instructions>
Generate 5 independent semantic-preserving refactored versions of the given code. Each variant should:
- Be valid, executable code that passes ALL test cases
- Keep the underlying problem isomorphic
- Vary the solving paradigm
- Avoid cosmetic changes

For each variant, output within <response> tags:
1. <label>: Semicolon-separated refactoring method names
2. <code>: The refactored code

LABEL TAXONOMY (prefer existing labels):

Naming: variable renaming, function renaming, method renaming, class renaming, parameter renaming, constant renaming, string formatting change, literal rewrite, import reorganization, formatting only

Control Flow: early return, guard clause, nested conditional, flattened conditional, combined condition, split condition, reversed condition, conditional expression, loop rewrite, for loop, while loop, enumerate loop, index-based loop, recursion, helper function extraction, helper function inlining, exception handling, try-except wrapper, branch reordering, sentinel flag

Data Structure: accumulator list, accumulator string, accumulator dict, accumulator set, temporary variable, state variable, tuple unpacking, dictionary lookup, set conversion, list conversion, generator expression, list comprehension, dict comprehension, set comprehension, slicing, in-place mutation, copy before mutation

Built-ins/Libraries: builtin function, standard library helper, itertools usage, functools reduce, map/filter usage, lambda expression, sort key, regex usage, membership test, any/all usage, zip usage, join/split usage, f-string

Algorithmic: brute force, two pointer, sliding window, stack-based, queue-based, heap-based, greedy strategy, dynamic programming, memoization, recursion with base case, divide and conquer, mathematical formula, sorting-based approach, frequency counting, prefix/suffix computation, graph traversal, tree traversal, binary search

Robustness: input validation, empty input handling, boundary case handling, type conversion, type check, default value handling, error suppression, behavior change

LABEL RULES:
- Pick labels at highest useful granularity
- Use semicolons to separate multiple labels (e.g., "list comprehension; enumerate loop")
- For approaches NOT in the taxonomy: write a new lowercase label (2-4 words, no verbs like "uses")
- Prefer existing labels; only invent new ones when nothing fits
</instructions>

Original Code:
```
{code}
```

Generate 5 independent semantic-preserving refactored variants:
"""


def extract_variants_from_response(response_text: str) -> List[Dict]:
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


def generate_variants(code: str, model: str, test_code: str, method_name: str,
                     prompt_str: str = None, max_attempts: int = 5,
                     verbose: bool = False) -> List[Dict]:
    prompt = create_refactoring_prompt(code)
    variants = []

    def vlog(msg: str):
        if verbose:
            print(f"  [VERBOSE] {msg}", flush=True)

    for attempt in range(max_attempts):
        vlog(f"Sending request to {model} (attempt {attempt+1}/{max_attempts})")
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            raw = response.message.content
            vlog(f"Response received ({len(raw)} chars)")
            if verbose:
                print(f"  [VERBOSE] Raw response:\n{'─'*60}\n{raw}\n{'─'*60}", flush=True)

            raw_variants = extract_variants_from_response(raw)
            vlog(f"Parsed {len(raw_variants)} variants from response")

            for i, var in enumerate(raw_variants):
                if len(variants) >= 5:
                    break

                variant_code = var['code']
                label = var['label']

                vlog(f"Testing variant {i+1}: [{label}]")
                passed, msg = test_variant(variant_code, test_code, method_name, prompt_str)
                vlog(f"  → {'PASSED' if passed else 'FAILED'}: {msg}")

                if passed:
                    variants.append({
                        'code': variant_code,
                        'label': label,
                        'test_passed': True,
                        'test_msg': msg
                    })
                else:
                    retry_success = False
                    for retry in range(3):
                        vlog(f"  Retry {retry+1}/3 for variant {i+1} (error: {msg[:80]})")
                        retry_prompt = f"""{prompt}

The previous variant failed the test with error: {msg}
Please generate a corrected variant that passes all test cases.

Output format:
<response>
<label>refactoring method names</label>
<code>
corrected code here
</code>
</response>"""
                        try:
                            retry_response = ollama.chat(
                                model=model,
                                messages=[{"role": "user", "content": retry_prompt}],
                                stream=False
                            )
                            retry_raw = retry_response.message.content
                            vlog(f"  Retry response ({len(retry_raw)} chars)")
                            if verbose:
                                print(f"  [VERBOSE] Retry raw:\n{'─'*40}\n{retry_raw[:500]}\n{'─'*40}", flush=True)

                            retry_vars = extract_variants_from_response(retry_raw)
                            if retry_vars:
                                retry_code = retry_vars[0]['code']
                                retry_label = retry_vars[0]['label']
                                passed, msg = test_variant(retry_code, test_code, method_name, prompt_str)
                                vlog(f"  → Retry {'PASSED' if passed else 'FAILED'}: {msg}")
                                if passed:
                                    variants.append({
                                        'code': retry_code,
                                        'label': retry_label,
                                        'test_passed': True,
                                        'test_msg': msg
                                    })
                                    retry_success = True
                                    break
                        except Exception:
                            continue

                    if not retry_success:
                        variants.append({
                            'code': variant_code,
                            'label': label,
                            'test_passed': False,
                            'test_msg': msg
                        })

        except Exception as e:
            print(f"  Generation attempt {attempt+1} failed: {e}", file=sys.stderr)
            continue

        if len(variants) >= 5:
            break

    # Pad with original code if needed
    while len(variants) < 5:
        variants.append({
            'code': code,
            'label': 'original',
            'test_passed': True,
            'test_msg': 'fallback'
        })

    return variants[:5]


def extract_function_name_from_code(code: str) -> str:
    match = re.search(r'^\s*def\s+(\w+)\s*\(', code, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def ensure_function_name_in_code(variant_code: str, expected_func_name: str) -> str:
    actual_func_name = extract_function_name_from_code(variant_code)
    if actual_func_name and actual_func_name != expected_func_name:
        variant_code = re.sub(
            r'^(\s*def\s+)' + re.escape(actual_func_name) + r'(\s*\()',
            r'\1' + expected_func_name + r'\2',
            variant_code,
            flags=re.MULTILINE
        )
    return variant_code


def extract_function_signature_and_body(prompt: str) -> Tuple[str, str]:
    lines = prompt.split('\n')
    imports = []
    sig_start = -1

    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            imports.append(line)
        elif line.startswith('def '):
            sig_start = i
            break

    if sig_start >= 0:
        sig_line = lines[sig_start]
        if sig_line.rstrip().endswith(':'):
            sig_line = sig_line.rstrip()[:-1]
        imports_str = '\n'.join(imports)
        return imports_str, sig_line

    return '', ''


def test_variant(variant_code: str, test_code: str, method_name: str, prompt: str = None) -> Tuple[bool, str]:
    try:
        variant_code = textwrap.dedent(variant_code)
        test_code = textwrap.dedent(test_code)

        if prompt:
            imports_str, signature = extract_function_signature_and_body(prompt)
            body_lines = variant_code.split('\n')
            cleaned_body = []
            in_docstring = False
            for line in body_lines:
                if '"""' in line or "'''" in line:
                    in_docstring = not in_docstring
                    continue
                if not in_docstring:
                    cleaned_body.append(line)

            code_lines = [line for line in cleaned_body if line.strip()]

            if code_lines:
                body_text = '\n'.join(code_lines)
                body_text = textwrap.dedent(body_text)
                full_code = imports_str + '\n' + signature + ':\n'
                for line in body_text.split('\n'):
                    if line.strip():
                        full_code += '    ' + line + '\n'
            else:
                full_code = imports_str + '\n' + signature + ':\n    pass'

            variant_code = full_code
        else:
            if not variant_code.strip().startswith('def '):
                body_lines = variant_code.split('\n')
                variant_code = f"def {method_name}(*args, **kwargs):\n"
                for line in body_lines:
                    if line.strip():
                        variant_code += "    " + line + "\n"

        if 'def ' in variant_code:
            variant_code = ensure_function_name_in_code(variant_code, method_name)

        namespace = {}
        exec(variant_code, namespace)

        if method_name not in namespace:
            return False, f"Function {method_name} not found in variant"

        candidate = namespace[method_name]

        test_namespace = {'candidate': candidate}
        exec(test_code, test_namespace)

        if 'check' not in test_namespace:
            return False, "No check function found in test"

        try:
            test_namespace['check'](candidate)
            return True, "All assertions passed"
        except AssertionError as e:
            return False, f"Assertion failed: {str(e)[:100]}"
        except Exception as e:
            return False, f"Test error: {str(e)[:100]}"
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)[:100]}"
    except Exception as e:
        return False, f"Execution error: {str(e)[:100]}"


def process_human_eval_dataset(model: str, sample_size: str = "full",
                               output_path: Path = None, verbose: bool = False):
    if output_path is None:
        output_path = OUTPUT_PATH

    output_path.mkdir(parents=True, exist_ok=True)

    human_eval = pd.read_csv(DATASET_PATH / "human_eval_164.csv")
    code_results = []
    label_results = []

    total = len(human_eval)
    valid_codes = human_eval.dropna(subset=['solution_code'])

    if sample_size != "full":
        try:
            sample_count = int(sample_size)
            valid_codes = valid_codes.iloc[:sample_count]
            sample_info = f" (sampling {sample_count} items)"
        except ValueError:
            sample_info = ""
    else:
        sample_info = " (full dataset)"

    print(f"\n{'='*70}")
    print(f"Processing human_eval_164.csv with model: {model}{sample_info}")
    print(f"Total rows: {total}, Valid codes: {len(valid_codes)}")
    if verbose:
        print(f"Verbose mode: ON")
    print(f"{'='*70}\n")

    for idx, row in valid_codes.iterrows():
        task_id = row['task_id']
        base_code = str(row['solution_code'])
        test_code = str(row['test']).strip() if 'test' in row.index else ""
        method_name = str(row['method_name']).strip() if 'method_name' in row.index else "candidate"
        prompt_str = str(row['prompt']).strip() if 'prompt' in row.index else None

        if not base_code or base_code == 'nan':
            continue

        print(f"[{idx+1}/{len(valid_codes)}] Processing task {task_id}...")

        try:
            variants = generate_variants(
                base_code, model, test_code, method_name, prompt_str,
                verbose=verbose
            )

            base_passed, base_msg = test_variant(base_code, test_code, method_name, prompt_str)

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
                code_row[f'variant_{i+1}'] = variant_data['code']
                code_row[f'variant_{i+1}_test_passed'] = variant_data['test_passed']
                code_row[f'variant_{i+1}_test_msg'] = variant_data['test_msg']
                label_row[f'variant_{i+1}_label'] = variant_data['label']
                if variant_data['test_passed']:
                    passed_count += 1

            code_results.append(code_row)
            label_results.append(label_row)
            print(f"  ✓ Generated {len(variants)} variants, {passed_count}/{len(variants)} passed tests")

        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
            continue

    model_name = model.replace(':', '_').replace('.', '_')

    code_df = pd.DataFrame(code_results)
    code_file = output_path / f"human_eval_variants_code_{model_name}.csv"
    code_df.to_csv(code_file, index=False)

    label_df = pd.DataFrame(label_results)
    label_file = output_path / f"human_eval_variants_labels_{model_name}.csv"
    label_df.to_csv(label_file, index=False)

    print(f"\n{'='*70}")
    print(f"✓ Saved {len(code_df)} tasks")
    print(f"  Code: {code_file}")
    print(f"  Labels: {label_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate human_eval code variants")
    parser.add_argument("model", help="Ollama model name (e.g. qwen2.5-coder:7b)")
    parser.add_argument("sample_size", nargs="?", default="full",
                        help="Number of samples or 'full' (default: full)")
    parser.add_argument("output_path", nargs="?", default=None,
                        help="Output directory path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed per-step logs (requests, responses, test results)")
    args = parser.parse_args()

    out = Path(args.output_path) if args.output_path else OUTPUT_PATH
    process_human_eval_dataset(args.model, args.sample_size, out, verbose=args.verbose)
