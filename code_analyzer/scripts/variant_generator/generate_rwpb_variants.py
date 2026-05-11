#!/usr/bin/env python3

'''
Usage:
    python generate_rwpb_variants.py --provider ollama --model qwen2.5-coder:7b --start-id 0 --total 3 --verbose

    # Claude (process from ID 5, all remaining)
    python generate_rwpb_variants.py --provider claude --start-id 5

    # Gemini
    python generate_rwpb_variants.py --provider gemini --model gemini-2.0-flash --total 10
'''

import argparse
import json
import re
import sys
import io
import textwrap
from pathlib import Path
from typing import List, Dict, Tuple

from llm_providers import LLMProviderFactory

DATASET_PATH = Path(__file__).parent.parent / "dataset"
OUTPUT_PATH = Path(__file__).parent.parent / "output"
OUTPUT_PATH.mkdir(exist_ok=True)


def create_vs_generation_prompt(code: str) -> str:
    return f"""<instructions>
You are generating a probability-weighted distribution of Python solutions.

Given the original code below, generate 5 independent Python solutions to the same problem.
Each solution must be correct and executable. Think about the full space of ways this problem could be solved.

For each solution, assign a probability (0.0 to 1.0) representing how likely this 
approach would appear across the full distribution of valid solutions to this problem.
Probabilities do not need to sum to 1.0 across your 5 samples.

Output each variant within <response> tags containing:
- <probability>: float between 0.0 and 1.0
- <code>: complete, executable Python code only

Do not explain. Do not add comments describing what changed. Do not add any main function or test cases. 
Output only the 5 <response> blocks.
</instructions>

Original Code:
```python
{code}
```

Generate 5 probability-weighted independent solutions:"""


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


def extract_function_name_from_prompt(prompt: str) -> str:
    match = re.search(r'def\s+(\w+)\s*\(', prompt)
    if match:
        return match.group(1)
    return None


def extract_function_signature_from_prompt(prompt: str) -> str:
    lines = prompt.split('\n')
    sig_line = None
    for line in lines:
        if line.strip().startswith('def '):
            sig_line = line
            break
    if not sig_line:
        return None
    if sig_line.rstrip().endswith(':'):
        sig_line = sig_line.rstrip()[:-1]
    return sig_line


def extract_imports_from_prompt(prompt: str) -> List[str]:
    imports = []
    for line in prompt.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(stripped)
        elif stripped.startswith('def '):
            break
    return imports


def build_full_code(prompt: str, body: str) -> str:
    imports = extract_imports_from_prompt(prompt)
    imports_str = '\n'.join(imports)
    signature = extract_function_signature_from_prompt(prompt)
    
    if signature and body:
        body_dedent = textwrap.dedent(body)
        full_code = imports_str + '\n' + signature + ':\n'
        for line in body_dedent.split('\n'):
            if line.strip():
                full_code += '    ' + line + '\n'
        return full_code
    return body


def ensure_function_name(code: str, expected_name: str) -> str:
    match = re.search(r'def\s+(\w+)\s*\(', code)
    if match:
        actual_name = match.group(1)
        if actual_name != expected_name:
            code = re.sub(
                r'def\s+' + re.escape(actual_name) + r'\s*\(',
                f'def {expected_name}(',
                code
            )
    return code


def generate_variants(provider, prompt: str, canonical_body: str, test_cases: str, task_id: str,
                      max_attempts: int = 5, model_name: str = "unknown",
                      verbose: bool = False) -> List[Dict]:
    
    def vlog(msg: str):
        if verbose:
            print(f"  [VERBOSE] {msg}", flush=True)

    def llm_call(prompt_text: str) -> str:
        response = provider.generate(prompt_text, max_tokens=8192, temperature=0.7)
        return response.text

    func_name = extract_function_name_from_prompt(prompt)
    if not func_name:
        vlog("Could not extract function name from prompt")
        return []

    base_code = build_full_code(prompt, canonical_body)
    base_code = ensure_function_name(base_code, func_name)

    print("BASE CODE >> ", prompt, flush=True)

    gen_prompt = create_vs_generation_prompt(base_code)
    raw_variants: list[dict] = []

    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    raw_responses_dir = OUTPUT_PATH / "raw_llm_responses" / safe_model_name
    raw_responses_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_attempts):
        vlog(f"Generation attempt {attempt+1}/{max_attempts}")
        try:
            # print("PROMPT > ", gen_prompt)
            raw = llm_call(gen_prompt)
            vlog(f"Response received ({len(raw)} chars)")
            if verbose:
                print(f"  [VERBOSE] Raw response:\n{'─'*60}\n{raw}\n{'─'*60}", flush=True)

            response_file = raw_responses_dir / f"{task_id.replace('/', '_')}_gen_attempt{attempt+1}.txt"
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

    variants: List[Dict] = []

    for i, var in enumerate(raw_variants[:5]):
        variant_body = var['code']
        probability = var.get('probability', 0.0)

        variant_code = build_full_code(prompt, variant_body)
        variant_code = ensure_function_name(variant_code, func_name)

        vlog(f"Classifying variant {i+1}")
        try:
            cls_prompt = create_classification_prompt(base_code, variant_code)
            cls_raw = llm_call(cls_prompt)
            labels = cls_raw.strip()
            vlog(f"  → Labels: {labels}")

            cls_response_file = raw_responses_dir / f"{task_id.replace('/', '_')}_variant{i+1}_classification.py"
            cls_response_file.write_text(f"{variant_code}\n\n# Classification response:\n# {cls_raw.replace(chr(10), chr(10) + '# ')}")
        except Exception as e:
            labels = "unknown"
            vlog(f"  → Classification failed: {e}")

        passed, msg = test_variant(variant_code, test_cases, func_name)
        vlog(f"  → Test {'PASSED' if passed else 'FAILED'}: {msg}")

        variants.append({
            'code': variant_code,
            'label': labels,
            'probability': probability,
            'test_passed': passed,
            'test_msg': msg,
        })

    while len(variants) < 5:
        vlog("Padding with original code (model returned fewer than 5 variants)")
        variants.append({
            'code': base_code,
            'label': 'original',
            'probability': 0.0,
            'test_passed': True,
            'test_msg': 'fallback',
        })

    return variants[:5]


def test_variant(variant_code: str, test_cases: str, func_name: str) -> Tuple[bool, str]:
    try:
        variant_code = textwrap.dedent(variant_code)
        test_cases = test_cases.strip()

        test_cases_processed = test_cases.replace('SOLUTION_SIGNATURE', func_name)

        namespace = {}
        exec(variant_code, namespace)

        if func_name not in namespace:
            return False, f"Function {func_name} not found in variant"

        candidate = namespace[func_name]

        test_namespace = {'candidate': candidate}
        exec(test_cases_processed, test_namespace)

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


def process_rwpb_dataset(provider, model_name: str,
                          output_path: Path = None, verbose: bool = False,
                          start_id: int = None, total: int = None):
    if output_path is None:
        output_path = OUTPUT_PATH

    output_path.mkdir(parents=True, exist_ok=True)

    with open(DATASET_PATH / "RWPB" / "rwpb.json", 'r') as f:
        rwpb_data = json.load(f)

    code_results = []
    label_results = []

    total_rows = len(rwpb_data)

    def get_task_num(task_id: str) -> int:
        match = re.search(r'RWPB/(\d+)', task_id)
        return int(match.group(1)) if match else 0

    if start_id is not None:
        rwpb_data = [item for item in rwpb_data if get_task_num(item['task_id']) >= start_id]
        range_info = f" from ID {start_id}"
    else:
        range_info = " from start"

    if total is not None:
        rwpb_data = rwpb_data[:total]
        range_info += f", total {total} samples"
    else:
        range_info += f", all remaining ({len(rwpb_data)} samples)"

    provider_name = provider.__class__.__name__.replace("Provider", "")
    print(f"\n{'='*70}")
    print(f"Processing RWPB dataset with provider: {provider_name}{range_info}")
    print(f"Total dataset rows: {total_rows}")
    if verbose:
        print(f"Verbose mode: ON")
    print(f"{'='*70}\n")

    processed = 0
    for item in rwpb_data:
        task_id = item['task_id']
        prompt = item['prompt']
        canonical_body = item.get('canonical_solution', '')
        test_cases = item.get('unprocess_testcases', '')

        if not canonical_body or not canonical_body.strip():
            continue

        processed += 1
        print(f"[{processed}/{len(rwpb_data)}] Processing task {task_id}...")

        try:
            func_name = extract_function_name_from_prompt(prompt)
            if not func_name:
                print(f"  ✗ Could not extract function name from prompt", file=sys.stderr)
                continue

            base_code = build_full_code(prompt, canonical_body)
            base_code = ensure_function_name(base_code, func_name)

            variants = generate_variants(
                provider, prompt, canonical_body, test_cases, task_id,
                model_name=model_name, verbose=verbose
            )

            base_passed, base_msg = test_variant(base_code, test_cases, func_name)

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
                label_row[f'variant_{i+1}_probability'] = variant_data.get('probability', 0.0)
                if variant_data['test_passed']:
                    passed_count += 1

            code_results.append(code_row)
            label_results.append(label_row)
            print(f"  ✓ Generated {len(variants)} variants, {passed_count}/{len(variants)} passed tests")

        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
            continue

    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    provider_name = provider.__class__.__name__.replace("Provider", "").lower()
    file_prefix = f"{provider_name}_{safe_model_name}"

    import pandas as pd
    code_df = pd.DataFrame(code_results)
    code_file = output_path / f"rwpb_variants_code_{file_prefix}.csv"
    code_df.to_csv(code_file, index=False)

    label_df = pd.DataFrame(label_results)
    label_file = output_path / f"rwpb_variants_labels_{file_prefix}.csv"
    label_df.to_csv(label_file, index=False)

    print(f"\n{'='*70}")
    print(f"✓ Saved {len(code_df)} tasks")
    print(f"  Code: {code_file}")
    print(f"  Labels: {label_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RWPB code variants")
    parser.add_argument("--provider", default="ollama",
                        help="LLM provider (ollama, claude, gemini, groq, cerebras, aisuite)")
    parser.add_argument("--model", default="llama3",
                        help="Model name (for ollama) or provider-specific model id")
    parser.add_argument("--start-id", type=int, default=None,
                        help="Starting task ID number (default: process from start)")
    parser.add_argument("--total", type=int, default=None,
                        help="Total number of samples to process (default: all)")
    parser.add_argument("output_path", nargs="?", default=None,
                        help="Output directory path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed per-step logs (requests, responses, test results)")
    args = parser.parse_args()

    provider = LLMProviderFactory.create(args.provider, model=args.model)
    print(f"Using provider: {args.provider}, model: {args.model}")

    out = Path(args.output_path) if args.output_path else OUTPUT_PATH
    process_rwpb_dataset(provider, args.model, out, verbose=args.verbose,
                         start_id=args.start_id, total=args.total)
