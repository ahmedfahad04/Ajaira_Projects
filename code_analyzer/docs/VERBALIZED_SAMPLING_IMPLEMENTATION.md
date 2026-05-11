---
name: Verbalized Sampling Implementation
description: Summary of changes to generate labeled refactored variants with retry logic and dual CSV output
version: 1.0
---

# Verbalized Sampling Implementation

## Overview

Updated both `generate_class_eval_variants.py` and `generate_human_eval_variants.py` to:
1. Generate refactored code variants with refactoring method labels
2. Implement retry logic (up to 3 retries per failed variant)
3. Save outputs in dual CSV format: one for code, one for labels

## Pipeline Architecture

### 1. Variant Generation with Labels

**Prompt Structure:**
- Embedded full label taxonomy directly in the prompt
- Instructions to use semicolon-separated labels
- Rules for creating new labels when needed (2-4 words, lowercase, no verbs)

**Output Format:**
```xml
<response>
<label>list comprehension; enumerate loop</label>
<code>
def solve(...):
    ...
</code>
</response>
```

### 2. Extraction & Parsing

`extract_variants_from_response()` now returns:
```python
[
    {
        'code': '...',
        'label': 'refactoring method names'
    },
    ...
]
```

### 3. Testing & Retry Logic

For each variant:
1. **Initial Test**: Run against test cases
2. **If Failed**: Retry up to 3 times with:
   - Original prompt + error message
   - Request for corrected variant
   - Parse new label and code from response
3. **Store Result**: Include `test_passed` flag and test message

Function signature:
```python
def generate_variants(code: str, model: str, test_code: str, task_id: str,
                     class_name: str = None, max_attempts: int = 5) -> List[Dict]
```

Returns:
```python
[
    {
        'code': '...',
        'label': '...',
        'test_passed': bool,
        'test_msg': str
    },
    ...
]
```

### 4. Dual CSV Output

**Code CSV** (`{dataset}_variants_code_{model}.csv`):
- `task_id`
- `base_code`
- `base_code_test_passed`
- `base_code_test_msg`
- `variant_1`, `variant_1_test_passed`, `variant_1_test_msg`
- `variant_2`, ... (up to 5 variants)

**Labels CSV** (`{dataset}_variants_labels_{model}.csv`):
- `task_id`
- `base_code_label`: "original"
- `variant_1_label`, `variant_2_label`, ... (up to 5 variants)

Both CSVs share the same `task_id` for easy joining.

## Updated Functions

### Class Eval (`generate_class_eval_variants.py`)

- `create_refactoring_prompt()` - New prompt with taxonomy
- `extract_variants_from_response()` - Extract both label and code
- `generate_variants()` - Full pipeline with retry logic
- `process_class_eval_dataset()` - Dual CSV output

### Human Eval (`generate_human_eval_variants.py`)

- Same updates as Class Eval
- `generate_variants()` accepts `prompt_str` parameter for function signature context

## Usage

```bash
# Full dataset
python generate_class_eval_variants.py qwen2.5-coder:7b

# Sample 10 tasks
python generate_class_eval_variants.py qwen2.5-coder:7b 10

# Custom output path
python generate_class_eval_variants.py qwen2.5-coder:7b full ../output/custom/path
```

Output files:
- `classEval_variants_code_qwen2_5-coder_7b.csv`
- `classEval_variants_labels_qwen2_5-coder_7b.csv`

## Key Design Decisions

1. **Embedded Taxonomy**: Avoids context fragmentation; models see full rules upfront
2. **Retry Logic**: Per-variant, not per-batch (better granularity and tracking)
3. **Dual CSV**: Separates code (large, slow to query) from labels (lightweight)
4. **Fallback Padding**: Original code used if generation/retries exhaust
5. **Test Messages**: Stored for analysis of why variants pass/fail

## Label Taxonomy

6 categories with 8-30 labels each:
- **Naming**: 10 labels (variable renaming, function renaming, etc.)
- **Control Flow**: 20 labels (early return, guard clause, loops, recursion, etc.)
- **Data Structure**: 16 labels (accumulators, comprehensions, slicing, etc.)
- **Built-ins/Libraries**: 15 labels (map/filter, itertools, regex, etc.)
- **Algorithmic**: 17 labels (two pointer, DP, tree traversal, etc.)
- **Robustness**: 8 labels (validation, type checks, error handling, etc.)

Total: ~8 labels per variant on average (ranges from 1-4 typically)
