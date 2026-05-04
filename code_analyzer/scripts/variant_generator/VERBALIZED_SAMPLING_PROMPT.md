---
name: Verbalized Sampling Prompt
description: Prompt template for generating semantic-preserving refactored code variants with labeled refactoring methods
version: 1.0
---

# Verbalized Sampling Prompt Template

## Usage

Replace `{code}` with the original code snippet. The model will output 5 variants, each with a label and code block.

## Prompt

```
<instructions>
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
\`\`\`
{code}
\`\`\`

Generate 5 independent semantic-preserving refactored variants:
```

## Output Format Example

```xml
<response>
<label>list comprehension; enumerate loop</label>
<code>
def solve(arr):
    return [val * 2 for i, val in enumerate(arr)]
</code>
</response>
```

## Notes

- Each label section should be concise but descriptive
- Multiple labels are separated by semicolons with no extra spaces
- The prompt is designed to fit in typical token budgets while maintaining output quality
- Context usage: ~650 tokens for the prompt, leaving room for code generation
