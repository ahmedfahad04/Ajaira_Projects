## Label Taxonomy

Use these labels before inventing new ones.

### Naming And Surface Changes

- variable renaming
- function renaming
- method renaming
- class renaming
- parameter renaming
- constant renaming
- string formatting change
- literal rewrite
- import reorganization
- formatting only

### Control Flow

- early return
- guard clause
- nested conditional
- flattened conditional
- combined condition
- split condition
- reversed condition
- conditional expression
- loop rewrite
- for loop
- while loop
- enumerate loop
- index-based loop
- recursion
- helper function extraction
- helper function inlining
- exception handling
- try-except wrapper
- branch reordering
- sentinel flag

### Data Structure And State

- accumulator list
- accumulator string
- accumulator dict
- accumulator set
- temporary variable
- state variable
- tuple unpacking
- dictionary lookup
- set conversion
- list conversion
- generator expression
- list comprehension
- dict comprehension
- set comprehension
- slicing
- in-place mutation
- copy before mutation

### Built-ins, Libraries, And Idioms

- builtin function
- standard library helper
- itertools usage
- functools reduce
- map/filter usage
- lambda expression
- sort key
- regex usage
- membership test
- any/all usage
- zip usage
- join/split usage
- f-string

### Algorithmic Strategy

- brute force
- two pointer
- sliding window
- stack-based
- queue-based
- heap-based
- greedy strategy
- dynamic programming
- memoization
- recursion with base case
- divide and conquer
- mathematical formula
- sorting-based approach
- frequency counting
- prefix/suffix computation
- graph traversal
- tree traversal
- binary search

### Behavioral Or Robustness Changes

- input validation
- empty input handling
- boundary case handling
- type conversion
- type check
- default value handling
- error suppression
- behavior change

## Label Selection Rules

Choose labels at highest useful level.

Good:

```text
list comprehension; enumerate loop
itertools usage; zip usage
recursion; helper function extraction
variable renaming; method renaming; f-string
```

Avoid:

```text
uses brackets; adds line; changes x to y; more Pythonic
```

If variant has broad renaming but same algorithm, label renaming plus any meaningful expression/control-flow change.

If variant changes from explicit loop to comprehension, use:

```text
list comprehension
```

Add `enumerate loop`, `zip usage`, or `conditional expression` only if central to mechanism.

If variant adds helper only to hold same logic, use:

```text
helper function extraction
```

If helper makes recursive decomposition, use:

```text
recursion; helper function extraction
```

If variant imports library to replace manual logic, use library label:

```text
standard library helper
```

or more specific:

```text
itertools usage; zip usage
```