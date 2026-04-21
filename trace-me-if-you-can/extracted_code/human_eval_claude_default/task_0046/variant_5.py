# Version 5: Functional programming with reduce
from functools import reduce

def compute_next_state(state, _):
    return (state[1], state[2], state[3], sum(state))

if n < 4:
    return (0, 0, 2, 0)[n]

initial_state = (0, 0, 2, 0)
final_state = reduce(compute_next_state, range(n - 3), initial_state)
return final_state[3]
