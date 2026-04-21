# Version 3: Matrix exponentiation approach
import numpy as np

if n < 4:
    return [0, 0, 2, 0][n]

# Transition matrix for the recurrence relation
transition_matrix = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 1, 1]
], dtype=object)

# Initial state vector
initial_state = np.array([0, 0, 0, 2], dtype=object)

# Matrix power
result_matrix = np.linalg.matrix_power(transition_matrix, n - 3)
final_state = result_matrix @ initial_state

return int(final_state[3])
