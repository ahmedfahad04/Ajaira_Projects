class EightPuzzle:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    def serialize_state(self, state):
        return ''.join(str(state[i][j]) for i in range(3) for j in range(3))

    def deserialize_state(self, serialized):
        state = []
        for i in range(3):
            row = []
            for j in range(3):
                row.append(int(serialized[i * 3 + j]))
            state.append(row)
        return state

    def find_blank(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j

    def swap_positions(self, state, pos1, pos2):
        new_state = [row[:] for row in state]
        i1, j1 = pos1
        i2, j2 = pos2
        new_state[i1][j1], new_state[i2][j2] = new_state[i2][j2], new_state[i1][j1]
        return new_state

    def get_valid_swaps(self, state):
        blank_pos = self.find_blank(state)
        i, j = blank_pos
        swaps = []
        
        adjacent = [(i-1, j, 'up'), (i+1, j, 'down'), (i, j-1, 'left'), (i, j+1, 'right')]
        
        for ni, nj, direction in adjacent:
            if 0 <= ni < 3 and 0 <= nj < 3:
                new_state = self.swap_positions(state, blank_pos, (ni, nj))
                swaps.append((new_state, direction))
        
        return swaps

    def solve(self):
        search_queue = [(self.initial_state, [])]
        visited_states = set([self.serialize_state(self.initial_state)])

        while search_queue:
            current_state, solution_path = search_queue.pop(0)

            if current_state == self.goal_state:
                return solution_path

            for next_state, move_direction in self.get_valid_swaps(current_state):
                state_signature = self.serialize_state(next_state)
                if state_signature not in visited_states:
                    visited_states.add(state_signature)
                    search_queue.append((next_state, solution_path + [move_direction]))

        return None
