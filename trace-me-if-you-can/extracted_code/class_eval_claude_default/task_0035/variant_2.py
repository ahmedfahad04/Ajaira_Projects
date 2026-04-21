class EightPuzzle:
    def __init__(self, initial_state):
        self.initial_state = tuple(tuple(row) for row in initial_state)
        self.goal_state = ((1, 2, 3), (4, 5, 6), (7, 8, 0))
        self.moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

    def state_to_list(self, state):
        return [list(row) for row in state]

    def list_to_state(self, state_list):
        return tuple(tuple(row) for row in state_list)

    def find_blank(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j

    def is_valid_position(self, i, j):
        return 0 <= i < 3 and 0 <= j < 3

    def generate_next_states(self, state):
        blank_i, blank_j = self.find_blank(state)
        next_states = []
        
        for direction, (di, dj) in self.moves.items():
            new_i, new_j = blank_i + di, blank_j + dj
            if self.is_valid_position(new_i, new_j):
                state_list = self.state_to_list(state)
                state_list[blank_i][blank_j], state_list[new_i][new_j] = state_list[new_i][new_j], state_list[blank_i][blank_j]
                next_states.append((self.list_to_state(state_list), direction))
        
        return next_states

    def solve(self):
        frontier = [(self.initial_state, [])]
        explored = {self.initial_state}

        while frontier:
            current_state, path = frontier.pop(0)

            if current_state == self.goal_state:
                return path

            for next_state, move in self.generate_next_states(current_state):
                if next_state not in explored:
                    explored.add(next_state)
                    frontier.append((next_state, path + [move]))

        return None
