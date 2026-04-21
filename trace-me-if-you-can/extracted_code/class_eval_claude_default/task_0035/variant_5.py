class EightPuzzle:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.move_deltas = {
            'up': (-1, 0),
            'down': (1, 0), 
            'left': (0, -1),
            'right': (0, 1)
        }

    def find_blank(self, state):
        flat = [cell for row in state for cell in row]
        blank_idx = flat.index(0)
        return divmod(blank_idx, 3)

    def apply_move(self, state, move_name):
        blank_row, blank_col = self.find_blank(state)
        delta_row, delta_col = self.move_deltas[move_name]
        target_row, target_col = blank_row + delta_row, blank_col + delta_col
        
        if not (0 <= target_row < 3 and 0 <= target_col < 3):
            return None
            
        result = [row[:] for row in state]
        result[blank_row][blank_col] = result[target_row][target_col]
        result[target_row][target_col] = 0
        return result

    def get_legal_moves(self, state):
        legal = []
        for move_name in self.move_deltas:
            if self.apply_move(state, move_name) is not None:
                legal.append(move_name)
        return legal

    def breadth_first_search(self):
        def state_hash(s):
            return hash(tuple(tuple(row) for row in s))
        
        queue = [(self.initial_state, [])]
        visited = {state_hash(self.initial_state)}

        while queue:
            state, path = queue.pop(0)

            if state == self.goal_state:
                return path

            for move in self.get_legal_moves(state):
                next_state = self.apply_move(state, move)
                next_hash = state_hash(next_state)
                
                if next_hash not in visited:
                    visited.add(next_hash)
                    queue.append((next_state, path + [move]))

        return None

    def solve(self):
        return self.breadth_first_search()
