class PuzzleSolver:
        def __init__(self, start_state):
            self.start_state = start_state
            self.final_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

        def get_zero_position(self, state):
            for row_index, row in enumerate(state):
                for col_index, cell in enumerate(row):
                    if cell == 0:
                        return row_index, col_index

        def make_move(self, state, direction):
            row, col = self.get_zero_position(state)
            new_state = [row[:] for row in state]

            if direction == 'up' and row > 0:
                new_state[row][col], new_state[row - 1][col] = new_state[row - 1][col], new_state[row][col]
            elif direction == 'down' and row < 2:
                new_state[row][col], new_state[row + 1][col] = new_state[row + 1][col], new_state[row][col]
            elif direction == 'left' and col > 0:
                new_state[row][col], new_state[row][col - 1] = new_state[row][col - 1], new_state[row][col]
            elif direction == 'right' and col < 2:
                new_state[row][col], new_state[row][col + 1] = new_state[row][col + 1], new_state[row][col]

            return new_state

        def get_available_moves(self, state):
            row, col = self.get_zero_position(state)
            moves = []

            if row > 0:
                moves.append('up')
            if row < 2:
                moves.append('down')
            if col > 0:
                moves.append('left')
            if col < 2:
                moves.append('right')

            return moves

        def solve_puzzle(self):
            open_set = [(self.start_state, [])]
            closed_set = set()

            while open_set:
                current_state, path = open_set.pop(0)
                closed_set.add(tuple(map(tuple, current_state)))

                if current_state == self.final_state:
                    return path

                for move in self.get_available_moves(current_state):
                    new_state = self.make_move(current_state, move)
                    if tuple(map(tuple, new_state)) not in closed_set:
                        open_set.append((new_state, path + [move]))

            return None
