class PuzzleResolution:
        def __init__(self, start):
            self.start = start
            self.goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

        def locate_zero(self, board):
            for row_idx, row in enumerate(board):
                for col_idx, value in enumerate(row):
                    if value == 0:
                        return row_idx, col_idx

        def make_move(self, board, direction):
            zero_row, zero_col = self.locate_zero(board)
            new_board = [row[:] for row in board]

            if direction == 'up' and zero_row > 0:
                new_board[zero_row][zero_col], new_board[zero_row - 1][zero_col] = new_board[zero_row - 1][zero_col], new_board[zero_row][zero_col]
            elif direction == 'down' and zero_row < 2:
                new_board[zero_row][zero_col], new_board[zero_row + 1][zero_col] = new_board[zero_row + 1][zero_col], new_board[zero_row][zero_col]
            elif direction == 'left' and zero_col > 0:
                new_board[zero_row][zero_col], new_board[zero_row][zero_col - 1] = new_board[zero_row][zero_col - 1], new_board[zero_row][zero_col]
            elif direction == 'right' and zero_col < 2:
                new_board[zero_row][zero_col], new_board[zero_row][zero_col + 1] = new_board[zero_row][zero_col + 1], new_board[zero_row][zero_col]

            return new_board

        def potential_moves(self, state):
            moves = []
            zero_row, zero_col = self.locate_zero(state)

            if zero_row > 0:
                moves.append('up')
            if zero_row < 2:
                moves.append('down')
            if zero_col > 0:
                moves.append('left')
            if zero_col < 2:
                moves.append('right')

            return moves

        def resolve_puzzle(self):
            open_list = [(self.start, [])]
            closed_set = set()

            while open_list:
                current_state, path = open_list.pop(0)
                closed_set.add(tuple(map(tuple, current_state)))

                if current_state == self.goal:
                    return path

                for move in self.potential_moves(current_state):
                    new_state = self.make_move(current_state, move)
                    if tuple(map(tuple, new_state)) not in closed_set:
                        open_list.append((new_state, path + [move]))

            return None
