class PuzzleGame:
        def __init__(self, initial):
            self.initial = initial
            self.target = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

        def locate_blank(self, board):
            for row_idx, row in enumerate(board):
                for col_idx, value in enumerate(row):
                    if value == 0:
                        return row_idx, col_idx

        def switch(self, board, direction):
            blank_row, blank_col = self.locate_blank(board)
            new_board = [row[:] for row in board]

            if direction == 'up' and blank_row > 0:
                new_board[blank_row][blank_col], new_board[blank_row - 1][blank_col] = new_board[blank_row - 1][blank_col], new_board[blank_row][blank_col]
            elif direction == 'down' and blank_row < 2:
                new_board[blank_row][blank_col], new_board[blank_row + 1][blank_col] = new_board[blank_row + 1][blank_col], new_board[blank_row][blank_col]
            elif direction == 'left' and blank_col > 0:
                new_board[blank_row][blank_col], new_board[blank_row][blank_col - 1] = new_board[blank_row][blank_col - 1], new_board[blank_row][blank_col]
            elif direction == 'right' and blank_col < 2:
                new_board[blank_row][blank_col], new_board[blank_row][blank_col + 1] = new_board[blank_row][blank_col + 1], new_board[blank_row][blank_col]

            return new_board

        def potential_moves(self, state):
            moves = []
            blank_row, blank_col = self.locate_blank(state)

            if blank_row > 0:
                moves.append('up')
            if blank_row < 2:
                moves.append('down')
            if blank_col > 0:
                moves.append('left')
            if blank_col < 2:
                moves.append('right')

            return moves

        def resolve(self):
            open_queue = [(self.initial, [])]
            explored_states = set()

            while open_queue:
                current_state, path = open_queue.pop(0)
                explored_states.add(tuple(map(tuple, current_state)))

                if current_state == self.target:
                    return path

                for move in self.potential_moves(current_state):
                    new_state = self.switch(current_state, move)
                    if tuple(map(tuple, new_state)) not in explored_states:
                        open_queue.append((new_state, path + [move]))

            return None
