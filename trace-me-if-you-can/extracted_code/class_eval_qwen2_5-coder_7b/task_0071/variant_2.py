class SlidePuzzle:
    def __init__(self, board):
        self.board = board
        self.blank_row = 0
        self.blank_col = 0
        self.cleared_positions = []
        self.empty_count = 0
        self.has_solved = False

        self.initialize_game()

    def initialize_game(self):
        for row_idx, row in enumerate(self.board):
            for col_idx, cell in enumerate(row):
                if cell == "B":
                    self.blank_row = row_idx
                    self.blank_col = col_idx
                elif cell == "E":
                    self.cleared_positions.append((row_idx, col_idx))
                    self.empty_count += 1
                elif cell == "X":
                    self.board[row_idx][col_idx] = None

    def verify_completion(self):
        box_on_empty_count = 0
        for block in self.cleared_positions:
            if block in self.board:
                box_on_empty_count += 1
        if box_on_empty_count == self.empty_count:
            self.has_solved = True
        return self.has_solved

    def perform_move(self, move):
        new_blank_row = self.blank_row
        new_blank_col = self.blank_col

        if move == "up":
            new_blank_row -= 1
        elif move == "down":
            new_blank_row += 1
        elif move == "left":
            new_blank_col -= 1
        elif move == "right":
            new_blank_col += 1

        if self.board[new_blank_row][new_blank_col] is not None:
            if (new_blank_row, new_blank_col) in self.cleared_positions:
                new_cleared_x = new_blank_row + (new_blank_row - self.blank_row)
                new_cleared_y = new_blank_col + (new_blank_col - self.blank_col)

                if self.board[new_cleared_x][new_cleared_y] is None:
                    self.cleared_positions.remove((new_blank_row, new_blank_col))
                    self.cleared_positions.append((new_cleared_x, new_cleared_y))
                    self.blank_row = new_blank_row
                    self.blank_col = new_blank_col
            else:
                self.blank_row = new_blank_row
                self.blank_col = new_blank_col

        return self.verify_completion()
