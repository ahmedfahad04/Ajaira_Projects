class GomokuBoardGame:
    def __init__(self, size):
        self.size = size
        self.squares = [[' ' for _ in range(size)] for _ in range(size)]
        self.current_mark = 'X'

    def insert_mark(self, row, col):
        if self.squares[row][col] == ' ':
            self.squares[row][col] = self.current_mark
            self.current_mark = 'O' if self.current_mark == 'X' else 'X'
            return True
        return False

    def check_winner(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for i in range(self.size):
            for j in range(self.size):
                if self.squares[i][j] != ' ':
                    for dx, dy in directions:
                        if self._check_sequence(i, j, dx, dy):
                            return self.squares[i][j]
        return None

    def _check_sequence(self, row, col, dx, dy):
        count = 1
        mark = self.squares[row][col]
        for k in range(1, 5):
            new_row, new_col = row + dx * k, col + dy * k
            if not (0 <= new_row < self.size and 0 <= new_col < self.size):
                return False
            if self.squares[new_row][new_col] != mark:
                return False
            count += 1
        return count == 5
