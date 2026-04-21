class GomokuGameBoard:
    def __init__(self, dimension):
        self.dimension = dimension
        self.board = [[' ' for _ in range(dimension)] for _ in range(dimension)]
        self.active_player = 'X'

    def place_marker(self, row, col):
        if self.board[row][col] == ' ':
            self.board[row][col] = self.active_player
            self.active_player = 'O' if self.active_player == 'X' else 'X'
            return True
        return False

    def determine_winner(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for i in range(self.dimension):
            for j in range(self.dimension):
                if self.board[i][j] != ' ':
                    for dx, dy in directions:
                        if self._check_sequence(i, j, dx, dy):
                            return self.board[i][j]
        return None

    def _check_sequence(self, row, col, dx, dy):
        count = 1
        marker = self.board[row][col]
        for k in range(1, 5):
            new_row, new_col = row + dx * k, col + dy * k
            if not (0 <= new_row < self.dimension and 0 <= new_col < self.dimension):
                return False
            if self.board[new_row][new_col] != marker:
                return False
            count += 1
        return count == 5
