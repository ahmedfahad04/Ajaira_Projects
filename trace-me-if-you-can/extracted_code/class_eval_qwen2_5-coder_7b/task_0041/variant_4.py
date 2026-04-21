class GomokuGrid:
    def __init__(self, size):
        self.size = size
        self.positions = [[' ' for _ in range(size)] for _ in range(size)]
        self.player_turn = 'X'

    def place_symbol(self, row, col):
        if self.positions[row][col] == ' ':
            self.positions[row][col] = self.player_turn
            self.player_turn = 'O' if self.player_turn == 'X' else 'X'
            return True
        return False

    def identify_winner(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for i in range(self.size):
            for j in range(self.size):
                if self.positions[i][j] != ' ':
                    for dx, dy in directions:
                        if self._check_sequence(i, j, dx, dy):
                            return self.positions[i][j]
        return None

    def _check_sequence(self, row, col, dx, dy):
        count = 1
        symbol = self.positions[row][col]
        for k in range(1, 5):
            new_row, new_col = row + dx * k, col + dy * k
            if not (0 <= new_row < self.size and 0 <= new_col < self.size):
                return False
            if self.positions[new_row][new_col] != symbol:
                return False
            count += 1
        return count == 5
