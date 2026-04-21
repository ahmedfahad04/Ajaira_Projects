class GomokuBoard:
    def __init__(self, size):
        self.size = size
        self.grid = [[' ' for _ in range(size)] for _ in range(size)]
        self.player = 'X'

    def add_mark(self, x, y):
        if self.grid[x][y] == ' ':
            self.grid[x][y] = self.player
            self.player = 'O' if self.player == 'X' else 'X'
            return True
        return False

    def find_winner(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] != ' ':
                    for dx, dy in directions:
                        if self._validate_sequence(i, j, dx, dy):
                            return self.grid[i][j]
        return None

    def _validate_sequence(self, x, y, dx, dy):
        count = 1
        mark = self.grid[x][y]
        for k in range(1, 5):
            nx, ny = x + dx * k, y + dy * k
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                return False
            if self.grid[nx][ny] != mark:
                return False
            count += 1
        return count == 5
