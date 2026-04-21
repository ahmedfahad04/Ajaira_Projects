class GomokuField:
    def __init__(self, dimension):
        self.dimension = dimension
        self.cells = [[' ' for _ in range(dimension)] for _ in range(dimension)]
        self.turn = 'X'

    def insert_piece(self, x, y):
        if self.cells[x][y] == ' ':
            self.cells[x][y] = self.turn
            self.turn = 'O' if self.turn == 'X' else 'X'
            return True
        return False

    def get_winner(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for i in range(self.dimension):
            for j in range(self.dimension):
                if self.cells[i][j] != ' ':
                    for dx, dy in directions:
                        if self._validate_line(i, j, dx, dy):
                            return self.cells[i][j]
        return None

    def _validate_line(self, x, y, dx, dy):
        count = 1
        piece = self.cells[x][y]
        for k in range(1, 5):
            new_x, new_y = x + dx * k, y + dy * k
            if not (0 <= new_x < self.dimension and 0 <= new_y < self.dimension):
                return False
            if self.cells[new_x][new_y] != piece:
                return False
            count += 1
        return count == 5
