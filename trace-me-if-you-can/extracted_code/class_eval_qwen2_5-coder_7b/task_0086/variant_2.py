class TicTacToeGame:
    def __init__(self, size=3):
        self.grid = [[' ' for _ in range(size)] for _ in range(3)]
        self.symbol = 'X'

    def place_mark(self, i, j):
        if self.grid[i][j] == ' ':
            self.grid[i][j] = self.symbol
            self.symbol = 'O' if self.symbol == 'X' else 'X'
            return True
        else:
            return False

    def identify_winner(self):
        for row in self.grid:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
        for col in range(3):
            if self.grid[0][col] == self.grid[1][col] == self.grid[2][col] != ' ':
                return self.grid[0][col]
        if self.grid[0][0] == self.grid[1][1] == self.grid[2][2] != ' ':
            return self.grid[0][0]
        if self.grid[0][2] == self.grid[1][1] == self.grid[2][0] != ' ':
            return self.grid[0][2]
        return None

    def check_stalemate(self):
        for row in self.grid:
            if ' ' in row:
                return False
        return True
