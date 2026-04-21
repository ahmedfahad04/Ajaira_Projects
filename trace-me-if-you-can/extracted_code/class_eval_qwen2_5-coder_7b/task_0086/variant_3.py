class TicTacToeEngine:
    def __init__(self, board_size=3):
        self.table = [[' ' for _ in range(board_size)] for _ in range(3)]
        self.mark = 'X'

    def insert_mark(self, x, y):
        if self.table[x][y] == ' ':
            self.table[x][y] = self.mark
            self.mark = 'O' if self.mark == 'X' else 'X'
            return True
        else:
            return False

    def find_winner(self):
        for row in self.table:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
        for col in range(3):
            if self.table[0][col] == self.table[1][col] == self.table[2][col] != ' ':
                return self.table[0][col]
        if self.table[0][0] == self.table[1][1] == self.table[2][2] != ' ':
            return self.table[0][0]
        if self.table[0][2] == self.table[1][1] == self.table[2][0] != ' ':
            return self.table[0][2]
        return None

    def is_game_over(self):
        for row in self.table:
            if ' ' in row:
                return False
        return True
