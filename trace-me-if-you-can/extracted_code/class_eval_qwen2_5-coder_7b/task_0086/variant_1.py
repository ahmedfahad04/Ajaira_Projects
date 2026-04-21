class GameBoard:
    def __init__(self, dimension=3):
        self.matrix = [[' ' for _ in range(dimension)] for _ in range(dimension)]
        self.player_turn = 'X'

    def update_board(self, x, y):
        if self.matrix[x][y] == ' ':
            self.matrix[x][y] = self.player_turn
            self.player_turn = 'O' if self.player_turn == 'X' else 'X'
            return True
        else:
            return False

    def evaluate_winner(self):
        for row in self.matrix:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
        for col in range(3):
            if self.matrix[0][col] == self.matrix[1][col] == self.matrix[2][col] != ' ':
                return self.matrix[0][col]
        if self.matrix[0][0] == self.matrix[1][1] == self.matrix[2][2] != ' ':
            return self.matrix[0][0]
        if self.matrix[0][2] == self.matrix[1][1] == self.matrix[2][0] != ' ':
            return self.matrix[0][2]
        return None

    def check_draw(self):
        for row in self.matrix:
            if ' ' in row:
                return False
        return True
