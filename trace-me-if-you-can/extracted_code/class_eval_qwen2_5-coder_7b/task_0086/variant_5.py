class TicTacToeGame:
    def __init__(self, dimension=3):
        self.game_board = [[' ' for _ in range(dimension)] for _ in range(dimension)]
        self.player_turn = 'X'

    def insert_token(self, row, col):
        if self.game_board[row][col] == ' ':
            self.game_board[row][col] = self.player_turn
            self.player_turn = 'O' if self.player_turn == 'X' else 'X'
            return True
        else:
            return False

    def determine_winner(self):
        for row in self.game_board:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
        for col in range(3):
            if self.game_board[0][col] == self.game_board[1][col] == self.game_board[2][col] != ' ':
                return self.game_board[0][col]
        if self.game_board[0][0] == self.game_board[1][1] == self.game_board[2][2] != ' ':
            return self.game_board[0][0]
        if self.game_board[0][2] == self.game_board[1][1] == self.game_board[2][0] != ' ':
            return self.game_board[0][2]
        return None

    def is_game_full(self):
        for row in self.game_board:
            if ' ' in row:
                return False
        return True
