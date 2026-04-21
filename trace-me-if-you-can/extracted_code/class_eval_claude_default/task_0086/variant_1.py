class TicTacToe:
    def __init__(self, N=3):
        self.board = [[' ' for _ in range(N)] for _ in range(3)]
        self.current_player = 'X'
        self.players = ['X', 'O']
        self.player_index = 0

    def make_move(self, row, col):
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            self.player_index = 1 - self.player_index
            self.current_player = self.players[self.player_index]
            return True
        return False

    def check_winner(self):
        # Check rows and columns using list comprehension
        lines = (
            [self.board[i] for i in range(3)] +  # rows
            [[self.board[i][j] for i in range(3)] for j in range(3)] +  # columns
            [[self.board[i][i] for i in range(3)]] +  # main diagonal
            [[self.board[i][2-i] for i in range(3)]]  # anti-diagonal
        )
        
        for line in lines:
            if line[0] == line[1] == line[2] != ' ':
                return line[0]
        return None

    def is_board_full(self):
        return all(cell != ' ' for row in self.board for cell in row)
