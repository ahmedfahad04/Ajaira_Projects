class TicTacToe:
    def __init__(self, N=3):
        self.board = [[' ' for _ in range(N)] for _ in range(3)]
        self.current_player = 'X'
        self.win_patterns = [
            [(0,0), (0,1), (0,2)], [(1,0), (1,1), (1,2)], [(2,0), (2,1), (2,2)],  # rows
            [(0,0), (1,0), (2,0)], [(0,1), (1,1), (2,1)], [(0,2), (1,2), (2,2)],  # columns
            [(0,0), (1,1), (2,2)], [(0,2), (1,1), (2,0)]  # diagonals
        ]

    def make_move(self, row, col):
        if self._is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self._switch_player()
            return True
        return False

    def _is_valid_move(self, row, col):
        return self.board[row][col] == ' '

    def _switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        for pattern in self.win_patterns:
            values = [self.board[r][c] for r, c in pattern]
            if values[0] == values[1] == values[2] != ' ':
                return values[0]
        return None

    def is_board_full(self):
        return ' ' not in [cell for row in self.board for cell in row]
