class GomokuGame:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = [[' ' for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 'X'
        self.move_history = []

    def make_move(self, row, col):
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            self.move_history.append((row, col, self.current_player))
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

    def check_winner(self):
        for row, col, player in self.move_history:
            if self._has_winning_line(row, col, player):
                return player
        return None

    def _has_winning_line(self, row, col, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            line_length = 1
            # Check forward direction
            for step in range(1, 5):
                nr, nc = row + dx * step, col + dy * step
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr][nc] == player:
                    line_length += 1
                else:
                    break
            # Check backward direction
            for step in range(1, 5):
                nr, nc = row - dx * step, col - dy * step
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr][nc] == player:
                    line_length += 1
                else:
                    break
            if line_length >= 5:
                return True
        return False
