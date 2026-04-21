class GomokuGame:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = [[' ' for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 'X'

    def make_move(self, row, col):
        if self._is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

    def _is_valid_move(self, row, col):
        return (0 <= row < self.board_size and 
                0 <= col < self.board_size and 
                self.board[row][col] == ' ')

    def check_winner(self):
        def check_line(start_row, start_col, delta_row, delta_col):
            if not (0 <= start_row < self.board_size and 0 <= start_col < self.board_size):
                return None
            
            symbol = self.board[start_row][start_col]
            if symbol == ' ':
                return None
                
            consecutive_count = 0
            row, col = start_row, start_col
            
            while (0 <= row < self.board_size and 
                   0 <= col < self.board_size and 
                   self.board[row][col] == symbol):
                consecutive_count += 1
                row += delta_row
                col += delta_col
                
            return symbol if consecutive_count >= 5 else None

        # Check all possible starting positions and directions
        for row in range(self.board_size):
            for col in range(self.board_size):
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    winner = check_line(row, col, dr, dc)
                    if winner:
                        return winner
        return None
