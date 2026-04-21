class GomokuGame:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = [[' ' for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 'X'

    def make_move(self, row, col):
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            self._switch_player()
            return True
        return False

    def _switch_player(self):
        self.current_player = {'X': 'O', 'O': 'X'}[self.current_player]

    def check_winner(self):
        import itertools
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] != ' ':
                    current_symbol = self.board[row][col]
                    
                    # Check all four directions using itertools
                    direction_vectors = [(0, 1), (1, 0), (1, 1), (1, -1)]
                    
                    for dx, dy in direction_vectors:
                        positions = [(row + dx * i, col + dy * i) for i in range(5)]
                        
                        if all(0 <= r < self.board_size and 0 <= c < self.board_size 
                              for r, c in positions):
                            if all(self.board[r][c] == current_symbol 
                                  for r, c in positions):
                                return current_symbol
        return None
