class GomokuGame:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = [[' ' for _ in range(board_size)] for _ in range(board_size)]
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
        winning_patterns = self._generate_all_winning_patterns()
        for pattern in winning_patterns:
            symbols = [self.board[r][c] for r, c in pattern]
            if len(set(symbols)) == 1 and symbols[0] != ' ':
                return symbols[0]
        return None

    def _generate_all_winning_patterns(self):
        patterns = []
        # Horizontal patterns
        for row in range(self.board_size):
            for col in range(self.board_size - 4):
                patterns.append([(row, col + i) for i in range(5)])
        
        # Vertical patterns
        for row in range(self.board_size - 4):
            for col in range(self.board_size):
                patterns.append([(row + i, col) for i in range(5)])
        
        # Diagonal (top-left to bottom-right) patterns
        for row in range(self.board_size - 4):
            for col in range(self.board_size - 4):
                patterns.append([(row + i, col + i) for i in range(5)])
        
        # Diagonal (top-right to bottom-left) patterns
        for row in range(self.board_size - 4):
            for col in range(4, self.board_size):
                patterns.append([(row + i, col - i) for i in range(5)])
        
        return patterns
