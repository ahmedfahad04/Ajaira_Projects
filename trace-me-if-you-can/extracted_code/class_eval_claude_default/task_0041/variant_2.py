class GomokuGame:
    DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = {}
        self.current_player = 'X'

    def make_move(self, row, col):
        if (row, col) not in self.board:
            self.board[(row, col)] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

    def check_winner(self):
        for (row, col), player in self.board.items():
            for direction in self.DIRECTIONS:
                if self._count_consecutive(row, col, direction, player) >= 5:
                    return player
        return None

    def _count_consecutive(self, start_row, start_col, direction, player):
        dx, dy = direction
        count = 0
        row, col = start_row, start_col
        
        while (0 <= row < self.board_size and 
               0 <= col < self.board_size and 
               self.board.get((row, col)) == player):
            count += 1
            if count >= 5:
                return count
            row += dx
            col += dy
        
        return count
