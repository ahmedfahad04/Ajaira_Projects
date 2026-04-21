import random

class MinesweeperGame:
    MINE_SYMBOL = 'X'
    HIDDEN_SYMBOL = '-'
    DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def __init__(self, n, k) -> None:
        self.n = n
        self.k = k
        self.minesweeper_map = self.generate_mine_sweeper_map()
        self.player_map = self.generate_playerMap()
        self.score = 0

    def _is_valid_position(self, row, col):
        return 0 <= row < self.n and 0 <= col < self.n

    def _place_single_mine(self, board, mine_row, mine_col):
        board[mine_row][mine_col] = self.MINE_SYMBOL
        
        for dr, dc in self.DIRECTIONS:
            neighbor_row, neighbor_col = mine_row + dr, mine_col + dc
            if (self._is_valid_position(neighbor_row, neighbor_col) and 
                board[neighbor_row][neighbor_col] != self.MINE_SYMBOL):
                board[neighbor_row][neighbor_col] += 1

    def generate_mine_sweeper_map(self):
        board = [[0 for _ in range(self.n)] for _ in range(self.n)]
        
        for _ in range(self.k):
            mine_col = random.randint(0, self.n-1)
            mine_row = random.randint(0, self.n-1)
            self._place_single_mine(board, mine_row, mine_col)
        
        return board
    
    def generate_playerMap(self):
        return [[self.HIDDEN_SYMBOL for _ in range(self.n)] for _ in range(self.n)]

    def check_won(self, map):
        for row in range(self.n):
            for col in range(self.n):
                if (map[row][col] == self.HIDDEN_SYMBOL and 
                    self.minesweeper_map[row][col] != self.MINE_SYMBOL):
                    return False
        return True
    
    def sweep(self, x, y):
        if self.minesweeper_map[x][y] == self.MINE_SYMBOL:
            return False
        
        self.player_map[x][y] = self.minesweeper_map[x][y]
        self.score += 1
        
        if self.check_won(self.player_map):
            return True
        return self.player_map
