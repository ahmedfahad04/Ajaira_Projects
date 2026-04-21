import random

class MinesweeperGame:
    def __init__(self, n, k) -> None:
        self.n = n
        self.k = k
        self.score = 0
        self._setup_game()

    def _setup_game(self):
        """Initialize both game maps in a single setup phase"""
        self.minesweeper_map = self.generate_mine_sweeper_map()
        self.player_map = self.generate_playerMap()

    def _increment_if_safe(self, board, row, col):
        """Helper to safely increment a cell if it's not a mine"""
        if board[row][col] != 'X':
            board[row][col] += 1

    def generate_mine_sweeper_map(self):
        board = [[0 for _ in range(self.n)] for _ in range(self.n)]
        
        for mine_num in range(self.k):
            x, y = random.randint(0, self.n-1), random.randint(0, self.n-1)
            board[y][x] = 'X'
            
            # Use boundary checking with helper method
            boundary_checks = [
                (y, x+1, x >= 0 and x < self.n-1),      # right
                (y, x-1, x > 0 and x < self.n),         # left  
                (y-1, x-1, x > 0 and y > 0),            # top-left
                (y-1, x+1, x >= 0 and x < self.n-1 and y > 0),  # top-right
                (y-1, x, y > 0),                        # top
                (y+1, x+1, x >= 0 and x < self.n-1 and y < self.n-1),  # bottom-right
                (y+1, x-1, x > 0 and y < self.n-1),     # bottom-left
                (y+1, x, y < self.n-1)                  # bottom
            ]
            
            for row, col, is_valid in boundary_checks:
                if is_valid:
                    self._increment_if_safe(board, row, col)
        
        return board
    
    def generate_playerMap(self):
        return [['-' for _ in range(self.n)] for _ in range(self.n)]

    def check_won(self, map):
        remaining_safe_cells = [
            (i, j) for i in range(self.n) for j in range(self.n)
            if map[i][j] == '-' and self.minesweeper_map[i][j] != 'X'
        ]
        return len(remaining_safe_cells) == 0
    
    def sweep(self, x, y):
        revealed_value = self.minesweeper_map[x][y]
        
        if revealed_value == 'X':
            return False
        else:
            self.player_map[x][y] = revealed_value
            self.score += 1
            game_won = self.check_won(self.player_map)
            return True if game_won else self.player_map
