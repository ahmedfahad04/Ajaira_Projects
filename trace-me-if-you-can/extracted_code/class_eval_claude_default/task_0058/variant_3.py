import random
from itertools import product

class MinesweeperGame:
    def __init__(self, n, k) -> None:
        self.n = n
        self.k = k
        self.minesweeper_map = self.generate_mine_sweeper_map()
        self.player_map = self.generate_playerMap()
        self.score = 0

    def _get_neighbors(self, x, y):
        """Generator for valid neighboring coordinates"""
        for dx, dy in product([-1, 0, 1], repeat=2):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.n and 0 <= ny < self.n:
                yield nx, ny

    def generate_mine_sweeper_map(self):
        game_board = [[0 for _ in range(self.n)] for _ in range(self.n)]
        
        mines_placed = 0
        while mines_placed < self.k:
            x, y = random.randint(0, self.n-1), random.randint(0, self.n-1)
            game_board[y][x] = 'X'
            
            # Increment all valid neighbors
            for nx, ny in self._get_neighbors(x, y):
                if game_board[ny][nx] != 'X':
                    game_board[ny][nx] += 1
            
            mines_placed += 1
        
        return game_board
    
    def generate_playerMap(self):
        return [['-' for _ in range(self.n)] for _ in range(self.n)]

    def check_won(self, map):
        unrevealed_safe_cells = sum(
            1 for i, j in product(range(self.n), repeat=2)
            if map[i][j] == '-' and self.minesweeper_map[i][j] != 'X'
        )
        return unrevealed_safe_cells == 0
    
    def sweep(self, x, y):
        cell_value = self.minesweeper_map[x][y]
        
        if cell_value == 'X':
            return False
        
        self.player_map[x][y] = cell_value
        self.score += 1
        
        return True if self.check_won(self.player_map) else self.player_map
