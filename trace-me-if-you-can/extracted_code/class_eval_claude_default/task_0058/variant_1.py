import random
from typing import List, Union

class MinesweeperGame:
    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k
        self.minesweeper_map = self._create_game_board()
        self.player_map = [['-'] * n for _ in range(n)]
        self.score = 0

    def _create_game_board(self) -> List[List[Union[int, str]]]:
        board = [[0] * self.n for _ in range(self.n)]
        
        # Place mines
        for _ in range(self.k):
            mine_x, mine_y = random.randint(0, self.n-1), random.randint(0, self.n-1)
            board[mine_y][mine_x] = 'X'
            
            # Update adjacent cells using relative positions
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    adj_y, adj_x = mine_y + dy, mine_x + dx
                    if (0 <= adj_y < self.n and 0 <= adj_x < self.n and 
                        board[adj_y][adj_x] != 'X'):
                        board[adj_y][adj_x] += 1
        
        return board

    def generate_playerMap(self):
        return [['-'] * self.n for _ in range(self.n)]

    def check_won(self, player_board):
        return all(
            player_board[i][j] != '-' or self.minesweeper_map[i][j] == 'X'
            for i in range(self.n)
            for j in range(self.n)
        )
    
    def sweep(self, x, y):
        if self.minesweeper_map[x][y] == 'X':
            return False
        
        self.player_map[x][y] = self.minesweeper_map[x][y]
        self.score += 1
        
        return True if self.check_won(self.player_map) else self.player_map
