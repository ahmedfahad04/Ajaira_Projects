import random

class MinesweeperGame:
    def __init__(self, n, k) -> None:
        self.n = n
        self.k = k
        self.minesweeper_map, self.player_map = self._initialize_maps()
        self.score = 0

    def _initialize_maps(self):
        mine_map = self._build_minefield()
        player_map = [row[:] for row in [['-'] * self.n] * self.n]
        return mine_map, player_map

    def _build_minefield(self):
        field = [[0] * self.n for _ in range(self.n)]
        
        mine_positions = []
        for _ in range(self.k):
            x, y = random.randint(0, self.n-1), random.randint(0, self.n-1)
            mine_positions.append((x, y))
            field[y][x] = 'X'
        
        # Calculate numbers for each mine
        for mine_x, mine_y in mine_positions:
            self._update_adjacent_counts(field, mine_x, mine_y)
        
        return field

    def _update_adjacent_counts(self, field, mine_x, mine_y):
        adjacent_offsets = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        for dx, dy in adjacent_offsets:
            adj_x, adj_y = mine_x + dx, mine_y + dy
            if (0 <= adj_x < self.n and 0 <= adj_y < self.n and 
                field[adj_y][adj_x] != 'X'):
                field[adj_y][adj_x] += 1

    def generate_mine_sweeper_map(self):
        return self.minesweeper_map
    
    def generate_playerMap(self):
        return self.player_map

    def check_won(self, map):
        for row_idx in range(self.n):
            for col_idx in range(self.n):
                is_hidden = map[row_idx][col_idx] == '-'
                is_safe = self.minesweeper_map[row_idx][col_idx] != 'X'
                if is_hidden and is_safe:
                    return False
        return True
    
    def sweep(self, x, y):
        target_cell = self.minesweeper_map[x][y]
        
        if target_cell == 'X':
            return False
        else:
            self.player_map[x][y] = target_cell
            self.score += 1
            won = self.check_won(self.player_map)
            return True if won else self.player_map
