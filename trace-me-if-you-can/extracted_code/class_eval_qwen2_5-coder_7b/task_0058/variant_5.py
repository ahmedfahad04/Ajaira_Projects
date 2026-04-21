import random

class Minesweeper:
    def __init__(self, grid_size, mine_amount):
        self.grid_size = grid_size
        self.mine_amount = mine_amount
        self.mine_grid = self.distribute_mines()
        self.player_grid = [['-' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.game_finished = False
        self.score_points = 0

    def distribute_mines(self):
        mine_grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for _ in range(self.mine_amount):
            while True:
                row, col = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
                if mine_grid[row][col] == 0:
                    mine_grid[row][col] = 'X'
                    break
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= row + i < self.grid_size and 0 <= col + j < self.grid_size and mine_grid[row + i][col + j] != 'X':
                        mine_grid[row + i][col + j] += 1
        return mine_grid

    def create_player_grid(self):
        return [['-' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

    def is_game_won(self):
        for row in self.player_grid:
            if '-' in row:
                return False
        return True

    def reveal_square(self, row, col):
        if self.mine_grid[row][col] == 'X':
            self.game_finished = True
            return False
        else:
            self.player_grid[row][col] = self.mine_grid[row][col]
            self.score_points += 1
            if self.is_game_won():
                self.game_finished = True
                return True
            return self.player_grid[row][col]
