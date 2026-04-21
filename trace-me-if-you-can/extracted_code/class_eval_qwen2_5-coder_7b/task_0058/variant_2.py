import random

class Minesweeper:
    def __init__(self, grid_size, mine_quantity):
        self.grid_size = grid_size
        self.mine_quantity = mine_quantity
        self.field = self.create_field()
        self.revealed = [['-' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.game_over = False
        self.points = 0

    def create_field(self):
        field = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for _ in range(self.mine_quantity):
            while True:
                row, col = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
                if field[row][col] == 0:
                    field[row][col] = 'X'
                    break
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= row + i < self.grid_size and 0 <= col + j < self.grid_size and field[row + i][col + j] != 'X':
                        field[row + i][col + j] += 1
        return field

    def start_new_game(self):
        self.field = self.create_field()
        self.revealed = [['-' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.game_over = False
        self.points = 0

    def evaluate_game(self):
        for row in self.revealed:
            if '-' in row:
                return False
        return True

    def uncover(self, x, y):
        if self.field[x][y] == 'X':
            self.game_over = True
            return False
        else:
            self.revealed[x][y] = self.field[x][y]
            self.points += 1
            if self.evaluate_game():
                self.game_over = True
                return True
            return self.revealed[x][y]
