import random

class MinesweeperGame:
    def __init__(self, board_size, mine_count):
        self.board_size = board_size
        self.mine_count = mine_count
        self.mine_field = self.initialize_mine_field()
        self.revealed_field = [['-' for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.game_status = 'ongoing'
        self.score = 0

    def initialize_mine_field(self):
        mine_field = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        for _ in range(self.mine_count):
            while True:
                x, y = random.randint(0, self.board_size - 1), random.randint(0, self.board_size - 1)
                if mine_field[y][x] == 0:
                    mine_field[y][x] = 'X'
                    break
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and mine_field[ny][nx] != 'X':
                    mine_field[ny][nx] += 1
        return mine_field

    def initialize_revealed_field(self):
        return [['-' for _ in range(self.board_size)] for _ in range(self.board_size)]

    def has_won(self):
        for row in self.revealed_field:
            if '-' in row:
                return False
        return True

    def uncover_cell(self, row, col):
        if self.mine_field[row][col] == 'X':
            self.game_status = 'lost'
            return False
        else:
            self.revealed_field[row][col] = self.mine_field[row][col]
            self.score += 1
            if self.has_won():
                self.game_status = 'won'
                return True
            return self.revealed_field[row][col]
