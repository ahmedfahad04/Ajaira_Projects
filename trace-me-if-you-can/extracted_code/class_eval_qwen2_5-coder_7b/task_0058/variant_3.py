import random

class MinefieldGame:
    def __init__(self, dimension, mine_number):
        self.dimension = dimension
        self.mine_number = mine_number
        self.mine_positions = self.place_mines()
        self.player_view = [['-' for _ in range(self.dimension)] for _ in range(self.dimension)]
        self.game_ended = False
        self.point_counter = 0

    def place_mines(self):
        mine_positions = []
        while len(mine_positions) < self.mine_number:
            x, y = random.randint(0, self.dimension - 1), random.randint(0, self.dimension - 1)
            if (x, y) not in mine_positions:
                mine_positions.append((x, y))
                for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.dimension and 0 <= ny < self.dimension and (nx, ny) not in mine_positions:
                        mine_positions.append((nx, ny))
        return mine_positions

    def initialize_player_view(self):
        return [['-' for _ in range(self.dimension)] for _ in range(self.dimension)]

    def is_game_over(self):
        for row in self.player_view:
            if '-' in row:
                return False
        return True

    def reveal_cell(self, x, y):
        if (x, y) in self.mine_positions:
            self.game_ended = True
            return False
        else:
            self.player_view[x][y] = self.count_adjacent_mines(x, y)
            self.point_counter += 1
            if self.is_game_over():
                self.game_ended = True
                return True
            return self.player_view[x][y]

    def count_adjacent_mines(self, x, y):
        count = 0
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.dimension and 0 <= ny < self.dimension and (nx, ny) in self.mine_positions:
                count += 1
        return count
