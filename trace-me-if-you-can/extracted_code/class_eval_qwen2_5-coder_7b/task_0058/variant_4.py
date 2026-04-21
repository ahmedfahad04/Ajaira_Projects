import random

class MinesweeperGame:
    def __init__(self, board_size, mine_count):
        self.board_size = board_size
        self.mine_count = mine_count
        self.mine_map = self.generate_mine_map()
        self.player_map = self.create_player_map()
        self.game_over = False
        self.score = 0

    def generate_mine_map(self):
        mine_map = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        mines_placed = 0
        while mines_placed < self.mine_count:
            x, y = random.randint(0, self.board_size - 1), random.randint(0, self.board_size - 1)
            if mine_map[y][x] != 'X':
                mine_map[y][x] = 'X'
                mines_placed += 1
                for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size and mine_map[ny][nx] != 'X':
                        mine_map[ny][nx] += 1
        return mine_map

    def create_player_map(self):
        return [['-' for _ in range(self.board_size)] for _ in range(self.board_size)]

    def check_win(self):
        for row in self.player_map:
            if '-' in row:
                return False
        return True

    def uncover(self, x, y):
        if self.mine_map[x][y] == 'X':
            self.game_over = True
            return False
        else:
            self.player_map[x][y] = self.mine_map[x][y]
            self.score += 1
            if self.check_win():
                self.game_over = True
                return True
            return self.player_map[x][y]
