import random

class Tile:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return str(self.value)

class MahjongTileGame:
    def __init__(self, board_size, icons):
        self.board_size = board_size
        self.icons = icons
        self.tile_grid = self.generate_board()

    def generate_board(self):
        return [[Tile(random.choice(self.icons)) for _ in range(self.board_size[1])] for _ in range(self.board_size[0])]

    def is_move_valid(self, tile1, tile2):
        x1, y1 = tile1
        x2, y2 = tile2

        if not (0 <= x1 < self.board_size[0] and 0 <= y1 < self.board_size[1] and 0 <= x2 < self.board_size[0] and 0 <= y2 < self.board_size[1]):
            return False

        if tile1 == tile2:
            return False

        if self.tile_grid[x1][y1] != self.tile_grid[x2][y2]:
            return False

        return self.has_path(tile1, tile2)

    def has_path(self, pos1, pos2):
        visited = set()
        stack = [pos1]

        while stack:
            current_pos = stack.pop()
            if current_pos == pos2:
                return True

            if current_pos in visited:
                continue

            visited.add(current_pos)
            x, y = current_pos

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.board_size[0] and 0 <= new_y < self.board_size[1]:
                    if (new_x, new_y) not in visited and self.tile_grid[new_x][new_y] == self.tile_grid[x][y]:
                        stack.append((new_x, new_y))

        return False

    def remove_tiles(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        self.tile_grid[x1][y1].value = ' '
        self.tile_grid[x2][y2].value = ' '

    def is_game_finished(self):
        return all(tile.value == ' ' for row in self.tile_grid for tile in row)
