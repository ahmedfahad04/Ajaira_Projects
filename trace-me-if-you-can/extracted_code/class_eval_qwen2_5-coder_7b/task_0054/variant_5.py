import random

class MahjongGame:
    def __init__(self, board_size, icons):
        self.size = board_size
        self.icons = icons
        self.tile_board = self.setup_board()

    def setup_board(self):
        return [[random.choice(self.icons) for _ in range(self.size[1])] for _ in range(self.size[0])]

    def is_move_valid(self, tile1, tile2):
        x1, y1 = tile1
        x2, y2 = tile2

        if not (0 <= x1 < self.size[0] and 0 <= y1 < self.size[1] and 0 <= x2 < self.size[0] and 0 <= y2 < self.size[1]):
            return False

        if tile1 == tile2:
            return False

        if self.tile_board[x1][y1] != self.tile_board[x2][y2]:
            return False

        return self.is_connected(tile1, tile2)

    def is_connected(self, start, end):
        stack = [start]
        visited = set()

        while stack:
            current_pos = stack.pop()
            if current_pos == end:
                return True

            if current_pos in visited:
                continue

            visited.add(current_pos)
            x, y = current_pos

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.size[0] and 0 <= new_y < self.size[1]:
                    if (new_x, new_y) not in visited and self.tile_board[new_x][new_y] == self.tile_board[x][y]:
                        stack.append((new_x, new_y))

        return False

    def remove_tiles(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        self.tile_board[x1][y1] = ' '
        self.tile_board[x2][y2] = ' '

    def is_all_tiles_eliminated(self):
        return all(icon == ' ' for row in self.tile_board for icon in row)
