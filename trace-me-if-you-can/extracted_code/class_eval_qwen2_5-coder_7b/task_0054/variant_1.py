import random

class MahjongBoard:
    def __init__(self, board_size, icons):
        self.board_size = board_size
        self.icons = icons
        self.tile_board = self.initialize_board()

    def initialize_board(self):
        return [[random.choice(self.icons) for _ in range(self.board_size[1])] for _ in range(self.board_size[0])]

    def validate_move(self, tile1, tile2):
        x1, y1 = tile1
        x2, y2 = tile2

        if not (0 <= x1 < self.board_size[0] and 0 <= y1 < self.board_size[1] and 0 <= x2 < self.board_size[0] and 0 <= y2 < self.board_size[1]):
            return False

        if tile1 == tile2:
            return False

        if self.tile_board[x1][y1] != self.tile_board[x2][y2]:
            return False

        if not self.check_path(tile1, tile2):
            return False

        return True

    def check_path(self, tile1, tile2):
        visited = set()
        stack = [tile1]

        while stack:
            current_tile = stack.pop()
            if current_tile == tile2:
                return True

            if current_tile in visited:
                continue

            visited.add(current_tile)
            x, y = current_tile

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.board_size[0] and 0 <= new_y < self.board_size[1]:
                    if (new_x, new_y) not in visited and self.tile_board[new_x][new_y] == self.tile_board[x][y]:
                        stack.append((new_x, new_y))

        return False

    def eliminate_tiles(self, tile1, tile2):
        x1, y1 = tile1
        x2, y2 = tile2
        self.tile_board[x1][y1] = ' '
        self.tile_board[x2][y2] = ' '

    def check_game_over(self):
        return all(icon == ' ' for row in self.tile_board for icon in row)
