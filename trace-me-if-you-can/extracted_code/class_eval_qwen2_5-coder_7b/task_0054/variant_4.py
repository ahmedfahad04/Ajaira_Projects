import numpy as np
import random

class MahjongBoard:
    def __init__(self, board_size, icons):
        self.board_size = board_size
        self.icons = icons
        self.tile_matrix = self.create_tile_board()

    def create_tile_board(self):
        return np.array([[random.choice(self.icons) for _ in range(self.board_size[1])] for _ in range(self.board_size[0])])

    def is_move_valid(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2

        if not (0 <= x1 < self.board_size[0] and 0 <= y1 < self.board_size[1] and 0 <= x2 < self.board_size[0] and 0 <= y2 < self.board_size[1]):
            return False

        if pos1 == pos2:
            return False

        if self.tile_matrix[x1, y1] != self.tile_matrix[x2, y2]:
            return False

        return self.has_path(pos1, pos2)

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
                    if (new_x, new_y) not in visited and self.tile_matrix[new_x, new_y] == self.tile_matrix[x, y]:
                        stack.append((new_x, new_y))

        return False

    def remove_tiles(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        self.tile_matrix[x1, y1] = ' '
        self.tile_matrix[x2, y2] = ' '

    def is_game_over(self):
        return np.all(self.tile_matrix == ' ')
