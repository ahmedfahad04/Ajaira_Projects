import random

class MahjongGame:
    def __init__(self, board_size, icons):
        self.size = board_size
        self.icons = icons
        self.game_board = self.setup_board()

    def setup_board(self):
        return [[random.choice(self.icons) for _ in range(self.size[1])] for _ in range(self.size[0])]

    def is_valid_transition(self, position1, position2):
        x1, y1 = position1
        x2, y2 = position2

        if not (0 <= x1 < self.size[0] and 0 <= y1 < self.size[1] and 0 <= x2 < self.size[0] and 0 <= y2 < self.size[1]):
            return False

        if position1 == position2:
            return False

        if self.game_board[x1][y1] != self.game_board[x2][y2]:
            return False

        return self.find_path(position1, position2)

    def find_path(self, pos1, pos2):
        stack = [pos1]
        visited = set()

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
                if 0 <= new_x < self.size[0] and 0 <= new_y < self.size[1]:
                    if (new_x, new_y) not in visited and self.game_board[new_x][new_y] == self.game_board[x][y]:
                        stack.append((new_x, new_y))

        return False

    def remove_tiles(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        self.game_board[x1][y1] = ' '
        self.game_board[x2][y2] = ' '

    def is_all_tiles_eliminated(self):
        return all(icon == ' ' for row in self.game_board for icon in row)
