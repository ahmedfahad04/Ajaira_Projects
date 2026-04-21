import random
from collections import deque


class MahjongConnect:
    def __init__(self, BOARD_SIZE, ICONS):
        self.rows, self.cols = BOARD_SIZE
        self.ICONS = ICONS
        self.board = self._initialize_board()

    def _initialize_board(self):
        return [[random.choice(self.ICONS) for _ in range(self.cols)] for _ in range(self.rows)]

    def is_valid_move(self, pos1, pos2):
        if not self._are_positions_valid(pos1, pos2):
            return False
        if not self._have_matching_icons(pos1, pos2):
            return False
        return self._can_connect(pos1, pos2)

    def _are_positions_valid(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return (pos1 != pos2 and 
                0 <= x1 < self.rows and 0 <= y1 < self.cols and
                0 <= x2 < self.rows and 0 <= y2 < self.cols)

    def _have_matching_icons(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return self.board[x1][y1] == self.board[x2][y2]

    def _can_connect(self, start, end):
        queue = deque([start])
        visited = {start}
        target_icon = self.board[start[0]][start[1]]
        
        while queue:
            x, y = queue.popleft()
            if (x, y) == end:
                return True
                
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.rows and 0 <= ny < self.cols and
                    (nx, ny) not in visited and 
                    self.board[nx][ny] == target_icon):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return False

    has_path = _can_connect

    def remove_icons(self, pos1, pos2):
        for x, y in [pos1, pos2]:
            self.board[x][y] = ' '

    def is_game_over(self):
        return all(icon == ' ' for row in self.board for icon in row)
