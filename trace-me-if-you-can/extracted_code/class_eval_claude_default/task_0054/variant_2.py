import random


class MahjongConnect:
    def __init__(self, BOARD_SIZE, ICONS):
        self.BOARD_SIZE = BOARD_SIZE
        self.ICONS = ICONS
        self.board = self.create_board()
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def create_board(self):
        board = [[random.choice(self.ICONS) for _ in range(self.BOARD_SIZE[1])] for _ in range(self.BOARD_SIZE[0])]
        return board

    def _within_bounds(self, x, y):
        return 0 <= x < self.BOARD_SIZE[0] and 0 <= y < self.BOARD_SIZE[1]

    def is_valid_move(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2

        conditions = [
            self._within_bounds(x1, y1) and self._within_bounds(x2, y2),
            pos1 != pos2,
            self.board[x1][y1] == self.board[x2][y2],
            self.has_path(pos1, pos2)
        ]
        
        return all(conditions)

    def has_path(self, start, target):
        def dfs(current, visited_set):
            if current == target:
                return True
            if current in visited_set:
                return False
            
            visited_set.add(current)
            x, y = current
            icon = self.board[x][y]
            
            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                if (self._within_bounds(nx, ny) and 
                    (nx, ny) not in visited_set and 
                    self.board[nx][ny] == icon):
                    if dfs((nx, ny), visited_set):
                        return True
            
            return False
        
        return dfs(start, set())

    def remove_icons(self, pos1, pos2):
        positions = [pos1, pos2]
        for x, y in positions:
            self.board[x][y] = ' '

    def is_game_over(self):
        return not any(cell != ' ' for row in self.board for cell in row)
