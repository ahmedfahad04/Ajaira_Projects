import random
from functools import reduce
from operator import and_


class MahjongConnect:
    def __init__(self, BOARD_SIZE, ICONS):
        self.BOARD_SIZE = BOARD_SIZE
        self.ICONS = ICONS
        self.board = self.create_board()

    def create_board(self):
        board = [[random.choice(self.ICONS) for _ in range(self.BOARD_SIZE[1])] for _ in range(self.BOARD_SIZE[0])]
        return board

    def is_valid_move(self, pos1, pos2):
        validation_checks = [
            lambda: self._validate_bounds(pos1, pos2),
            lambda: self._validate_different_positions(pos1, pos2),
            lambda: self._validate_matching_icons(pos1, pos2),
            lambda: self.has_path(pos1, pos2)
        ]
        
        return reduce(and_, (check() for check in validation_checks), True)

    def _validate_bounds(self, pos1, pos2):
        positions = [pos1, pos2]
        bounds_check = lambda pos: (0 <= pos[0] < self.BOARD_SIZE[0] and 
                                   0 <= pos[1] < self.BOARD_SIZE[1])
        return all(map(bounds_check, positions))

    def _validate_different_positions(self, pos1, pos2):
        return pos1 != pos2

    def _validate_matching_icons(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return self.board[x1][y1] == self.board[x2][y2]

    def has_path(self, start, target):
        path_finder = PathFinder(self.board, self.BOARD_SIZE, start, target)
        return path_finder.find_connection()

    def remove_icons(self, pos1, pos2):
        removal_positions = [pos1, pos2]
        for x, y in removal_positions:
            self.board[x][y] = ' '

    def is_game_over(self):
        cell_checker = lambda cell: cell == ' '
        row_checker = lambda row: all(map(cell_checker, row))
        return all(map(row_checker, self.board))


class PathFinder:
    def __init__(self, board, board_size, start, target):
        self.board = board
        self.board_size = board_size
        self.start = start
        self.target = target
        self.target_icon = board[start[0]][start[1]]
        
    def find_connection(self):
        visited = set()
        stack = [self.start]
        
        while stack:
            current = stack.pop()
            if current == self.target:
                return True
            if current in visited:
                continue
                
            visited.add(current)
            stack.extend(self._get_valid_neighbors(current, visited))
        
        return False
    
    def _get_valid_neighbors(self, position, visited):
        x, y = position
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbors = []
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.board_size[0] and 0 <= ny < self.board_size[1] and
                (nx, ny) not in visited and 
                self.board[nx][ny] == self.target_icon):
                neighbors.append((nx, ny))
        
        return neighbors
