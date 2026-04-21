import random


class MahjongConnect:
    EMPTY_CELL = ' '
    NEIGHBOR_OFFSETS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    def __init__(self, BOARD_SIZE, ICONS):
        self.BOARD_SIZE = BOARD_SIZE
        self.ICONS = ICONS
        self.board = self.create_board()

    def create_board(self):
        board = [[random.choice(self.ICONS) for _ in range(self.BOARD_SIZE[1])] for _ in range(self.BOARD_SIZE[0])]
        return board

    def is_valid_move(self, pos1, pos2):
        try:
            x1, y1 = pos1
            x2, y2 = pos2
            
            if pos1 == pos2:
                return False
                
            if (x1 < 0 or x1 >= self.BOARD_SIZE[0] or y1 < 0 or y1 >= self.BOARD_SIZE[1] or
                x2 < 0 or x2 >= self.BOARD_SIZE[0] or y2 < 0 or y2 >= self.BOARD_SIZE[1]):
                return False
                
            if self.board[x1][y1] != self.board[x2][y2]:
                return False
                
            return self.has_path(pos1, pos2)
        except (ValueError, IndexError, TypeError):
            return False

    def has_path(self, source, destination):
        explored = set()
        frontier = [source]
        target_symbol = self.board[source[0]][source[1]]
        
        while frontier:
            current = frontier.pop()
            
            if current == destination:
                return True
                
            if current in explored:
                continue
                
            explored.add(current)
            x, y = current
            
            neighbors = [(x + dx, y + dy) for dx, dy in self.NEIGHBOR_OFFSETS]
            valid_neighbors = [
                (nx, ny) for nx, ny in neighbors
                if (0 <= nx < self.BOARD_SIZE[0] and 0 <= ny < self.BOARD_SIZE[1] and
                    (nx, ny) not in explored and
                    self.board[nx][ny] == target_symbol)
            ]
            
            frontier.extend(valid_neighbors)
            
        return False

    def remove_icons(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        self.board[x1][y1] = self.EMPTY_CELL
        self.board[x2][y2] = self.EMPTY_CELL

    def is_game_over(self):
        for i in range(self.BOARD_SIZE[0]):
            for j in range(self.BOARD_SIZE[1]):
                if self.board[i][j] != self.EMPTY_CELL:
                    return False
        return True
