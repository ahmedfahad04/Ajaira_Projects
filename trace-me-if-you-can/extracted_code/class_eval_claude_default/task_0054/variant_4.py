import random


class MahjongConnect:
    def __init__(self, BOARD_SIZE, ICONS):
        self.BOARD_SIZE = BOARD_SIZE
        self.ICONS = ICONS
        self.board = self.create_board()

    def create_board(self):
        board = [[random.choice(self.ICONS) for _ in range(self.BOARD_SIZE[1])] for _ in range(self.BOARD_SIZE[0])]
        return board

    def _get_cell_value(self, position):
        x, y = position
        return self.board[x][y]

    def _is_position_in_bounds(self, position):
        x, y = position
        return 0 <= x < self.BOARD_SIZE[0] and 0 <= y < self.BOARD_SIZE[1]

    def _get_adjacent_positions(self, position):
        x, y = position
        return [(x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]]

    def is_valid_move(self, pos1, pos2):
        # Validate positions are in bounds
        if not (self._is_position_in_bounds(pos1) and self._is_position_in_bounds(pos2)):
            return False

        # Validate positions are different
        if pos1 == pos2:
            return False

        # Validate icons match
        if self._get_cell_value(pos1) != self._get_cell_value(pos2):
            return False

        # Validate path exists
        return self.has_path(pos1, pos2)

    def has_path(self, start_pos, end_pos):
        if start_pos == end_pos:
            return True
            
        target_icon = self._get_cell_value(start_pos)
        visited_positions = set()
        positions_to_check = [start_pos]
        
        while positions_to_check:
            current_pos = positions_to_check.pop()
            
            if current_pos in visited_positions:
                continue
                
            visited_positions.add(current_pos)
            
            if current_pos == end_pos:
                return True
            
            adjacent_positions = self._get_adjacent_positions(current_pos)
            
            for adj_pos in adjacent_positions:
                if (self._is_position_in_bounds(adj_pos) and 
                    adj_pos not in visited_positions and
                    self._get_cell_value(adj_pos) == target_icon):
                    positions_to_check.append(adj_pos)
        
        return False

    def remove_icons(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        self.board[x1][y1] = ' '
        self.board[x2][y2] = ' '

    def is_game_over(self):
        remaining_icons = sum(1 for row in self.board for icon in row if icon != ' ')
        return remaining_icons == 0
