class PushBoxGame:
    DIRECTIONS = {
        "w": (-1, 0),
        "s": (1, 0),
        "a": (0, -1),
        "d": (0, 1)
    }
    
    def __init__(self, map):
        self.map = map
        self.player_pos = None
        self.targets = set()
        self.boxes = set()
        self.is_game_over = False
        self._parse_map()

    def _parse_map(self):
        for row_idx, row in enumerate(self.map):
            for col_idx, cell in enumerate(row):
                pos = (row_idx, col_idx)
                if cell == "O":
                    self.player_pos = pos
                elif cell == "G":
                    self.targets.add(pos)
                elif cell == "X":
                    self.boxes.add(pos)

    def check_win(self):
        self.is_game_over = self.targets.issubset(self.boxes)
        return self.is_game_over

    def move(self, direction):
        if direction not in self.DIRECTIONS:
            return self.check_win()
            
        dr, dc = self.DIRECTIONS[direction]
        curr_row, curr_col = self.player_pos
        new_pos = (curr_row + dr, curr_col + dc)
        
        if not self._is_valid_position(new_pos):
            return self.check_win()
            
        if new_pos in self.boxes:
            box_new_pos = (new_pos[0] + dr, new_pos[1] + dc)
            if self._is_valid_position(box_new_pos):
                self.boxes.remove(new_pos)
                self.boxes.add(box_new_pos)
                self.player_pos = new_pos
        else:
            self.player_pos = new_pos
            
        return self.check_win()
    
    def _is_valid_position(self, pos):
        row, col = pos
        return (0 <= row < len(self.map) and 
                0 <= col < len(self.map[0]) and 
                self.map[row][col] != "#")
