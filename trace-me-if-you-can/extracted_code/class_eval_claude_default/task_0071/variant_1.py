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
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                cell = self.map[row][col]
                if cell == "O":
                    self.player_pos = (row, col)
                elif cell == "G":
                    self.targets.add((row, col))
                elif cell == "X":
                    self.boxes.add((row, col))

    def check_win(self):
        self.is_game_over = self.boxes <= self.targets
        return self.is_game_over

    def move(self, direction):
        if direction not in self.DIRECTIONS:
            return self.check_win()
            
        dr, dc = self.DIRECTIONS[direction]
        new_pos = (self.player_pos[0] + dr, self.player_pos[1] + dc)
        
        if self._is_wall(new_pos):
            return self.check_win()
            
        if new_pos in self.boxes:
            box_new_pos = (new_pos[0] + dr, new_pos[1] + dc)
            if not self._is_wall(box_new_pos):
                self.boxes.remove(new_pos)
                self.boxes.add(box_new_pos)
                self.player_pos = new_pos
        else:
            self.player_pos = new_pos
            
        return self.check_win()
    
    def _is_wall(self, pos):
        return self.map[pos[0]][pos[1]] == "#"
