class PushBoxGame:
    def __init__(self, map):
        self.map = map
        self.game_state = self._initialize_game_state()

    def _initialize_game_state(self):
        state = {
            'player': None,
            'targets': [],
            'boxes': [],
            'game_over': False
        }
        
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                cell_type = self.map[row][col]
                position = (row, col)
                
                if cell_type == "O":
                    state['player'] = position
                elif cell_type == "G":
                    state['targets'].append(position)
                elif cell_type == "X":
                    state['boxes'].append(position)
        
        return state

    @property
    def player_row(self):
        return self.game_state['player'][0]
    
    @property
    def player_col(self):
        return self.game_state['player'][1]
    
    @property
    def targets(self):
        return self.game_state['targets']
    
    @property
    def boxes(self):
        return self.game_state['boxes']
    
    @property
    def target_count(self):
        return len(self.game_state['targets'])
    
    @property
    def is_game_over(self):
        return self.game_state['game_over']

    def init_game(self):
        pass  # State already initialized in constructor

    def check_win(self):
        completed_targets = len([box for box in self.game_state['boxes'] if box in self.game_state['targets']])
        self.game_state['game_over'] = completed_targets == len(self.game_state['targets'])
        return self.game_state['game_over']

    def move(self, direction):
        movement_vectors = {
            "w": lambda pos: (pos[0] - 1, pos[1]),
            "s": lambda pos: (pos[0] + 1, pos[1]),
            "a": lambda pos: (pos[0], pos[1] - 1),
            "d": lambda pos: (pos[0], pos[1] + 1)
        }
        
        if direction not in movement_vectors:
            return self.check_win()
        
        current_player_pos = self.game_state['player']
        new_player_pos = movement_vectors[direction](current_player_pos)
        
        if self._is_blocked(new_player_pos):
            return self.check_win()
        
        if self._has_box_at(new_player_pos):
            new_box_pos = movement_vectors[direction](new_player_pos)
            if not self._is_blocked(new_box_pos):
                self._move_box(new_player_pos, new_box_pos)
                self.game_state['player'] = new_player_pos
        else:
            self.game_state['player'] = new_player_pos
        
        return self.check_win()

    def _is_blocked(self, position):
        return self.map[position[0]][position[1]] == "#"
    
    def _has_box_at(self, position):
        return position in self.game_state['boxes']
    
    def _move_box(self, from_pos, to_pos):
        self.game_state['boxes'].remove(from_pos)
        self.game_state['boxes'].append(to_pos)
