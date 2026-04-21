from dataclasses import dataclass
from typing import List, Tuple, Set

@dataclass
class GameState:
    player_pos: Tuple[int, int]
    boxes: Set[Tuple[int, int]]
    targets: Set[Tuple[int, int]]
    is_game_over: bool = False

class PushBoxGame:
    def __init__(self, map):
        self.map = map
        self.state = self._create_initial_state()

    def _create_initial_state(self) -> GameState:
        player_pos = None
        boxes = set()
        targets = set()
        
        for row_idx, row in enumerate(self.map):
            for col_idx, cell in enumerate(row):
                pos = (row_idx, col_idx)
                if cell == "O":
                    player_pos = pos
                elif cell == "G":
                    targets.add(pos)
                elif cell == "X":
                    boxes.add(pos)
        
        return GameState(player_pos, boxes, targets)

    @property
    def player_row(self):
        return self.state.player_pos[0]
    
    @property
    def player_col(self):
        return self.state.player_pos[1]
    
    @property
    def targets(self):
        return list(self.state.targets)
    
    @property
    def boxes(self):
        return list(self.state.boxes)
    
    @property
    def target_count(self):
        return len(self.state.targets)
    
    @property
    def is_game_over(self):
        return self.state.is_game_over

    def init_game(self):
        pass  # Already handled in constructor

    def check_win(self):
        self.state.is_game_over = self.state.boxes.issubset(self.state.targets) and len(self.state.boxes) == len(self.state.targets)
        return self.state.is_game_over

    def move(self, direction):
        direction_vectors = {"w": (-1, 0), "s": (1, 0), "a": (0, -1), "d": (0, 1)}
        
        if direction not in direction_vectors:
            return self.check_win()
        
        dr, dc = direction_vectors[direction]
        current_pos = self.state.player_pos
        new_pos = (current_pos[0] + dr, current_pos[1] + dc)
        
        if self.map[new_pos[0]][new_pos[1]] == "#":
            return self.check_win()
        
        if new_pos in self.state.boxes:
            box_new_pos = (new_pos[0] + dr, new_pos[1] + dc)
            if self.map[box_new_pos[0]][box_new_pos[1]] != "#":
                self.state.boxes.remove(new_pos)
                self.state.boxes.add(box_new_pos)
                self.state.player_pos = new_pos
        else:
            self.state.player_pos = new_pos
        
        return self.check_win()
