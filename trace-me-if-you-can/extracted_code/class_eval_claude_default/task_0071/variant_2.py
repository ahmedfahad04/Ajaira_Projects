class PushBoxGame:
    def __init__(self, map):
        self.map = map
        self.player_row = 0
        self.player_col = 0
        self.targets = []
        self.boxes = []
        self.is_game_over = False
        self.init_game()

    def init_game(self):
        cell_handlers = {
            "O": self._handle_player,
            "G": self._handle_target,
            "X": self._handle_box
        }
        
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                cell = self.map[row][col]
                if cell in cell_handlers:
                    cell_handlers[cell](row, col)

    def _handle_player(self, row, col):
        self.player_row = row
        self.player_col = col

    def _handle_target(self, row, col):
        self.targets.append((row, col))

    def _handle_box(self, row, col):
        self.boxes.append((row, col))

    def check_win(self):
        boxes_on_targets = sum(1 for box in self.boxes if box in self.targets)
        self.is_game_over = boxes_on_targets == len(self.targets)
        return self.is_game_over

    def move(self, direction):
        delta_row, delta_col = self._get_direction_delta(direction)
        new_player_row = self.player_row + delta_row
        new_player_col = self.player_col + delta_col

        if self._can_move_to(new_player_row, new_player_col):
            self._execute_move(new_player_row, new_player_col, delta_row, delta_col)

        return self.check_win()

    def _get_direction_delta(self, direction):
        direction_map = {"w": (-1, 0), "s": (1, 0), "a": (0, -1), "d": (0, 1)}
        return direction_map.get(direction, (0, 0))

    def _can_move_to(self, row, col):
        return self.map[row][col] != "#"

    def _execute_move(self, new_row, new_col, delta_row, delta_col):
        if (new_row, new_col) in self.boxes:
            self._try_push_box(new_row, new_col, delta_row, delta_col)
        else:
            self._move_player(new_row, new_col)

    def _try_push_box(self, box_row, box_col, delta_row, delta_col):
        new_box_row = box_row + delta_row
        new_box_col = box_col + delta_col
        
        if self._can_move_to(new_box_row, new_box_col):
            self.boxes.remove((box_row, box_col))
            self.boxes.append((new_box_row, new_box_col))
            self._move_player(box_row, box_col)

    def _move_player(self, row, col):
        self.player_row = row
        self.player_col = col
