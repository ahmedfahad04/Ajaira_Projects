class SlideGame:
    def __init__(self, matrix):
        self.matrix = matrix
        self.player_i = 0
        self.player_j = 0
        self.markers = []
        self.blockers = []
        self.marker_count = 0
        self.is_complete = False

        self.setup_game()

    def setup_game(self):
        for row_idx, row in enumerate(self.matrix):
            for col_idx, cell in enumerate(row):
                if cell == "I":
                    self.player_i = row_idx
                    self.player_j = col_idx
                elif cell == "F":
                    self.markers.append((row_idx, col_idx))
                    self.marker_count += 1
                elif cell == "X":
                    self.blockers.append((row_idx, col_idx))

    def check_win(self):
        marker_on_target_count = 0
        for marker in self.markers:
            if marker in self.matrix:
                marker_on_target_count += 1
        if marker_on_target_count == self.marker_count:
            self.is_complete = True
        return self.is_complete

    def move_piece(self, direction):
        new_player_i = self.player_i
        new_player_j = self.player_j

        if direction == "north":
            new_player_i -= 1
        elif direction == "south":
            new_player_i += 1
        elif direction == "west":
            new_player_j -= 1
        elif direction == "east":
            new_player_j += 1

        if self.matrix[new_player_i][new_player_j] != "B":
            if (new_player_i, new_player_j) in self.markers:
                new_marker_i = new_player_i + (new_player_i - self.player_i)
                new_marker_j = new_player_j + (new_player_j - self.player_j)

                if self.matrix[new_marker_i][new_marker_j] != "B":
                    self.markers.remove((new_player_i, new_player_j))
                    self.markers.append((new_marker_i, new_marker_j))
                    self.player_i = new_player_i
                    self.player_j = new_player_j
            else:
                self.player_i = new_player_i
                self.player_j = new_player_j

        return self.check_win()
