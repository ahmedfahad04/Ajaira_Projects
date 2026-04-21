class BoxPuzzle:
    def __init__(self, field):
        self.field = field
        self.player_i = 0
        self.player_j = 0
        self.markers = []
        self.obstacles = []
        self.marker_count = 0
        self.game_over = False

        self.configure_game()

    def configure_game(self):
        for row_idx, row in enumerate(self.field):
            for col_idx, element in enumerate(row):
                if element == "P":
                    self.player_i = row_idx
                    self.player_j = col_idx
                elif element == "T":
                    self.markers.append((row_idx, col_idx))
                    self.marker_count += 1
                elif element == "W":
                    self.obstacles.append((row_idx, col_idx))

    def assess_victory(self):
        box_on_marker_count = 0
        for marker in self.markers:
            if marker in self.field:
                box_on_marker_count += 1
        if box_on_marker_count == self.marker_count:
            self.game_over = True
        return self.game_over

    def undertake_movement(self, command):
        new_player_i = self.player_i
        new_player_j = self.player_j

        if command == "north":
            new_player_i -= 1
        elif command == "south":
            new_player_i += 1
        elif command == "west":
            new_player_j -= 1
        elif command == "east":
            new_player_j += 1

        if self.field[new_player_i][new_player_j] != "B":
            if (new_player_i, new_player_j) in self.markers:
                new_marker_i = new_player_i + (new_player_i - self.player_i)
                new_marker_j = new_player_j + (new_player_j - self.player_j)

                if self.field[new_marker_i][new_marker_j] != "B":
                    self.markers.remove((new_player_i, new_player_j))
                    self.markers.append((new_marker_i, new_marker_j))
                    self.player_i = new_player_i
                    self.player_j = new_player_j
            else:
                self.player_i = new_player_i
                self.player_j = new_player_j

        return self.assess_victory()
