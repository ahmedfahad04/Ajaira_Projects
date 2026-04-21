class MazeGame:
    def __init__(self, layout):
        self.layout = layout
        self.player_x = 0
        self.player_y = 0
        self.goals = []
        self.blockers = []
        self.goal_count = 0
        self.is_completed = False

        self.setup_game()

    def setup_game(self):
        for row_idx, row in enumerate(self.layout):
            for col_idx, cell in enumerate(row):
                if cell == "S":
                    self.player_x = row_idx
                    self.player_y = col_idx
                elif cell == "F":
                    self.goals.append((row_idx, col_idx))
                    self.goal_count += 1
                elif cell == "X":
                    self.blockers.append((row_idx, col_idx))

    def evaluate_win(self):
        box_at_goal_count = 0
        for block in self.blockers:
            if block in self.goals:
                box_at_goal_count += 1
        if box_at_goal_count == self.goal_count:
            self.is_completed = True
        return self.is_completed

    def execute_move(self, command):
        new_player_x = self.player_x
        new_player_y = self.player_y

        if command == "up":
            new_player_x -= 1
        elif command == "down":
            new_player_x += 1
        elif command == "left":
            new_player_y -= 1
        elif command == "right":
            new_player_y += 1

        if self.layout[new_player_x][new_player_y] != "W":
            if (new_player_x, new_player_y) in self.blockers:
                new_block_x = new_player_x + (new_player_x - self.player_x)
                new_block_y = new_player_y + (new_player_y - self.player_y)

                if self.layout[new_block_x][new_block_y] != "W":
                    self.blockers.remove((new_player_x, new_player_y))
                    self.blockers.append((new_block_x, new_block_y))
                    self.player_x = new_player_x
                    self.player_y = new_player_y
            else:
                self.player_x = new_player_x
                self.player_y = new_player_y

        return self.evaluate_win()
