class PushBoxGame:
    def __init__(self, level_map):
        self.level_map = level_map
        self.player_position = (0, 0)
        self.goal_positions = []
        self.block_positions = []
        self.goal_count = 0
        self.game_finished = False

        self.initialize_game()

    def initialize_game(self):
        for row_idx, row in enumerate(self.level_map):
            for col_idx, cell in enumerate(row):
                if cell == "O":
                    self.player_position = (row_idx, col_idx)
                elif cell == "G":
                    self.goal_positions.append((row_idx, col_idx))
                    self.goal_count += 1
                elif cell == "X":
                    self.block_positions.append((row_idx, col_idx))

    def is_win_condition_met(self):
        blocks_on_goals = 0
        for block in self.block_positions:
            if block in self.goal_positions:
                blocks_on_goals += 1
        if blocks_on_goals == self.goal_count:
            self.game_finished = True
        return self.game_finished

    def attempt_move(self, movement):
        new_player_y = self.player_position[0]
        new_player_x = self.player_position[1]

        if movement == "w":
            new_player_y -= 1
        elif movement == "s":
            new_player_y += 1
        elif movement == "a":
            new_player_x -= 1
        elif movement == "d":
            new_player_x += 1

        if self.level_map[new_player_y][new_player_x] != "#":
            if (new_player_y, new_player_x) in self.block_positions:
                new_block_y = new_player_y + (new_player_y - self.player_position[0])
                new_block_x = new_player_x + (new_player_x - self.player_position[1])

                if self.level_map[new_block_y][new_block_x] != "#":
                    self.block_positions.remove((new_player_y, new_player_x))
                    self.block_positions.append((new_block_y, new_block_x))
                    self.player_position = (new_player_y, new_player_x)
            else:
                self.player_position = (new_player_y, new_player_x)

        return self.is_win_condition_met()
