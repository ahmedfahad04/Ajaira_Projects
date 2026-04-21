import random

class SnakeGame:
    def __init__(self, width, height, block_size, initial_food):
        self.snake_length = 1
        self.screen_width = width
        self.screen_height = height
        self.block_size = block_size
        self.snake_positions = [(width // 2, height // 2)]
        self.score = 0
        self.food_position = initial_food

    def change_direction(self, dx, dy):
        new_head = ((self.snake_positions[0][0] + (dx * self.block_size)) % self.screen_width,
                    (self.snake_positions[0][1] + (dy * self.block_size)) % self.screen_height)

        if new_head == self.food_position:
            self.snake_grows()

        if len(self.snake_positions) > 2 and new_head in self.snake_positions[2:]:
            self.reset_game()
        else:
            self.snake_positions.insert(0, new_head)
            if len(self.snake_positions) > self.snake_length:
                self.snake_positions.pop()

    def place_random_food(self):
        while self.food_position in self.snake_positions:
            self.food_position = (random.randint(0, self.screen_width // self.block_size - 1) * self.block_size,
                                  random.randint(0, self.screen_height // self.block_size - 1) * self.block_size)

    def reset_game(self):
        self.snake_length = 1
        self.snake_positions = [(self.screen_width // 2, self.screen_height // 2)]
        self.score = 0
        self.place_random_food()

    def snake_grows(self):
        self.snake_length += 1
        self.score += 100
        self.place_random_food()
