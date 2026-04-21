import random

class SnakePart:
    def __init__(self, x, y, block_size):
        self.x = x
        self.y = y
        self.block_size = block_size

    def update_position(self, dx, dy, screen_width, screen_height):
        self.x = (self.x + dx * self.block_size) % screen_width
        self.y = (self.y + dy * self.block_size) % screen_height

class SnakeGame:
    def __init__(self, screen_width, screen_height, block_size, food_position):
        self.snake_head = SnakePart(screen_width // 2, screen_height // 2, block_size)
        self.snake_body = [self.snake_head]
        self.snake_length = 1
        self.score = 0
        self.food = SnakePart(food_position[0], food_position[1], block_size)

    def update(self, direction):
        dx, dy = direction
        self.snake_head.update_position(dx, dy, self.food.screen_width, self.food.screen_height)

        if self.snake_head.x == self.food.x and self.snake_head.y == self.food.y:
            self.eat_food()

        if any(self.snake_head == part for part in self.snake_body[1:]):
            self.reset()
        else:
            self.snake_body.insert(0, SnakePart(self.snake_head.x, self.snake_head.y, self.snake_head.block_size))
            if len(self.snake_body) > self.snake_length:
                self.snake_body.pop()

    def eat_food(self):
        self.snake_length += 1
        self.score += 100
        self.place_food()

    def place_food(self):
        while any(part == self.food for part in self.snake_body):
            self.food = SnakePart(random.randint(0, self.food.screen_width // self.food.block_size - 1) * self.food.block_size,
                                   random.randint(0, self.food.screen_height // self.food.block_size - 1) * self.food.block_size,
                                   self.food.block_size)

    def reset(self):
        self.snake_length = 1
        self.snake_head.x = self.food.screen_width // 2
        self.snake_head.y = self.food.screen_height // 2
        self.snake_body = [self.snake_head]
        self.score = 0
        self.place_food()
