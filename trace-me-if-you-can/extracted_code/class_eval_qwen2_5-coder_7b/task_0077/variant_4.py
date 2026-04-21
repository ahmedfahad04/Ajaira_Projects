import random

class SnakeBodyPart:
    def __init__(self, x, y, block_size):
        self.x = x
        self.y = y
        self.block_size = block_size

    def relocate(self, dx, dy, screen_width, screen_height):
        self.x = (self.x + dx * self.block_size) % screen_width
        self.y = (self.y + dy * self.block_size) % screen_height

class SnakeGameModel:
    def __init__(self, screen_width, screen_height, block_size, food_position):
        self.head = SnakeBodyPart(screen_width // 2, screen_height // 2, block_size)
        self.body = [self.head]
        self.body_length = 1
        self.score = 0
        self.food = SnakeBodyPart(food_position[0], food_position[1], block_size)

    def process_movement(self, direction):
        dx, dy = direction
        self.head.relocate(dx, dy, self.food.screen_width, self.food.screen_height)

        if self.head.x == self.food.x and self.head.y == self.food.y:
            self.consume_food()

        if any(self.head == part for part in self.body[1:]):
            self.restart_game()
        else:
            new_part = SnakeBodyPart(self.head.x, self.head.y, self.head.block_size)
            self.body.insert(0, new_part)
            if len(self.body) > self.body_length:
                self.body.pop()

    def consume_food(self):
        self.body_length += 1
        self.score += 100
        self.respawn_food()

    def respawn_food(self):
        while any(part == self.food for part in self.body):
            self.food = SnakeBodyPart(random.randint(0, self.food.screen_width // self.food.block_size - 1) * self.food.block_size,
                                   random.randint(0, self.food.screen_height // self.food.block_size - 1) * self.food.block_size,
                                   self.food.block_size)

    def restart_game(self):
        self.body_length = 1
        self.head.x = self.food.screen_width // 2
        self.head.y = self.food.screen_height // 2
        self.body = [self.head]
        self.score = 0
        self.respawn_food()
