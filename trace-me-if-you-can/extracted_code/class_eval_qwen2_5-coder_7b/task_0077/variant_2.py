import random

class SnakeSegment:
    def __init__(self, x, y, block_size):
        self.x = x
        self.y = y
        self.block_size = block_size

    def move(self, dx, dy, screen_width, screen_height):
        self.x = (self.x + dx * self.block_size) % screen_width
        self.y = (self.y + dy * self.block_size) % screen_height

class Snake:
    def __init__(self, screen_width, screen_height, block_size, food_position):
        self.head = SnakeSegment(screen_width // 2, screen_height // 2, block_size)
        self.length = 1
        self.positions = [self.head]
        self.score = 0
        self.food = SnakeSegment(food_position[0], food_position[1], block_size)

    def move(self, direction):
        dx, dy = direction
        self.head.move(dx, dy, self.food.screen_width, self.food.screen_height)

        if self.head.x == self.food.x and self.head.y == self.food.y:
            self.eat_food()

        if len(self.positions) > 2 and self.head in self.positions[2:]:
            self.reset()
        else:
            new_segment = SnakeSegment(self.head.x, self.head.y, self.head.block_size)
            self.positions.insert(0, new_segment)
            if len(self.positions) > self.length:
                self.positions.pop()

    def eat_food(self):
        self.length += 1
        self.score += 100
        self.place_food()

    def place_food(self):
        while any(segment.x == self.food.x and segment.y == self.food.y for segment in self.positions):
            self.food.x = random.randint(0, self.food.screen_width // self.food.block_size - 1) * self.food.block_size
            self.food.y = random.randint(0, self.food.screen_height // self.food.block_size - 1) * self.food.block_size

    def reset(self):
        self.length = 1
        self.head.x = self.food.screen_width // 2
        self.head.y = self.food.screen_height // 2
        self.positions = [self.head]
        self.score = 0
        self.place_food()
