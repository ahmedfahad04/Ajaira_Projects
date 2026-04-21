import random


class SnakePosition:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if isinstance(other, tuple):
            return (self.x, self.y) == other
        return self.x == other.x and self.y == other.y
    
    def __iter__(self):
        yield self.x
        yield self.y
    
    def to_tuple(self):
        return (self.x, self.y)


class Snake:
    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, food_position):
        self.length = 1
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.BLOCK_SIZE = BLOCK_SIZE
        self.positions = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))]
        self.score = 0
        self.food_position = food_position

    def move(self, direction):
        head = self.positions[0]
        direction_x, direction_y = direction

        next_position = (
            ((head[0] + (direction_x * self.BLOCK_SIZE)) % self.SCREEN_WIDTH),
            (head[1] + (direction_y * self.BLOCK_SIZE)) % self.SCREEN_HEIGHT,
        )

        food_eaten = next_position == self.food_position
        if food_eaten:
            self.eat_food()

        collision_detected = len(self.positions) > 2 and next_position in self.positions[2:]
        if collision_detected:
            self.reset()
            return

        self.positions.insert(0, next_position)
        should_trim_tail = len(self.positions) > self.length
        if should_trim_tail:
            self.positions.pop()

    def random_food_position(self):
        grid_width = self.SCREEN_WIDTH // self.BLOCK_SIZE
        grid_height = self.SCREEN_HEIGHT // self.BLOCK_SIZE
        
        while self.food_position in self.positions:
            grid_x = random.randint(0, grid_width - 1)
            grid_y = random.randint(0, grid_height - 1)
            self.food_position = (grid_x * self.BLOCK_SIZE, grid_y * self.BLOCK_SIZE)

    def reset(self):
        self.length = 1
        self.positions = [((self.SCREEN_WIDTH / 2), (self.SCREEN_HEIGHT / 2))]
        self.score = 0
        self.random_food_position()

    def eat_food(self):
        self.length += 1
        self.score += 100
        self.random_food_position()
