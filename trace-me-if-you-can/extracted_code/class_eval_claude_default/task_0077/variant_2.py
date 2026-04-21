import random


class Snake:
    FOOD_SCORE = 100
    INITIAL_LENGTH = 1
    
    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, food_position):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.BLOCK_SIZE = BLOCK_SIZE
        self.food_position = food_position
        self._initialize_snake()

    def _initialize_snake(self):
        self.length = self.INITIAL_LENGTH
        center_pos = ((self.SCREEN_WIDTH / 2), (self.SCREEN_HEIGHT / 2))
        self.positions = [center_pos]
        self.score = 0

    def _calculate_new_position(self, direction):
        current_head = self.positions[0]
        dx, dy = direction
        return (
            ((current_head[0] + (dx * self.BLOCK_SIZE)) % self.SCREEN_WIDTH),
            (current_head[1] + (dy * self.BLOCK_SIZE)) % self.SCREEN_HEIGHT,
        )

    def _check_collision_with_self(self, new_position):
        return len(self.positions) > 2 and new_position in self.positions[2:]

    def move(self, direction):
        new_position = self._calculate_new_position(direction)

        if new_position == self.food_position:
            self.eat_food()

        if self._check_collision_with_self(new_position):
            self.reset()
        else:
            self.positions.insert(0, new_position)
            if len(self.positions) > self.length:
                self.positions.pop()

    def random_food_position(self):
        while self.food_position in self.positions:
            self.food_position = (random.randint(0, self.SCREEN_WIDTH // self.BLOCK_SIZE - 1) * self.BLOCK_SIZE,
                                  random.randint(0, self.SCREEN_HEIGHT // self.BLOCK_SIZE - 1) * self.BLOCK_SIZE)

    def reset(self):
        self._initialize_snake()
        self.random_food_position()

    def eat_food(self):
        self.length += 1
        self.score += self.FOOD_SCORE
        self.random_food_position()
