import random
from collections import deque


class Snake:
    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, food_position):
        self.length = 1
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.BLOCK_SIZE = BLOCK_SIZE
        self.positions = deque([((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))])
        self.score = 0
        self.food_position = food_position

    def move(self, direction):
        head_x, head_y = self.positions[0]
        dx, dy = direction

        new_head = (
            ((head_x + (dx * self.BLOCK_SIZE)) % self.SCREEN_WIDTH),
            (head_y + (dy * self.BLOCK_SIZE)) % self.SCREEN_HEIGHT,
        )

        if new_head == self.food_position:
            self.eat_food()

        if len(self.positions) > 2 and new_head in list(self.positions)[2:]:
            self.reset()
        else:
            self.positions.appendleft(new_head)
            if len(self.positions) > self.length:
                self.positions.pop()

    def random_food_position(self):
        while self.food_position in self.positions:
            self.food_position = (random.randint(0, self.SCREEN_WIDTH // self.BLOCK_SIZE - 1) * self.BLOCK_SIZE,
                                  random.randint(0, self.SCREEN_HEIGHT // self.BLOCK_SIZE - 1) * self.BLOCK_SIZE)

    def reset(self):
        self.length = 1
        self.positions = deque([((self.SCREEN_WIDTH / 2), (self.SCREEN_HEIGHT / 2))])
        self.score = 0
        self.random_food_position()

    def eat_food(self):
        self.length += 1
        self.score += 100
        self.random_food_position()
