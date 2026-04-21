import random

class GamePiece:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size

    def move(self, dx, dy, screen_width, screen_height):
        self.x = (self.x + dx * self.size) % screen_width
        self.y = (self.y + dy * self.size) % screen_height

class SnakePiece(GamePiece):
    def __init__(self, x, y, size):
        super().__init__(x, y, size)
        self.length = 1
        self.positions = [(self.x, self.y)]
        self.score = 0

    def eat(self):
        self.length += 1
        self.score += 100
        self.positions.append((self.x, self.y))

    def check_collision(self, other):
        return (self.x, self.y) in other.positions

    def reset(self):
        self.length = 1
        self.positions = [(self.x, self.y)]
        self.score = 0

class Food(GamePiece):
    def __init__(self, screen_width, screen_height, size):
        super().__init__(0, 0, size)
        self.screen_width = screen_width
        self.screen_height = screen_height

    def place(self, snake_positions):
        while (self.x, self.y) in snake_positions:
            self.x = random.randint(0, self.screen_width // self.size - 1) * self.size
            self.y = random.randint(0, self.screen_height // self.size - 1) * self.size
