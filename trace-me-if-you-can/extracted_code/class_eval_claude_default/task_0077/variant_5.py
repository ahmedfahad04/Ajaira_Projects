import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GameConfig:
    screen_width: int
    screen_height: int
    block_size: int


class SnakeGame:
    def __init__(self, config: GameConfig):
        self.config = config
        
    def create_center_position(self) -> Tuple[float, float]:
        return (self.config.screen_width / 2, self.config.screen_height / 2)
    
    def wrap_position(self, x: float, y: float) -> Tuple[float, float]:
        wrapped_x = x % self.config.screen_width
        wrapped_y = y % self.config.screen_height
        return (wrapped_x, wrapped_y)
    
    def generate_food_position(self, occupied_positions: List[Tuple[float, float]]) -> Tuple[int, int]:
        max_grid_x = self.config.screen_width // self.config.block_size - 1
        max_grid_y = self.config.screen_height // self.config.block_size - 1
        
        while True:
            food_pos = (
                random.randint(0, max_grid_x) * self.config.block_size,
                random.randint(0, max_grid_y) * self.config.block_size
            )
            if food_pos not in occupied_positions:
                return food_pos


class Snake:
    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, food_position):
        config = GameConfig(SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE)
        self.game = SnakeGame(config)
        
        self.length = 1
        self.positions = [self.game.create_center_position()]
        self.score = 0
        self.food_position = food_position

    def move(self, direction):
        head_pos = self.positions[0]
        dx, dy = direction

        new_head = self.game.wrap_position(
            head_pos[0] + (dx * self.game.config.block_size),
            head_pos[1] + (dy * self.game.config.block_size)
        )

        if new_head == self.food_position:
            self.eat_food()

        if len(self.positions) > 2 and new_head in self.positions[2:]:
            self.reset()
        else:
            self.positions.insert(0, new_head)
            if len(self.positions) > self.length:
                self.positions.pop()

    def random_food_position(self):
        self.food_position = self.game.generate_food_position(self.positions)

    def reset(self):
        self.length = 1
        self.positions = [self.game.create_center_position()]
        self.score = 0
        self.random_food_position()

    def eat_food(self):
        self.length += 1
        self.score += 100
        self.random_food_position()
