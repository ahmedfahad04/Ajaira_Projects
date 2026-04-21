import random


class Snake:
    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, food_position):
        self.config = {
            'width': SCREEN_WIDTH,
            'height': SCREEN_HEIGHT,
            'block_size': BLOCK_SIZE,
            'initial_length': 1,
            'food_value': 100
        }
        self.state = self._create_initial_state()
        self.food_position = food_position

    def _create_initial_state(self):
        center_x = self.config['width'] / 2
        center_y = self.config['height'] / 2
        return {
            'length': self.config['initial_length'],
            'positions': [(center_x, center_y)],
            'score': 0
        }

    def _wrap_coordinate(self, coord, max_value):
        return coord % max_value

    def _get_next_head_position(self, direction):
        current_head = self.state['positions'][0]
        dx, dy = direction
        
        new_x = self._wrap_coordinate(
            current_head[0] + (dx * self.config['block_size']),
            self.config['width']
        )
        new_y = self._wrap_coordinate(
            current_head[1] + (dy * self.config['block_size']),
            self.config['height']
        )
        
        return (new_x, new_y)

    def move(self, direction):
        new_head = self._get_next_head_position(direction)

        if new_head == self.food_position:
            self.eat_food()

        if self._is_self_collision(new_head):
            self.reset()
        else:
            self._update_positions(new_head)

    def _is_self_collision(self, position):
        return len(self.state['positions']) > 2 and position in self.state['positions'][2:]

    def _update_positions(self, new_head):
        self.state['positions'].insert(0, new_head)
        if len(self.state['positions']) > self.state['length']:
            self.state['positions'].pop()

    def random_food_position(self):
        max_x = self.config['width'] // self.config['block_size'] - 1
        max_y = self.config['height'] // self.config['block_size'] - 1
        
        while self.food_position in self.state['positions']:
            self.food_position = (
                random.randint(0, max_x) * self.config['block_size'],
                random.randint(0, max_y) * self.config['block_size']
            )

    def reset(self):
        self.state = self._create_initial_state()
        self.random_food_position()

    def eat_food(self):
        self.state['length'] += 1
        self.state['score'] += self.config['food_value']
        self.random_food_position()

    @property
    def length(self):
        return self.state['length']
    
    @property
    def positions(self):
        return self.state['positions']
    
    @property
    def score(self):
        return self.state['score']
