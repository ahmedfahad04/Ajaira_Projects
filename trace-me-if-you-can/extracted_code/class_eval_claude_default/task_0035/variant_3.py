class EightPuzzle:
    GOAL = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    DIRECTIONS = {
        'up': lambda i, j: (i - 1, j),
        'down': lambda i, j: (i + 1, j),
        'left': lambda i, j: (i, j - 1),
        'right': lambda i, j: (i, j + 1)
    }

    def __init__(self, initial_state):
        self.start = initial_state

    def locate_zero(self, grid):
        return next((i, j) for i in range(3) for j in range(3) if grid[i][j] == 0)

    def create_successor(self, grid, direction):
        zero_i, zero_j = self.locate_zero(grid)
        new_i, new_j = self.DIRECTIONS[direction](zero_i, zero_j)
        
        if not (0 <= new_i < 3 and 0 <= new_j < 3):
            return None
            
        result = [row[:] for row in grid]
        result[zero_i][zero_j], result[new_i][new_j] = result[new_i][new_j], result[zero_i][zero_j]
        return result

    def expand_state(self, state):
        successors = []
        for direction in self.DIRECTIONS:
            successor = self.create_successor(state, direction)
            if successor is not None:
                successors.append((successor, direction))
        return successors

    def solve(self):
        agenda = [(self.start, [])]
        seen = []

        while agenda:
            current, moves = agenda.pop(0)
            
            if current in seen:
                continue
            seen.append(current)

            if current == self.GOAL:
                return moves

            for next_state, action in self.expand_state(current):
                if next_state not in seen:
                    agenda.append((next_state, moves + [action]))

        return None
