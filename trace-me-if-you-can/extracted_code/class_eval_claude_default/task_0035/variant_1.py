from collections import deque

class EightPuzzle:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    def find_blank(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j

    def get_neighbors(self, state):
        i, j = self.find_blank(state)
        neighbors = []
        directions = [(-1, 0, 'up'), (1, 0, 'down'), (0, -1, 'left'), (0, 1, 'right')]
        
        for di, dj, direction in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < 3 and 0 <= nj < 3:
                new_state = [row[:] for row in state]
                new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
                neighbors.append((new_state, direction))
        
        return neighbors

    def solve(self):
        queue = deque([(self.initial_state, [])])
        visited = set()
        visited.add(str(self.initial_state))

        while queue:
            current_state, path = queue.popleft()

            if current_state == self.goal_state:
                return path

            for new_state, move in self.get_neighbors(current_state):
                state_key = str(new_state)
                if state_key not in visited:
                    visited.add(state_key)
                    queue.append((new_state, path + [move]))

        return None
