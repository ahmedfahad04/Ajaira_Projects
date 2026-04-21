class SlidingPuzzle:
        def __init__(self, start):
            self.start = start
            self.target = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

        def locate_empty(self, board):
            for idx, row in enumerate(board):
                if 0 in row:
                    return idx, row.index(0)

        def swap_tiles(self, board, direction):
            x, y = self.locate_empty(board)
            new_board = [row[:] for row in board]

            if direction == 'up' and x > 0:
                new_board[x][y], new_board[x - 1][y] = new_board[x - 1][y], new_board[x][y]
            elif direction == 'down' and x < 2:
                new_board[x][y], new_board[x + 1][y] = new_board[x + 1][y], new_board[x][y]
            elif direction == 'left' and y > 0:
                new_board[x][y], new_board[x][y - 1] = new_board[x][y - 1], new_board[x][y]
            elif direction == 'right' and y < 2:
                new_board[x][y], new_board[x][y + 1] = new_board[x][y + 1], new_board[x][y]

            return new_board

        def explore_moves(self, board):
            directions = ['up', 'down', 'left', 'right']
            moves = []

            for direction in directions:
                new_board = self.swap_tiles(board, direction)
                if new_board != board:
                    moves.append((new_board, direction))

            return moves

        def resolve(self):
            queue = [(self.start, [])]
            visited = set()

            while queue:
                current, path = queue.pop(0)
                visited.add(tuple(map(tuple, current)))

                if current == self.target:
                    return path

                for next_board, move in self.explore_moves(current):
                    if tuple(map(tuple, next_board)) not in visited:
                        queue.append((next_board, path + [move]))

            return None
