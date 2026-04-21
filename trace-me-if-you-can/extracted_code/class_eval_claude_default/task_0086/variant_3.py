class TicTacToe:
    def __init__(self, N=3):
        # Use dictionary for board state tracking
        self.board = {(i, j): ' ' for i in range(3) for j in range(N)}
        self.current_player = 'X'
        self.size = 3

    def make_move(self, row, col):
        position = (row, col)
        if self.board.get(position) == ' ':
            self.board[position] = self.current_player
            self.current_player = {'X': 'O', 'O': 'X'}[self.current_player]
            return True
        return False

    def check_winner(self):
        def check_line(positions):
            values = [self.board[pos] for pos in positions]
            return values[0] if values[0] == values[1] == values[2] != ' ' else None

        # Generate all winning combinations
        for i in range(3):
            # Check rows
            winner = check_line([(i, j) for j in range(3)])
            if winner:
                return winner
            # Check columns
            winner = check_line([(j, i) for j in range(3)])
            if winner:
                return winner

        # Check diagonals
        winner = check_line([(i, i) for i in range(3)])
        if winner:
            return winner
        
        winner = check_line([(i, 2-i) for i in range(3)])
        if winner:
            return winner

        return None

    def is_board_full(self):
        return all(value != ' ' for value in self.board.values())
