class TicTacToe:
    def __init__(self, N=3):
        self.board = [[' ' for _ in range(N)] for _ in range(3)]
        self.current_player = 'X'
        self.move_count = 0

    def make_move(self, row, col):
        if self.board[row][col] != ' ':
            return False
        
        self.board[row][col] = self.current_player
        self.move_count += 1
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True

    def check_winner(self):
        board = self.board
        
        # Use functional approach with any() and generator expressions
        def three_in_a_row(a, b, c):
            return a == b == c != ' '

        # Check all possible winning combinations using iterators
        winning_combos = [
            # Rows
            *[(board[i][0], board[i][1], board[i][2]) for i in range(3)],
            # Columns  
            *[(board[0][j], board[1][j], board[2][j]) for j in range(3)],
            # Diagonals
            (board[0][0], board[1][1], board[2][2]),
            (board[0][2], board[1][1], board[2][0])
        ]

        for combo in winning_combos:
            if three_in_a_row(*combo):
                return combo[0]
        
        return None

    def is_board_full(self):
        return self.move_count == 9
