class TicTacToe:
    def __init__(self, N=3):
        self.board = [[' ' for _ in range(N)] for _ in range(3)]
        self.current_player = 'X'

    def make_move(self, row, col):
        success = self._place_piece(row, col)
        if success:
            self._toggle_player()
        return success

    def _place_piece(self, row, col):
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            return True
        return False

    def _toggle_player(self):
        self.current_player = 'X' if self.current_player == 'O' else 'O'

    def check_winner(self):
        return (self._check_rows() or 
                self._check_columns() or 
                self._check_diagonals())

    def _check_rows(self):
        for row in self.board:
            if self._is_winning_line(row):
                return row[0]
        return None

    def _check_columns(self):
        for col in range(3):
            column = [self.board[row][col] for row in range(3)]
            if self._is_winning_line(column):
                return column[0]
        return None

    def _check_diagonals(self):
        main_diag = [self.board[i][i] for i in range(3)]
        anti_diag = [self.board[i][2-i] for i in range(3)]
        
        if self._is_winning_line(main_diag):
            return main_diag[0]
        if self._is_winning_line(anti_diag):
            return anti_diag[0]
        return None

    def _is_winning_line(self, line):
        return line[0] == line[1] == line[2] != ' '

    def is_board_full(self):
        return not any(' ' in row for row in self.board)
