import numpy as np
from src.utils import hzcat


class Board:
    X = 1
    O = -1
    EMPTY = 0

    def __init__(self, n=3, win_combo_size=3):
        """
        N x N Tic-Tac-Toe board, with custom number of symbols in row to win.
        Params:
        :param n: Size of board ( n x n )
        :param win_combo_size: Number of required consecutive player symbols to win game.
            Can be horizontal, vertical or diagonal.
        """
        self.n = n
        self.win_combo_size = win_combo_size
        self.size = n * n
        self.turns, self.data, self.winner = None, None, None
        self.moving_player = 0
        self.reset()

    def reset(self):
        self.turns = 0
        self.data = [Board.EMPTY] * self.size
        self.winner = None

    def board_state(self):
        return np.array(self.data)

    def __str__(self):
        return BoardStringConverter().simple_convert(self)

    def set(self, row, col, symbol):
        self.data[row * self.n + col] = symbol

    def get(self, row, col):
        return self.data[row * self.n + col]

    def save_winner(self, player_symbol):
        self.winner = player_symbol

    def get_winner(self):
        return self.winner

    def is_move_winning(self, row, col, player_symbol):
        """Check if the given player that makes his last move to (row, col) wins. """
        checker = BoardChecker(self)
        return checker.is_move_winning(row, col, player_symbol)

    def is_board_full(self):
        return all([field != Board.EMPTY for field in self.data])

    def get_legal_moves(self):
        moves = []
        if self.winner is not None:
            return moves
        for row in range(self.n):
            for col in range(self.n):
                if self.get(row, col) == Board.EMPTY:
                    moves += [(row, col)]
        return moves

    def get_legal_moves_mask(self):
        if self.winner is not None:
            return []
        moves = np.zeros((len(self.data),))
        moves[np.array(self.data) == Board.EMPTY] = 1
        return moves.astype(np.bool)

    def is_state_terminal(self, player_symbol=None):
        if self.is_board_full():
            return True

        if player_symbol:
            check = lambda r, c: self.is_move_winning(r, c, player_symbol)
        else:
            check = lambda r, c: (self.is_move_winning(r, c, Board.X)
                                  or self.is_move_winning(r, c, Board.O))
        for row in range(self.n):
            for col in range(self.n):
                if check(row, col):
                    return True
        return False

    def display_board(self):
        print(BoardStringConverter().convert(self))

    @staticmethod
    def display_custom_board(board_data):
        new_board = Board()
        new_board.data = board_data[1:]
        print(BoardStringConverter().convert(new_board))

    def change_moving_player(self):
        self.moving_player = Board.X if self.moving_player == Board.O else Board.O

    def parse_state(self, moving_player, state_string):
        self.moving_player = moving_player
        for r in range(self.n):
            for c in range(self.n):
                symb, *state_string = state_string
                if symb == 'X' or symb == 'x':
                    self.set(r, c, 1)
                elif symb == 'O' or symb == 'o':
                    self.set(r, c, -1)
                else:
                    self.set(r, c, 0)


class BoardChecker:

    def __init__(self, board):
        self.board = board

    def is_move_winning(self, row, col, player_symbol):
        """Check if the given player that makes his last move to (row, col) wins. """
        # Check horizontal.
        return (self._is_move_winning_horizontal(row, col, player_symbol)
                or self._is_move_winning_vertical(row, col, player_symbol)
                or self._is_move_winning_upright_diagonal(row, col, player_symbol)
                or self._is_move_winning_downright_diagonal(row, col, player_symbol))

    def _is_move_winning_horizontal(self, row, col, player_symbol):
        min_col_ind = max(0, col - self.board.win_combo_size + 1)
        max_col_ind = min(self.board.n - 1, col + self.board.win_combo_size - 1)
        horizontal_cells = [(row, c) for c in range(min_col_ind, max_col_ind + 1)]
        return self._is_winning_combo_among_cells(player_symbol, horizontal_cells)

    def _is_move_winning_vertical(self, row, col, player_symbol):
        min_row_ind = max(0, row - self.board.win_combo_size + 1)
        max_row_ind = min(self.board.n - 1, row + self.board.win_combo_size - 1)
        vertical_cells = [(r, col) for r in range(min_row_ind, max_row_ind + 1)]
        return self._is_winning_combo_among_cells(player_symbol, vertical_cells)

    def _is_move_winning_upright_diagonal(self, row, col, player_symbol):
        diagonal_cells = []
        # go from left bottom to right up
        for i in range(-self.board.win_combo_size + 1, self.board.win_combo_size):
            r, c = row - i, col + i
            if self._is_field_on_board(r, c):
                diagonal_cells.append((r, c))

        return self._is_winning_combo_among_cells(player_symbol, diagonal_cells)

    def _is_move_winning_downright_diagonal(self, row, col, player_symbol):
        diagonal_cells = []
        # go from left rop to right bottom
        for i in range(-self.board.win_combo_size + 1, self.board.win_combo_size):
            r, c = row + i, col + i
            if self._is_field_on_board(r, c):
                diagonal_cells.append((r, c))

        return self._is_winning_combo_among_cells(player_symbol, diagonal_cells)

    def _is_winning_combo_among_cells(self, player_symbol, cells):
        count = 0
        for row, col in cells:
            if self.board.get(row, col) == player_symbol:
                count += 1
                if count == self.board.win_combo_size:
                    self.board.save_winner(player_symbol)
                    return True
            else:
                count = 0
        return False

    def _is_field_on_board(self, row, col):
        return 0 <= row < self.board.n and 0 <= col < self.board.n


class BoardStringConverter:
    TRANSLATOR = {Board.X: 'X',
                  Board.O: 'O',
                  Board.EMPTY: ' '}

    def convert(self, board, header=True):
        if header:
            return (self._header(board.n)
                    + hzcat([self._index(board.n),
                             self._content(board)]))
        else:
            return self._content(board)

    def simple_convert(self, board):
        out = ''
        for row in range(board.n):
            for col in range(board.n):
                out += "{} ".format(self.TRANSLATOR[board.get(row, col)])
            out += "\n"
        return out

    def _header(self, n):
        header = ' ' * 6
        for col in range(n):
            header += f'{col:^3d} '
        header += '\n'
        return header

    def _index(self, n):
        index = ''
        for row in range(n):
            index += f'{row:4d}: \n\n'
        return index

    def _content(self, board):
        cont = ''
        for row in range(board.n):
            for col in range(board.n):
                cont += " {} ".format(self.TRANSLATOR[board.get(row, col)])
                if col < board.n - 1:
                    cont += '|'
            if row < board.n - 1:
                cont += '\n' + '---' * board.n + '-' * (board.n - 1) + '\n'
            else:
                cont += '\n'
        return cont

