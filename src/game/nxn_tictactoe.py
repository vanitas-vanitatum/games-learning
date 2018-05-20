import random

import numpy as np


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
        self.turns, self.data, self.win_cells = None, None, None
        self.reset()

    def reset(self):
        self.turns = 0
        self.data = [Board.EMPTY] * self.size
        self.win_cells = []

    def board_state(self, current_player_symbol):
        return tuple([current_player_symbol] + self.data)

    def __str__(self):
        out = ""
        translator = {Board.X: 'X',
                      Board.O: 'O',
                      Board.EMPTY: ' '}
        for row in range(self.n):
            for col in range(self.n):
                out += "{} ".format(translator[self.get(row, col)])
            out += "\n"
        return out

    def set(self, row, col, symbol):
        self.data[row * self.n + col] = symbol

    def get(self, row, col):
        return self.data[row * self.n + col]

    def save_win_cells(self, player_symbol, cells):
        win_cells = []
        for row, col in cells:
            if self.get(row, col) == player_symbol:
                win_cells += [(row, col)]
            elif len(win_cells) < self.win_combo_size:
                win_cells = []
            else:
                break
        self.win_cells = win_cells

    def get_win_cells(self):
        return self.win_cells

    def is_winning_combo_among_cells(self, player_symbol, cells):
        count = 0
        for row, col in cells:
            if self.get(row, col) == player_symbol:
                count += 1
                if count == self.win_combo_size:
                    self.save_win_cells(player_symbol, cells)
                    return True
            else:
                count = 0
        return False

    def is_move_winning(self, row, col, player_symbol):
        """Check if the given player that makes his last move to (row, col) wins. """
        # Check horizontal.
        return (self._is_move_winning_horizontal(row, col, player_symbol)
                or self._is_move_winning_vertical(row, col, player_symbol)
                or self._is_move_winning_upright_diagonal(row, col, player_symbol)
                or self._is_move_winning_downright_diagonal(row, col, player_symbol))

    def is_board_full(self):
        return all([field != Board.EMPTY for field in self.data])

    def _is_move_winning_horizontal(self, row, col, player_symbol):
        min_col_ind = max(0, col - self.win_combo_size + 1)
        max_col_ind = min(self.n - 1, col + self.win_combo_size - 1)
        horizontal_cells = [(row, c) for c in range(min_col_ind, max_col_ind + 1)]
        return self.is_winning_combo_among_cells(player_symbol, horizontal_cells)

    def _is_move_winning_vertical(self, row, col, player_symbol):
        min_row_ind = max(0, row - self.win_combo_size + 1)
        max_row_ind = min(self.n - 1, row + self.win_combo_size - 1)
        vertical_cells = [(r, col) for r in range(min_row_ind, max_row_ind + 1)]
        return self.is_winning_combo_among_cells(player_symbol, vertical_cells)

    def _is_move_winning_upright_diagonal(self, row, col, player_symbol):
        diagonal_cells = []
        # go from left bottom to right up
        for i in range(-self.win_combo_size + 1, self.win_combo_size):
            r, c = row - i, col + i
            if self.is_field_on_board(r, c):
                diagonal_cells.append((r, c))

        return self.is_winning_combo_among_cells(player_symbol, diagonal_cells)

    def _is_move_winning_downright_diagonal(self, row, col, player_symbol):
        diagonal_cells = []
        # go from left rop to right bottom
        for i in range(-self.win_combo_size + 1, self.win_combo_size):
            r, c = row + i, col + i
            if self.is_field_on_board(r, c):
                diagonal_cells.append((r, c))

        return self.is_winning_combo_among_cells(player_symbol, diagonal_cells)

    def is_field_on_board(self, row, col):
        return 0 <= row < self.n and 0 <= col < self.n

    def get_legal_moves(self):
        moves = []
        for row in range(self.n):
            for col in range(self.n):
                if self.get(row, col) == Board.EMPTY:
                    moves += [(row, col)]
        return moves

    def display_board(self):
        out = ""
        translator = {Board.X: 'X',
                      Board.O: 'O',
                      Board.EMPTY: ' '}

        for row in range(self.n):
            for col in range(self.n):
                out += " {} ".format(translator[self.get(row, col)])
                if col < self.n - 1:
                    out += '|'
            if row < self.n - 1:
                out += '\n' + '---' * self.n + '-' * (self.n - 1) + '\n'
            else:
                out += '\n'
        print(out)


class TicTacToe:
    def __init__(self, board, player_x, player_o):

        self.board = board
        self.player_x = player_x
        self.player_o = player_o
        self.player_x_turn = None
        self.game_finished = False

    def play_game(self, player_x_starts=None, display=True):
        self.start_game(player_x_starts)

        if display:
            print("New game. {}x{} board, connect {} to win.".format(
                self.board.n, self.board.n, self.board.win_combo_size))

        if display:
            self.board.display_board()

        while not self.game_finished:
            if self.player_x_turn:
                player, symbol, opponent, opponent_symbol = self.player_x, Board.X, self.player_o, Board.O
            else:
                player, symbol, opponent, opponent_symbol = self.player_o, Board.O, self.player_x, Board.X

            if player.breed == "human" and display:
                self.board.display_board()

            action = player.move(self.board)
            reward = self.apply_action(action)
            multiplier = self.moving_player_to_multiplier(symbol)
            player.reward(reward * multiplier, self.board.board_state(symbol))
            opponent.reward(reward * -multiplier, self.board.board_state(symbol))
            self.player_x_turn = not self.player_x_turn

        if display:
            print('Game finished')
            self.board.display_board()

    def start_game(self, player_x_starts=None):
        self.player_x.start_game(Board.X)
        self.player_o.start_game(Board.O)
        self.board.reset()
        self.game_finished = False
        if player_x_starts is None:
            self.player_x_turn = random.choice([True, False])
        else:
            self.player_x_turn = player_x_starts

    def is_game_terminal(self):
        return self.game_finished

    def apply_action(self, action):
        row, col, symbol = action
        if self.board.get(row, col) != Board.EMPTY:
            self.game_finished = True
            return -99
        self.board.set(row, col, symbol)
        if self.board.is_move_winning(row, col, symbol):
            self.game_finished = True
            return 1
        elif self.board.is_board_full():
            self.game_finished = True
            return 0.5
        else:
            return 0

    def moving_player_to_multiplier(self, moving_symbol):
        return 1 if (self.player_x_turn and moving_symbol == Board.X) or (
                not self.player_x_turn and moving_symbol == Board.O) else -1
