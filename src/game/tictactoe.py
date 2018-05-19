import numpy as np
import random


class TicTacToe:
    """
    Based on:
    https://gist.github.com/fheisler/430e70fa249ba30e707f
    """
    X = 1
    O = -1
    EMPTY = 0

    WIN_COMBOS = [(0, 1, 2), (3, 4, 5), (6, 7, 8),  # horizontal
                  (0, 3, 6), (1, 4, 7), (2, 5, 8),  # vertical
                  (0, 4, 8), (2, 4, 6)]             # diagonal

    def __init__(self, playerX, playerO):
        self.board = [TicTacToe.EMPTY] * 9
        self.playerX, self.playerO = playerX, playerO
        self.playerX_turn = random.choice([True, False])

    @property
    def game_state(self):
        current_symbol = TicTacToe.X if self.playerX_turn else TicTacToe.O
        return np.asarray([current_symbol]+self.board)

    def play_game(self):
        self.playerX.start_game(TicTacToe.X)
        self.playerO.start_game(TicTacToe.O)
        game_finished = False
        while not game_finished:  # yolo
            if self.playerX_turn:
                player, symbol, opponent = self.playerX, TicTacToe.X, self.playerO
            else:
                player, symbol, opponent = self.playerO, TicTacToe.O, self.playerX

            if player.breed == "human":
                self.display_board()

            placing_index = player.move(self.board)

            if self.board[placing_index] != TicTacToe.EMPTY:  # illegal move
                player.reward(-99, self.board)  # score of shame
                game_finished = True
            else:
                self.board[placing_index] = symbol
                if TicTacToe.player_wins(self.board, symbol):
                    player.reward(1, self.board)
                    opponent.reward(-1, self.board)
                    game_finished = True
                elif TicTacToe.board_full(self.board):  # tie game
                    player.reward(0.5, self.board)
                    opponent.reward(0.5, self.board)
                    game_finished = True
                else:
                    opponent.reward(0, self.board)
                    self.playerX_turn = not self.playerX_turn

    @staticmethod
    def player_wins(board, symbol):
        for combo in TicTacToe.WIN_COMBOS:
            if all([board[i] == symbol for i in combo]):
                return True
        return False

    @staticmethod
    def board_full(board):
        return all([place != TicTacToe.EMPTY for place in board])

    @staticmethod
    def available_moves(board):
        return [i for i in range(len(board)) if board[i] == TicTacToe.EMPTY]

    def display_board(self):
        row = " {} | {} | {}"
        hr = "\n-----------\n"

        transl = {TicTacToe.X: 'X',
                  TicTacToe.O: 'O',
                  TicTacToe.EMPTY: ' '}
        board_repr = [transl[i] for i in self.board]
        print((row + hr + row + hr + row).format(*board_repr))

