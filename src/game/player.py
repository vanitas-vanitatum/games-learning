import numpy as np
from src.game.tictactoe import TicTacToe
import random


class Player:
    def __init__(self, breed='none'):
        self.breed = breed
        self.my_symb = None
        self.enemy_symb = None

    def start_game(self, symbol):
        self.my_symb = symbol
        self.enemy_symb = TicTacToe.O if symbol == TicTacToe.X else TicTacToe.X

    def move(self, board):
        raise NotImplementedError

    def reward(self, value, board):
        raise NotImplementedError

    def available_moves(self, board):
        return TicTacToe.available_moves(board)


class HumanPlayer(Player):
    def __init__(self):
        super().__init__('human')

    def start_game(self, char):
        print("\nNew game!")

    def move(self, board):
        return int(input("Your move? "))

    def reward(self, value, board):
        print("{} rewarded: {}".format(self.breed, value))


class RandomPlayer(Player):
    def __init__(self):
        super().__init__('random')

    def reward(self, value, board):
        pass

    def start_game(self, symbol):
        pass

    def move(self, board):
        return random.choice(self.available_moves(board))


class MinMaxPlayer(Player):
    def __init__(self):
        super().__init__('minimax')
        self.best_moves = {}

    def move(self, board):
        if tuple(board) in self.best_moves:
            return random.choice(self.best_moves[tuple(board)])
        if len(self.available_moves(board)) == 9:
            return random.choice([1, 3, 7, 9])
        best_yet = -2
        choices = []
        for move in self.available_moves(board):
            board[move] = self.my_symb
            optimal = self.minmax(board, self.enemy_symb, -2, 2)
            board[move] = TicTacToe.EMPTY
            if optimal > best_yet:
                choices = [move]
                best_yet = optimal
            elif optimal == best_yet:
                choices.append(move)
        self.best_moves[tuple(board)] = choices
        return random.choice(choices)

    def minmax(self, board, symbol, alpha, beta):
        if TicTacToe.player_wins(board, self.my_symb):
            return 1
        if TicTacToe.player_wins(board, self.enemy_symb):
            return -1
        if TicTacToe.board_full(board):
            return 0
        for move in self.available_moves(board):
            board[move] = symbol
            other = TicTacToe.O if symbol == TicTacToe.X else TicTacToe.X
            val = self.minmax(board, other, alpha, beta)
            board[move] = TicTacToe.EMPTY
            if symbol == self.my_symb:
                if val > alpha:
                    alpha = val
                if alpha >= beta:
                    return beta
            else:
                if val < beta:
                    beta = val
                if beta <= alpha:
                    return alpha
        if symbol == self.my_symb:
            return alpha
        else:
            return beta

    def reward(self, value, board):
        pass
