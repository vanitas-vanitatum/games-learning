import random
from collections import defaultdict
from itertools import groupby

import numpy as np

from src.game.action import Action
from src.game.nxn_tictactoe import TicTacToe, Board


class Player:
    def __init__(self, breed='none'):
        self.breed = breed
        self.my_symb = None
        self.enemy_symb = None

    def start_game(self, symbol):
        self.my_symb = symbol
        self.enemy_symb = Board.O if symbol == Board.X else Board.X

    def move(self, board):
        raise NotImplementedError

    def reward(self, value, board):
        raise NotImplementedError

    def available_moves(self, board):
        return board.get_legal_moves()


class HumanPlayer(Player):
    def __init__(self):
        super().__init__('human')

    def move(self, board):
        ind = int(input("Your move? "))

        row, col = ind // board.n, ind % board.n
        return Action(row, col, self.my_symb)

    def reward(self, value, board):
        print("{} rewarded: {}".format(self.breed, value))


class RandomPlayer(Player):
    def __init__(self):
        super().__init__('random')

    def reward(self, value, board):
        pass

    def move(self, board):
        row, col = random.choice(self.available_moves(board))
        return Action(row, col, self.my_symb)


class QPlayer(Player):
    def __init__(self, q_initial_value=0.0):
        super().__init__('qplayer')
        self.state_action_mapping = defaultdict(lambda: float(q_initial_value))
        self.q_initial_value = q_initial_value

    def get_max_action_for_state(self, board):
        action_value_mapping = self.get_action_value_for_state(board)
        best_moves = []
        best_value = -np.inf
        for action, qvalue in action_value_mapping.items():
            if qvalue > best_value:
                best_value = qvalue
                best_moves = [action]
            elif qvalue == best_value:
                best_moves.append(action)
        return random.choice(best_moves)

    def get_action_value_for_state(self, board):
        all_actions = [
            Action(row, col, self.my_symb) for row, col in board.get_legal_moves()
        ]
        result = {}
        state = board.board_state(self.my_symb)
        for (s, action), value in self.state_action_mapping.items():
            if s == state:
                result[action] = value
        for act in all_actions:
            if act not in result:
                result[act] = self.q_initial_value

        return result

    def add_value(self, state, action, value):
        self.state_action_mapping[(state, action)] += value

    def set_value(self, state, action, value):
        self.state_action_mapping[(state, action)] = value

    def get_q_value(self, state, action):
        return self.state_action_mapping[(state, action)]

    def move(self, board):
        return self.get_max_action_for_state(board)

    def reward(self, value, board):
        pass


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
