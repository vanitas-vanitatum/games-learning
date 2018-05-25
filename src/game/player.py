import random

from src.game.action import Action
from src.game.tictactoe import TicTacToe
from src.game.board import Board


class Player:
    def __init__(self, breed='none'):
        self.breed = breed
        # self.my_symb = None
        # self.enemy_symb = None

    def start_game(self, symbol):
        pass
        # self.my_symb = symbol
        # self.enemy_symb = Board.O if symbol == Board.X else Board.X

    def move(self, board):
        raise NotImplementedError

    def reward(self, value, board):
        raise NotImplementedError

    @staticmethod
    def available_moves(board):
        return board.get_legal_moves()


class HumanPlayer(Player):
    def __init__(self):
        super().__init__('human')

    def move(self, board):
        row_col = input("Your move? ")
        row, col = map(int, row_col.split(','))
        return Action(row, col)

    def reward(self, value, board):
        print("{} rewarded: {}".format(self.breed, value))


class RandomPlayer(Player):
    def __init__(self):
        super().__init__('random')

    def reward(self, value, board):
        pass

    def move(self, board, rand=False):
        row, col = random.choice(self.available_moves(board))
        return Action(row, col)

    def update_Q(self, reward, new_board, learning_rate, temporary_discount):
        pass


'''
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
            board[move] = Board.EMPTY
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
            other = Board.O if symbol == Board.X else Board.X
            val = self.minmax(board, other, alpha, beta)
            board[move] = Board.EMPTY
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
'''
