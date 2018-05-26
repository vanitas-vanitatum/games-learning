import random

from src.game.board import Board
from src.game.rewards import Reward


class TicTacToe:
    def __init__(self, board, player_x, player_o):

        self.board = board
        self.player_x = player_x
        self.player_o = player_o
        self.player_x_turn = None
        self.game_finished = False

    def play_game(self, player_x_starts=None, verbose=False):
        self.start_game(player_x_starts)
        verbose = self.player_x.breed == 'human' or self.player_o.breed == 'human' or verbose

        if verbose:
            print("New game. {}x{} board, connect {} to win.".format(
                self.board.n, self.board.n, self.board.win_combo_size))

        if verbose:
            self.board.display_board()

        while not self.game_finished:
            if self.player_x_turn:
                player, symbol, opponent, opponent_symbol = self.player_x, Board.X, self.player_o, Board.O
            else:
                player, symbol, opponent, opponent_symbol = self.player_o, Board.O, self.player_x, Board.X

            if player.breed == "human" or verbose:
                self.board.display_board()

            action = player.move(self.board)
            reward = self.apply_action(action)
            if self.game_finished:
                if reward == Reward.WIN:
                    winner = 'X' if self.board.get_winner() == Board.X else 'O'
                else:
                    winner = 'DRAW'
            self.change_turns()

            player.reward(reward, self.board.board_state())
            opponent.reward(reward, self.board.board_state())

        if verbose:
            self.board.display_board()
            print('Game finished')
            print(f'Winner is: {winner}')

    def change_turns(self):
        self.player_x_turn = not self.player_x_turn
        self.board.change_moving_player()

    def start_game(self, player_x_starts=None):
        self.player_x.start_game(Board.X)
        self.player_o.start_game(Board.O)
        self.board.reset()
        self.game_finished = False
        if player_x_starts is None:
            self.player_x_turn = random.choice([True, False])
        else:
            self.player_x_turn = player_x_starts

        if self.player_x_turn:
            self.board.moving_player = Board.X
        else:
            self.board.moving_player = Board.O

    def is_terminal(self):
        return self.game_finished

    def apply_action(self, action):
        row, col = action
        if self.board.get(row, col) != Board.EMPTY:
            self.game_finished = True
            return Reward.ILLEGAL

        self.board.set(row, col, self.board.moving_player)
        if self.board.is_move_winning(row, col, self.board.moving_player):
            self.game_finished = True
            return Reward.WIN
        elif self.board.is_board_full():
            self.game_finished = True
            return Reward.DRAW
        else:
            return Reward.NONE

    def moving_player_to_multiplier(self, moving_symbol):
        return 1 if (self.player_x_turn and moving_symbol == Board.X) or (
                not self.player_x_turn and moving_symbol == Board.O) else -1


