import numpy as np
import tqdm

from src.game.tictactoe import TicTacToe
from src.game.board import Board
from src.qlearning.qplayer import QPlayer
from src.game.rewards import Reward
import random


class Learner:
    def __init__(self, game, learning_rate=0.3, discount=0.9, seed=None):
        self.game = game
        self.steps_done = 0

        self.epsilon_initial_value = 0.2
        self.epsilon_final_value = 0.2
        self.decay_step = 200

        self.learning_rate = learning_rate
        self.discount = discount
        self.temporary_discount = discount
        self._rng = np.random.RandomState(seed)

    def reset(self):
        self.temporary_discount = self.discount
        plx = self.game.player_x
        self.game.player_x = self.game.player_o
        self.game.player_o = plx

    def fit(self, num_episodes):
        results = []
        for episode in tqdm.tqdm(range(num_episodes)):
            self.reset()
            results.append(self.evaluate_episode(episode))
        starting_players, winners = list(zip(*results))
        return list(starting_players), list(winners)

    def evaluate_episode(self, episode_num):
        self.game.start_game(self.player_x_starts(episode_num))
        starting_player = Board.X if self.game.player_x_turn else Board.O
        is_terminal = False
        while not is_terminal:
            player, enemy = ((self.game.player_x, self.game.player_o)
                             if self.game.player_x_turn
                             else (self.game.player_o, self.game.player_x))

            eps_threshold = self.epsilon_threshold(episode_num)
            if self._rng.uniform() > eps_threshold:
                action = player.move(self.game.board, rand=False)
            else:
                action = player.move(self.game.board, rand=True)

            reward = self.game.apply_action(action)
            self.game.change_turns()
            is_terminal = self.game.is_terminal()

            self.update_Q_values(player, enemy, reward, self.game.board, is_terminal)
        winner = self.game.board.get_winner()
        if winner is None:
            winner = 0
        return starting_player, winner

    def player_x_starts(self, episode_num):
        # return None
        # return episode_num % 6 < 3
        return random.choice([True, False])

    def epsilon_threshold(self, episode_num):
        return (self.epsilon_final_value
                + ((self.epsilon_initial_value - self.epsilon_final_value)
                   * np.exp(-1 * episode_num / self.decay_step)))

    def update_Q_values(self, player, enemy, reward, new_board, is_terminal):
        default = lambda r, b, lr, td: None
        player_update_Q = player.update_Q if isinstance(player, QPlayer) else default
        enemy_update_Q = enemy.update_Q if isinstance(enemy, QPlayer) else default

        if reward == Reward.NONE:
            enemy_update_Q(reward, new_board, self.learning_rate, self.temporary_discount)

        elif reward == Reward.DRAW:
            player_update_Q(reward, new_board, self.learning_rate, self.temporary_discount)
            enemy_update_Q(reward, new_board, self.learning_rate, self.temporary_discount)

        elif reward == Reward.WIN:
            player_update_Q(reward, new_board, self.learning_rate, self.temporary_discount)
            enemy_update_Q(-reward, new_board, self.learning_rate, self.temporary_discount)

        elif reward == Reward.ILLEGAL:
            player_update_Q(reward, new_board, self.learning_rate, self.temporary_discount)