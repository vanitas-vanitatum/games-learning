import random

import numpy as np
import tqdm
import tensorflow as tf

from src.deep_q_learning.deep_qplayer import DeepQPlayer
from src.deep_q_learning.memory import ReplayMemory, Transition
from src.game.board import Board
from src.game.rewards import Reward
from src.game.tictactoe import TicTacToe
from src.utils import logger
from collections import deque


class DeepLearner:
    def __init__(self, game, discount=0.9, batch_size=32, memory_size=1000, target_net_update_episode=50, seed=None):
        self.game = game
        self.steps_done = 0

        self.epsilon_initial_value = 0.6
        self.epsilon_final_value = 0.1
        self.decay_step = 2500

        self.discount = discount

        self.memory = ReplayMemory(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_net_update_episode = target_net_update_episode

        self.episode_history = deque(maxlen=100)

        self._rng = np.random.RandomState(seed)

    def reset(self):
        self.episode_history.clear()
        self.game.player_x, self.game.player_o = self.game.player_o, self.game.player_x

    def fit(self, num_episodes):
        results = []
        for episode in tqdm.tqdm(range(num_episodes)):
            self.reset()
            results.append(self.evaluate_episode(episode))
            if episode % self.target_net_update_episode and episode > 0:
                self.game.player_x.refresh_target_net()

            logger.add_tf_summary(
                tf.Summary(value=[tf.Summary.Value(tag='average episode reward', simple_value=np.mean(self.episode_history)),
                                  tf.Summary.Value(tag='epsilon', simple_value=self.epsilon_threshold(episode))]),
                episode
            )
        starting_players, winners = list(zip(*results))
        return list(starting_players), list(winners)

    def evaluate_episode(self, episode_num):
        self.game.start_game(self.player_x_starts())
        starting_player = Board.X if self.game.player_x_turn else Board.O
        is_terminal = False
        while not is_terminal:
            current_player = Board.X if self.game.player_x_turn else Board.O
            future_moving_player = Board.O if self.game.player_x_turn else Board.X

            player = self.game.player_x if self.game.player_x_turn else self.game.player_o

            eps_threshold = self.epsilon_threshold(episode_num)
            if self._rng.uniform() > eps_threshold:
                action = player.move(self.game.board, rand=False)
            else:
                action = player.move(self.game.board, rand=True)

            reward = self.game.apply_action(action)

            is_terminal = self.game.is_terminal()

            current_player_previous_state, current_player_previous_action = player.get_last_state_action_as_player(
                current_player)
            future_player_previous_state, future_player_previous_action = player.get_last_state_action_as_player(
                future_moving_player)
            self.game.change_turns()
            if future_player_previous_state is not None:
                if reward == Reward.WIN:
                    transition_future = Transition(np.array(future_player_previous_state),
                                                   future_player_previous_action.row * self.game.board.n + future_player_previous_action.col,
                                                   np.array(self.game.board.board_state()), Reward.LOOSE,
                                                   is_terminal)
                    transition_current = Transition(np.array(current_player_previous_state),
                                                    current_player_previous_action.row * self.game.board.n + current_player_previous_action.col,
                                                    np.array(self.game.board.board_state()), Reward.WIN,
                                                    is_terminal)
                    self.memory.add_transition(transition_future)
                    self.memory.add_transition(transition_current)

                elif reward == Reward.DRAW:
                    self.memory.add_transition(Transition(np.array(future_player_previous_state),
                                                          future_player_previous_action.row * self.game.board.n + future_player_previous_action.col,
                                                          np.array(self.game.board.board_state()) * future_moving_player, Reward.DRAW,
                                                          is_terminal))

                    self.memory.add_transition(Transition(np.array(current_player_previous_state),
                                                          current_player_previous_action.row * self.game.board.n + current_player_previous_action.col,
                                                          np.array(self.game.board.board_state()), Reward.DRAW,
                                                          is_terminal))

                elif reward == Reward.NONE:
                    self.memory.add_transition(Transition(np.array(future_player_previous_state),
                                                          future_player_previous_action.row * self.game.board.n + future_player_previous_action.col,
                                                          np.array(self.game.board.board_state()), Reward.NONE,
                                                          is_terminal))
                elif reward == Reward.ILLEGAL:
                    self.memory.add_transition(Transition(np.array(current_player_previous_state),
                                                          current_player_previous_action.row * self.game.board.n + current_player_previous_action.col,
                                                          np.array(self.game.board.board_state()), Reward.ILLEGAL,
                                                          is_terminal))
                self.episode_history.append(reward)
            if not self.memory.is_enough_memory_for_players(self.batch_size):
                continue

            self.update_q_values(player)
        winner = self.game.board.get_winner()
        if winner is None:
            winner = 0
        return starting_player, winner

    @classmethod
    def player_x_starts(cls):
        return random.choice([True, False])

    def epsilon_threshold(self, episode_num):
        return (self.epsilon_final_value
                + ((self.epsilon_initial_value - self.epsilon_final_value)
                   * np.exp(-1 * episode_num / self.decay_step)))

    def update_q_values(self, player):
        batch = self.memory.sample(self.batch_size)
        player.update_q_model(batch, self.discount)


if __name__ == '__main__':
    from src.deep_q_learning.model import get_model_3x3

    player_1 = DeepQPlayer(get_model_3x3())
    learner = DeepLearner(TicTacToe(Board(3), player_1, player_1))
    learner.fit(1000)
