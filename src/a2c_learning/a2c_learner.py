import random
from collections import deque

import numpy as np
import tensorflow as tf
import tqdm

from src.deep_q_learning.memory import Transition
from src.game.board import Board
from src.game.rewards import Reward
from src.utils import logger


class A2CLearner:
    def __init__(self, game, a2c_player_1, a2c_player_2, discount=0.9, seed=None):
        self.game = game
        self.steps_done = 0

        self.epsilon_initial_value = 0.6
        self.epsilon_final_value = 0.0001
        self.decay_step = 10000
        self.a2c_player_1 = a2c_player_1
        self.a2c_player_2 = a2c_player_2

        self.discount = discount
        self.episode_history_player_x = deque(maxlen=100)
        self.episode_history_player_o = deque(maxlen=100)
        self.updates_x_player = True

        self._rng = np.random.RandomState(seed)

    def reset(self):
        self.episode_history_player_x.clear()
        self.episode_history_player_o.clear()

    def fit(self, num_episodes):
        results = []
        for episode in tqdm.tqdm(range(num_episodes)):
            self.reset()
            results.append(self.evaluate_episode(episode))

            logger.add_tf_summary(
                tf.Summary(
                    value=[tf.Summary.Value(tag='average episode reward/player x', simple_value=np.mean(self.episode_history_player_x)),
                           tf.Summary.Value(tag='average episode reward/player o', simple_value=np.mean(self.episode_history_player_o)),
                           tf.Summary.Value(tag='epsilon', simple_value=self.epsilon_threshold(episode))]),
                episode
            )
        starting_players, winners = list(zip(*results))
        return list(starting_players), list(winners)

    def evaluate_episode(self, episode_num):

        self.game.start_game(self.player_x_starts())
        starting_player = Board.X if self.game.player_x_turn else Board.O
        is_terminal = False
        transitions = {
            Board.X: [],
            Board.O: []
        }
        rewards = {
            Board.X: self.episode_history_player_x,
            Board.O: self.episode_history_player_o
        }
        while not is_terminal:
            current_player = Board.X if self.game.player_x_turn else Board.O
            future_moving_player = Board.O if self.game.player_x_turn else Board.X

            player = self.a2c_player_1 if self.game.player_x_turn else self.a2c_player_2
            enemy = self.a2c_player_2 if self.game.player_x_turn else self.a2c_player_1

            eps_threshold = self.epsilon_threshold(episode_num)
            if self._rng.uniform() > eps_threshold:
                action = player.move(self.game.board, rand=False)
            else:
                action = player.move(self.game.board, rand=True)
            # action = player.move(self.game.board, rand=False)

            reward = self.game.apply_action(action)

            is_terminal = self.game.is_terminal()

            current_player_previous_state, current_player_previous_action = player.get_last_state_action_as_player(
                current_player)
            future_player_previous_state, future_player_previous_action = enemy.get_last_state_action_as_player(
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
                    transitions[current_player].append(transition_current)
                    transitions[future_moving_player].append(transition_future)
                    rewards[current_player].append(Reward.WIN)
                    rewards[future_moving_player].append(Reward.LOOSE)

                elif reward == Reward.DRAW:
                    transition_future = Transition(np.array(future_player_previous_state),
                                                   future_player_previous_action.row * self.game.board.n + future_player_previous_action.col,
                                                   np.array(self.game.board.board_state()) * future_moving_player,
                                                   Reward.DRAW,
                                                   is_terminal)

                    transition_current = Transition(np.array(current_player_previous_state),
                                                    current_player_previous_action.row * self.game.board.n + current_player_previous_action.col,
                                                    np.array(self.game.board.board_state()), Reward.DRAW,
                                                    is_terminal)
                    transitions[future_moving_player].append(transition_future)
                    transitions[current_player].append(transition_current)
                    rewards[current_player].append(Reward.DRAW)
                    rewards[future_moving_player].append(Reward.DRAW)

                elif reward == Reward.NONE:
                    transitions[future_moving_player].append(Transition(np.array(future_player_previous_state),
                                                                        future_player_previous_action.row * self.game.board.n + future_player_previous_action.col,
                                                                        np.array(self.game.board.board_state()),
                                                                        Reward.NONE,
                                                                        is_terminal))
                    rewards[future_moving_player].append(Reward.NONE)

                elif reward == Reward.ILLEGAL:
                    transitions[current_player].append(Transition(np.array(current_player_previous_state),
                                                                  current_player_previous_action.row * self.game.board.n + current_player_previous_action.col,
                                                                  np.array(self.game.board.board_state()),
                                                                  Reward.ILLEGAL,
                                                                  is_terminal))

                    rewards[current_player].append(Reward.ILLEGAL)

        if episode_num % 1000000 == 0:
            self.update_values(transitions[Board.X], self.a2c_player_1, False)
        else:
            self.update_values(transitions[Board.O], self.a2c_player_2, True)
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

    def update_values(self, transitions, player, log):
        player.update_models(transitions, self.discount, log)
