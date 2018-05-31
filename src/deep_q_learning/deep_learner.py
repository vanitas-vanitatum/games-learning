import random

import numpy as np
import tqdm
import tensorflow as tf

from src.deep_q_learning.deep_qplayer import DeepQPlayer
from src.deep_q_learning.memory import ReplayMemory, Transition
from src.game.board import Board
from src.game.rewards import Reward
from src.game.tictactoe import TicTacToe
from collections import deque


class DeepLearner:
    def __init__(self, game, discount=0.8, batch_size=32, memory_size=1000, seed=None):
        self.game = game
        self.steps_done = 0

        self.epsilon_initial_value = 1.0
        self.epsilon_final_value = 0.01
        self.decay_step = 5000
        self.epsilon_step = (self.epsilon_initial_value - self.epsilon_final_value) / self.decay_step

        self.discount = discount

        self.memory = ReplayMemory(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.episode_history = deque(maxlen=100)
        self.summary_writer = tf.summary.FileWriter('logdir')

        self._rng = np.random.RandomState(seed)

    def reset(self):
        self.episode_history.clear()
        self.game.player_x, self.game.player_o = self.game.player_o, self.game.player_x

    def fit(self, num_episodes):
        results = []
        for episode in tqdm.tqdm(range(num_episodes)):
            self.reset()
            res  = self.evaluate_episode(episode)
            res, loss = (res[0], res[1]), res[2]
            results.append(res)
            if episode % 100 == 0:
                print(f'Loss {loss.mean()}')

            self.summary_writer.add_summary(
                tf.Summary(value=[tf.Summary.Value(tag='average episode reward', simple_value=np.mean(self.episode_history)),
                                  tf.Summary.Value(tag='epsilon', simple_value=self.epsilon_threshold(episode))]),
                episode
            )
        starting_players, winners = list(zip(*results))
        return list(starting_players), list(winners)

    def evaluate_episode(self, episode_num):
        losses = []
        sar_prev = [(None, None, None), (None, None, None)]  # [(s, a, r(a)), (s(a), o, r(o)]

        self.game.start_game(self.player_x_starts())
        starting_player = Board.X if self.game.player_x_turn else Board.O
        is_terminal = False
        while not is_terminal:
            current_player = Board.X if self.game.player_x_turn else Board.O
            #future_moving_player = Board.O if self.game.player_x_turn else Board.X
            player = self.game.player_x if self.game.player_x_turn else self.game.player_o
            #enemy = self.game.player_o if self.game.player_x_turn else self.game.player_x
            state = self.game.board.board_state()
            q = player.get_possible_q_values_for_board(self.game.board)
            q_max = q.max()
            eps_threshold = self.epsilon_threshold(episode_num)
            if self._rng.uniform() > eps_threshold:
                action = player.move(self.game.board, rand=False)
            else:
                action = player.move(self.game.board, rand=True)
            a = action.row * self.game.board.n + action.col

            s_prev, a_prev, r_prev = sar_prev.pop(0)

            if s_prev is not None:
                transformer = StateActionTransformer(s_prev, a_prev, self.game.board.n)
                s_prev, a_prev = transformer.generate_transforms()
                y_prev = r_prev + self.discount * q_max
                loss = player.update_q(s_prev,
                                       a_prev, [y_prev]*len(s_prev))
            reward = self.game.apply_action(action)
            new_state = self.game.board.board_state()
            is_terminal = self.game.is_terminal()

            if is_terminal:
                s_prev, a_prev, r_prev = sar_prev[-1]
                y = reward
                y_prev = r_prev - self.discount * reward

                transformer = StateActionTransformer(s_prev, a_prev, self.game.board.n)
                s_prev, a_prev = transformer.generate_transforms()

                transformer = StateActionTransformer(state, a, self.game.board.n)
                s, a = transformer.generate_transforms()

                # Update Q network
                s_up = s + s_prev
                a_up = a + a_prev
                y_up = [y] * len(s) + [y_prev] * len(s_prev)
                loss = player.update_q(s_up, a_up, y_up)
                losses.append(loss.mean())

            sar_prev.append((state, a, reward))

            self.game.change_turns()
            self.episode_history.append(reward)

        winner = self.game.board.get_winner()
        if winner is None:
            winner = 0
        return starting_player, winner, np.asarray(losses)

    @classmethod
    def player_x_starts(cls):
        return random.choice([True, False])

    def epsilon_threshold(self, episode_num):
#        return (self.epsilon_final_value
#                + ((self.epsilon_initial_value - self.epsilon_final_value)
#                   * np.exp(-1 * episode_num / self.decay_step)))
        return (self.epsilon_initial_value
                - self.epsilon_step * min(episode_num, self.decay_step))

    def update_q_values(self, player, enemy, current_moving_player, future_moving_player):
        def default(r, b, td):
            return None

        player_update_q = player.update_q_model if isinstance(player, DeepQPlayer) else default
        enemy_update_q = enemy.update_q_model if isinstance(enemy, DeepQPlayer) else default

        player_transitions_batch = self.memory.sample_for_player(self.batch_size, current_moving_player)
        enemy_transitions_batch = self.memory.sample_for_player(self.batch_size, future_moving_player)

        batch = player_transitions_batch + enemy_transitions_batch
        np.random.shuffle(batch)
        player_update_q(batch, self.discount)
#        enemy_update_q(enemy_transitions_batch, self.discount)


class StateActionTransformer:
    ACTION_ROT90 = {0: 6, 1: 3, 2: 0,
                    3: 7, 4: 4, 5: 1,
                    6: 8, 7: 5, 8: 2}

    ACTION_FLIPLR = {0: 2, 1: 1, 2: 0,
                     3: 5, 4: 4, 5: 3,
                     6: 8, 7: 7, 8: 6}

    ACTION_FLIPUD = {0: 6, 1: 7, 2: 8,
                     3: 3, 4: 4, 5: 5,
                     6: 0, 7: 1, 8: 2}

    ACTION_TRANSPOSE = {0: 0, 1: 3, 2: 6,
                        3: 1, 4: 4, 5: 7,
                        6: 2, 7: 5, 8: 8}

    def __init__(self, state, action, n):
        self.n = n
        self.player = state[0]
        self.state = np.asarray(state[1:]).reshape(n, n)
        self.action = action

    def generate_transforms(self):
        transforms = [(self.state, self.action)]
        transforms = self._gen_rotations(transforms)
        transforms = self._gen_flips(transforms)
        transforms = self._gen_transposes(transforms)
        states, actions = self._multiply_per_player(transforms)
        while len(states) < 9:
            s = np.asarray([self.player] + list(self.state.flatten()))
            states.append(s)
            actions.append(self.action)
        return states, actions

    def _same_states(self, state_actions, sa):
        """
        Check states s1 (or one of in case of array-like) and s2 are the same.
        """
        s, a = sa
        for st, act in state_actions:
            if (st == s).all() and act == a:
                return True
        return False

    def _gen_rotations(self, transforms):
        sa = (self.state, self.action)

        # Apply rotations
        current_s, current_a = sa
        for i in range(1, 4):  # rotate to 90, 180, 270 degrees
            current_s = np.rot90(current_s)
            current_a = self.ACTION_ROT90[current_a]
            if not self._same_states(transforms, (current_s, current_a)):
                # Skip rotated state matching state already contained in list
                transforms.append((current_s, current_a))
        return transforms

    def _gen_flips(self, transforms):
        sa = (self.state, self.action)
        # Apply flips
        lr_s = np.fliplr(self.state)
        lr_a = self.ACTION_FLIPLR[self.action]
        if not self._same_states(transforms, (lr_s, lr_a)):
            transforms.append((lr_s, lr_a))

        ud_s = np.flipud(self.state)
        ud_a = self.ACTION_FLIPUD[self.action]
        if not self._same_states(transforms, (ud_s, ud_a)):
            transforms.append((ud_s, ud_a))
        return transforms

    def _gen_transposes(self, transforms):
        t1_s = self.state.T
        t1_a = self.ACTION_TRANSPOSE[self.action]
        if not self._same_states(transforms, (t1_s, t1_a)):
            transforms.append((t1_s, t1_a))

        t2_s = np.fliplr(t1_s)
        t2_a = self.ACTION_FLIPUD[t1_a]
        if not self._same_states(transforms, (t2_s, t2_a)):
            transforms.append((t2_s, t2_a))
        return transforms

    def _multiply_per_player(self, transforms):
        states, actions = [], []
        for state, action in transforms:
            s = np.asarray([self.player] + list(state.flatten()))
            states.append(s)
            actions.append(action)
            states.append(-s)
            actions.append(action)
        return states, actions


if __name__ == '__main__':
    from src.deep_q_learning.deep_qplayer import DeepQPlayer
    from src.deep_q_learning.memory import ReplayMemory, Transition
    from src.game.board import Board
    from src.game.rewards import Reward
    from src.game.tictactoe import TicTacToe
    from src.game.player import HumanPlayer
    from src.deep_q_learning.model import get_model_3x3, get_model_simple

    player_1 = DeepQPlayer(get_model_simple())
    learner = DeepLearner(TicTacToe(Board(3), player_1, player_1))
    res = learner.fit(10000)
    t = TicTacToe(Board(3, 3), player_1, HumanPlayer())
