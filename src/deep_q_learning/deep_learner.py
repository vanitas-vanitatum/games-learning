import random

import numpy as np
import tqdm

from src.deep_q_learning.deep_qplayer import DeepQPlayer
from src.deep_q_learning.memory import ReplayMemory, Transition
from src.game.board import Board
from src.game.rewards import Reward
from src.game.tictactoe import TicTacToe


class DeepLearner:
    def __init__(self, game, discount=0.9, batch_size=32, memory_size=1000, seed=None):
        self.game = game
        self.steps_done = 0

        self.epsilon_initial_value = 0.2
        self.epsilon_final_value = 0.2
        self.decay_step = 200

        self.discount = discount

        self.memory = ReplayMemory(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size

        self._rng = np.random.RandomState(seed)

    def reset(self):
        self.game.player_x, self.game.player_o = self.game.player_o, self.game.player_x

    def fit(self, num_episodes):
        results = []
        for episode in tqdm.tqdm(range(num_episodes)):
            self.reset()
            results.append(self.evaluate_episode(episode))
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
            enemy = self.game.player_o if self.game.player_x_turn else self.game.player_x

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
                                                   np.array(self.game.board.board_state()), -reward,
                                                   future_moving_player,
                                                   is_terminal)
                    transition_current = Transition(np.array(current_player_previous_state),
                                                    current_player_previous_action.row * self.game.board.n + current_player_previous_action.col,
                                                    np.array(self.game.board.board_state()), reward,
                                                    current_player,
                                                    is_terminal)
                    self.memory.add_transition(transition_future)

                    self.memory.add_transition(transition_current)

                elif reward == Reward.DRAW:
                    self.memory.add_transition(Transition(np.array(future_player_previous_state),
                                                          future_player_previous_action.row * self.game.board.n + future_player_previous_action.col,
                                                          np.array(self.game.board.board_state()), reward,
                                                          future_moving_player,
                                                          is_terminal))

                    self.memory.add_transition(Transition(np.array(current_player_previous_state),
                                                          current_player_previous_action.row * self.game.board.n + current_player_previous_action.col,
                                                          np.array(self.game.board.board_state()), reward,
                                                          current_player,
                                                          is_terminal))

                elif reward == Reward.NONE:
                    self.memory.add_transition(Transition(np.array(future_player_previous_state),
                                                          future_player_previous_action.row * self.game.board.n + future_player_previous_action.col,
                                                          np.array(self.game.board.board_state()), reward,
                                                          future_moving_player,
                                                          is_terminal))

            if not self.memory.is_enough_memory_for_players(self.batch_size):
                continue

            self.update_q_values(player, enemy, current_player, future_moving_player)
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

    def update_q_values(self, player, enemy, current_moving_player, future_moving_player):
        def default(r, b, td):
            return None

        player_update_q = player.update_q_model if isinstance(player, DeepQPlayer) else default
        enemy_update_q = enemy.update_q_model if isinstance(enemy, DeepQPlayer) else default

        player_transitions_batch = self.memory.sample_for_player(self.batch_size, current_moving_player)
        enemy_transitions_batch = self.memory.sample_for_player(self.batch_size, future_moving_player)

        player_update_q(player_transitions_batch, self.discount)
        enemy_update_q(enemy_transitions_batch, self.discount)


if __name__ == '__main__':
    from src.deep_q_learning.model import get_model_3x3

    player_1 = DeepQPlayer(get_model_3x3())
    learner = DeepLearner(TicTacToe(Board(3), player_1, player_1))
    learner.fit(1000)
