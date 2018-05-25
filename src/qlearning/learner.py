import numpy as np
import tqdm

from src.game.tictactoe import TicTacToe
from src.game.board import Board
from src.qlearning.qplayer import QPlayer
from src.game.rewards import Reward


class Learner:
    def __init__(self, game, player_to_learn, learning_rate=0.1, discount=0.9, seed=None):
        assert isinstance(player_to_learn, QPlayer)
        self.game = game
        self.player_to_learn = player_to_learn
        self.steps_done = 0

        self.epsilon_initial_value = 0.9
        self.epsilon_final_value = 0.05
        self.decay_step = 200

        self.learning_rate = learning_rate
        self.discount = discount
        self.temporary_discount = discount
        self._rng = np.random.RandomState(seed)

    def reset(self):
        self.temporary_discount = self.discount

    def fit(self, num_episodes):
        for episode in tqdm.tqdm(range(num_episodes)):
            self.reset()
            self.evaluate_episode(episode)
            # self.update_player_values()

    def evaluate_episode(self, episode_num):
        self.game.start_game()
        is_terminal = False
        while not is_terminal:
            current_state = self.game.board.board_state()
            player = self.game.player_x if self.game.player_x_turn else self.game.player_o
            enemy = self.game.player_o if self.game.player_x_turn else self.game.player_x
            eps_threshold = self.epsilon_threshold(episode_num)
            if self._rng.uniform() > eps_threshold:
                try:
                    action = player.move(self.game.board, rand=False)
                except IndexError:
                    action = player.move(self.game.board, rand=True)
                    self.game.board.display_board()
            else:
                action = player.move(self.game.board, rand=True)

            reward = self.game.apply_action(action)
            self.game.change_turns()

            is_terminal = self.game.is_terminal()
            self.update_Q_values(player, enemy, reward, self.game.board, is_terminal)

    def epsilon_threshold(self, episode_num):
        return (self.epsilon_final_value
                + ((self.epsilon_initial_value - self.epsilon_final_value)
                   * np.exp(-1 * episode_num / self.decay_step)))

    def update_Q_values(self, player, enemy, reward, new_board, is_terminal):
        if is_terminal:
            if reward == Reward.DRAW:
                player.update_Q(reward, new_board, self.learning_rate, self.temporary_discount)
                enemy.update_Q(reward, new_board, self.learning_rate, self.temporary_discount)
            else:
                player.update_Q(reward, new_board, self.learning_rate, self.temporary_discount)
                enemy.update_Q(-reward, new_board, self.learning_rate, self.temporary_discount)
        if reward == Reward.ILLEGAL:
            player.update_Q(reward, new_board, self.learning_rate, self.temporary_discount)

        elif reward == Reward.NONE:
            enemy.update_Q(-reward, new_board, self.learning_rate, self.temporary_discount)

        self.temporary_discount *= self.discount


if __name__ == '__main__':
    player_1 = QPlayer()
    learner = Learner(TicTacToe(Board(3), player_1, player_1), player_1)
    learner.fit(1000, 0.1, 0.99)
