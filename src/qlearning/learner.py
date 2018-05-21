import numpy as np
import tqdm

from src.game.nxn_tictactoe import TicTacToe, Board
from src.game.player import QPlayer


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

    def update_player_values(self, previous_state, new_state, action, reward):
        new_state_value = self.player_to_learn.get_max_q_value_for_state(self.game.board)
        current_qvalue = self.player_to_learn.get_q_value(previous_state, action)

        updated_value = current_qvalue + self.learning_rate * (
                reward + self.temporary_discount * new_state_value - current_qvalue)

        self.player_to_learn.set_value(
            previous_state, action, updated_value
        )
        self.temporary_discount *= self.discount

    def evaluate_episode(self, episode_num):
        self.game.start_game()
        while not self.game.is_game_terminal():
            current_state = self.game.board.board_state()
            if self.game.player_x_turn:
                player = self.game.player_x
            else:
                player = self.game.player_o
            sample = self._rng.uniform()
            eps_threshold = self.epsilon_final_value + (self.epsilon_initial_value - self.epsilon_final_value) * np.exp(
                -1 * episode_num / self.decay_step)
            if sample > eps_threshold:
                action = player.move(self.game.board)
            else:
                action = player.get_random_action_for_state(self.game.board)
            # action = player.move(self.game.board)
            reward = self.game.apply_action(action)
            self.game.board.change_moving_player()
            self.game.player_x_turn = not self.game.player_x_turn
            new_state = self.game.board.board_state()
            self.update_player_values(current_state, new_state, action, reward)
            # import ipdb
            # ipdb.set_trace()


if __name__ == '__main__':
    player_1 = QPlayer()
    learner = Learner(TicTacToe(Board(3), player_1, player_1), player_1)
    learner.fit(1000, 0.1, 0.99)
