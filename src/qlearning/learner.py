import numpy as np
import tqdm

from src.game.nxn_tictactoe import TicTacToe, Board
from src.game.player import QPlayer


class Learner:
    def __init__(self, game, player_to_learn, seed=None):
        assert isinstance(player_to_learn, QPlayer)
        self.game = game
        self.player_to_learn = player_to_learn
        self.memory = []

        self._rng = np.random.RandomState(seed)

    def fit(self, num_episodes, learning_rate, discount):
        for episode in tqdm.tqdm(range(num_episodes)):
            self.evaluate_episode()
            self.update_player_values(learning_rate, discount)

    def update_player_values(self, learning_rate, discount):
        while len(self.memory) > 0:
            previous_state, new_state, action, reward = self.pop()

            new_state_value = self.player_to_learn.get_q_value(new_state, action)
            current_qvalue = self.player_to_learn.get_q_value(previous_state, action)

            updated_value = (1 - learning_rate) * current_qvalue + learning_rate * (reward + discount * new_state_value)

            self.player_to_learn.set_value(
                previous_state, action, updated_value
            )
            discount *= discount

    def evaluate_episode(self):
        self.game.start_game()
        while not self.game.is_game_terminal():
            current_state = self.game.board.board_state(self.player_to_learn.my_symb)
            if self.game.player_x_turn:
                player = self.game.player_x
            else:
                player = self.game.player_o

            action = player.move(self.game.board)
            reward = self.game.apply_action(action)
            new_state = self.game.board.board_state(self.player_to_learn.my_symb)
            self.game.player_x_turn = not self.game.player_x_turn

            self.push((current_state, new_state, action, reward))

    def pop(self):
        return self.memory.pop(-1)

    def push(self, elem):
        return self.memory.append(elem)

    def clear(self):
        return self.memory.clear()


if __name__ == '__main__':
    player_1 = QPlayer()
    learner = Learner(TicTacToe(Board(3), player_1, player_1), player_1)
    learner.fit(1000, 0.1, 0.99)
