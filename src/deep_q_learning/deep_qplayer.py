import random

import numpy as np
import keras.backend as K
from keras.models import Model, load_model

from src.game.action import Action
from src.game.board import Board
from src.qlearning import QPlayer


def get_dummy_action_input(length):
    return np.empty((length,))


class DeepQPlayer(QPlayer):
    def __init__(self, model: Model, q_initial_value=0.0):
        super().__init__(q_initial_value)
        self.train_model, self.predict_function = model

    def get_last_state_action_as_player(self, as_player):
        return self.last_state_action_as_X if as_player == Board.X else self.last_state_action_as_O

    def get_max_action_for_state(self, board):
        q_s = self.predict_function([np.array([board.board_state()])])[0][0]
        legal_moves = board.get_legal_moves_mask()
        actions = [Action(ind // board.n, ind % board.n) for ind in range(len(q_s))]
        q_s[np.logical_not(legal_moves)] = -np.inf

        mx_q = np.max(q_s)
        best_moves = [action for action, q_value in zip(actions, q_s) if q_value == mx_q]
        chosen = random.choice(best_moves)
        return chosen

    def update_q_model(self, transitions_batch, discount):
        previous_states, actions, next_states, rewards, _, terminals = zip(*transitions_batch)

        previous_states = np.array(previous_states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        mask_terminals = (1 - np.array(terminals))

        outputs = (np.ones(len(previous_states)) * rewards
                   + mask_terminals * discount * np.max(
                    self.predict_function([next_states])[0], axis=1))

        loss = self.train_model.train_on_batch([previous_states, outputs, actions], np.ones((len(previous_states), 1)))
        return loss

    def get_possible_q_values_for_board(self, board):
        mask = board.get_legal_moves_mask()
        actions_q_values = self.predict_function([np.array([board.board_state()])])[0]
        return actions_q_values * mask

    def save_Q(self, file_name):  # save table
        self.train_model.save(file_name)

    def load_Q(self, file_name):  # save table
        model = load_model(file_name)
        self.train_model = model
        self.predict_function = K.function(model.inputs, model.outputs)
