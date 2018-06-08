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
        #state = self.conv_states([np.array([board.board_state()])])
        state = self.conv_states([board.board_state()])
        state = [np.asarray([state[0]])]
        q_s = self.predict_function(state)[0][0]
        actions = board.get_legal_moves()

        mx_q = np.max(q_s)
        action_qs = [q_s[act.row * board.n + act.col] for act in actions]
        best_q = max(action_qs)
        best_moves = [act for act, q in zip(actions, action_qs) if q == best_q]
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

    def update_q(self, previous_states, actions, y):
        previous_states = self.conv_states(previous_states)
        previous_states = np.array(previous_states)
        actions = np.array(actions).reshape(-1,1)
        #print(actions.shape)
        y = np.array(y).reshape(-1, 1)
        #print(y.shape)
        loss = self.train_model.train_on_batch([previous_states, y, actions], np.ones((len(previous_states), 1)))
        return loss

    def conv_states(self, states):
        st = []
        for s in states:
            s = np.asarray(s).flatten()
            #print(s.shape)
            #s1 = (s[1:] == s[0]).astype(np.int32)
            #s2 = (s[1:] == -s[0]).astype(np.int32)
            #s3 = (s[1:] == 0).astype(np.int32)
            #st.append(np.concatenate([s1, s2, s3]))#.reshape(1,-1))
            st.append(s[1:]*s[0])
        res = np.asarray(st).reshape(len(st), -1)
        return res

    def get_possible_q_values_for_board(self, board):
        #state = self.conv_states([np.array([board.board_state()])])
        state = self.conv_states([board.board_state()])
        state = [np.asarray([state[0]])]
        actions_q_values = self.predict_function(state)[0]
        return actions_q_values

    def save_Q(self, file_name):  # save table
        self.train_model.save(file_name)

    def load_Q(self, file_name):  # save table
        model = load_model(file_name)
        self.train_model = model
        self.predict_function = K.function(model.inputs, model.outputs)
