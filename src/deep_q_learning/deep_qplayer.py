import random

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model

from src.game.action import Action
from src.game.board import Board
from src.qlearning import QPlayer
from src.utils import logger

from src.deep_q_learning.model import create_actor_model, create_critic_model


def get_dummy_action_input(length):
    return np.empty((length,))


class DeepQPlayer(QPlayer):
    def __init__(self, model: Model, q_initial_value=0.0):
        super().__init__(q_initial_value)
        self.policy_net, self.predict_function = model
        self.target_net = Model(inputs=[self.policy_net.get_layer(name='input').input],
                                outputs=[self.policy_net.get_layer(name='output').output])
        self.target_net.trainable = False

        self.critic_model = create_critic_model()
        self.actor_model = create_actor_model()

    def get_last_state_action_as_player(self, as_player):
        return self.last_state_action_as_X if as_player == Board.X else self.last_state_action_as_O

    def get_max_action_for_state(self, board):
        q_s = self.target_net.predict(np.array([board.board_state()]))[0]
        actions = [Action(ind // board.n, ind % board.n) for ind in range(len(q_s))]
        actions_mask = board.get_legal_moves_mask()
        q_s[np.logical_not(actions_mask)] = -100000
        mx_q = np.max(q_s)
        best_moves = [action for action, q_value in zip(actions, q_s) if q_value == mx_q]
        chosen = random.choice(best_moves)
        return chosen

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)


    def _train_critic(self, samples, discount):
        for sample in samples:
            cur_state, action, reward, new_state, is_terminal = sample
            if not is_terminal:
                # target_action = self.
                pass

    def update_ac_model(self, transitions_batch, discount):
        self._train_critic(transitions_batch, discount)
        self._train_actor(transitions_batch)

    def update_q_model(self, transitions_batch, discount):
        previous_states, actions, next_states, rewards, terminals = zip(*transitions_batch)

        previous_states = np.array(previous_states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        mask_terminals = (1 - np.array(terminals)).astype(np.bool)
        next_state_values = np.zeros((len(previous_states),))
        if np.any(mask_terminals):
            next_state_values[mask_terminals] = np.max(self.target_net.predict(next_states[mask_terminals]), axis=1)

        outputs = rewards + next_state_values * discount
        loss = self.policy_net.train_on_batch([previous_states, outputs, actions], np.zeros((len(previous_states), 1)))

        logger.add_tf_summary_with_last_episode(
            tf.Summary(
                value=[tf.Summary.Value(tag='loss', simple_value=loss)]),
        )
        return loss

    def get_possible_q_values_for_board(self, board):
        actions_q_values = self.predict_function(
            [
                np.array([board.board_state()])
            ]
        )[0]
        return actions_q_values

    def refresh_target_net(self):
        self.target_net.set_weights(self.policy_net.get_weights())
        # self.target_net = Model(inputs=[self.policy_net.get_layer(name='input').input],
        #                         outputs=[self.policy_net.get_layer(name='output').output])
        # self.target_net.trainable = False

    def save_Q(self, file_name):  # save table
        self.policy_net.save(file_name)

    def load_Q(self, file_name):  # save table
        model = load_model(file_name)
        self.policy_net = model
        self.target_net = Model(inputs=[self.policy_net.get_layer(name='input').input],
                                outputs=[self.policy_net.get_layer(name='output').output])
        self.target_net.trainable = False
        self.predict_function = K.function(model.inputs, model.outputs)
