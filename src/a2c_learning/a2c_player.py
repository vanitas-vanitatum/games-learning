import random

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model

from src.a2c_learning.a2c_models import create_policy_network, create_value_network
from src.game.action import Action
from src.game.board import Board
from src.qlearning import QPlayer
from src.utils import logger

LEARNING_RATE = 1e-3


def get_dummy_action_input(length):
    return np.empty((length,))


class A2CPlayer(QPlayer):
    def __init__(self):
        super().__init__(0.0)
        self.sess = K.get_session()

        self.policy_network, self.policy_network_input = create_policy_network()
        self.value_network, self.value_network_input = create_value_network()

        self.policy_network_weights = self.policy_network.trainable_weights
        self.value_network_weights = self.value_network.trainable_weights

        self.current_reward = tf.placeholder(tf.float32)
        self.policy_loss = tf.log(self.policy_network.outputs[0] + K.epsilon()) * (self.current_reward - self.value_network.outputs[0])
        self.policy_network_grads = tf.gradients(self.policy_loss,
                                                 self.policy_network_weights)

        self.value_loss = tf.square(self.current_reward - self.value_network.outputs[0])
        self.value_network_grads = tf.gradients(self.value_loss,
                                                self.value_network_weights)

        self.total_policy_network_grads = [tf.placeholder(tf.float32, grad.get_shape()) for grad in self.policy_network_grads]
        self.total_value_network_grads = [tf.placeholder(tf.float32, grad.get_shape()) for grad in self.value_network_grads]

        self.policy_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).apply_gradients(
            zip(self.total_policy_network_grads, self.policy_network_weights))
        self.value_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).apply_gradients(
            zip(self.total_value_network_grads, self.value_network_weights))

    def get_last_state_action_as_player(self, as_player):
        return self.last_state_action_as_X if as_player == Board.X else self.last_state_action_as_O

    def get_max_action_for_state(self, board):
        q_s = self.policy_network.predict(np.array([board.board_state()]))[0]
        actions = [Action(ind // board.n, ind % board.n) for ind in range(len(q_s))]
        actions_mask = board.get_legal_moves_mask()
        q_s[np.logical_not(actions_mask)] = -100000
        mx_q = np.max(q_s)
        best_moves = [action for action, q_value in zip(actions, q_s) if q_value == mx_q]
        chosen = random.choice(best_moves)
        return chosen

    def update_models(self, transitions_batch, discount):
        if len(transitions_batch) == 0:
            return
        transitions_batch = transitions_batch[::-1]
        current_reward = transitions_batch[0].reward
        accumulated_policy_grads = [
            np.zeros(tensor.shape) for tensor in self.policy_network_weights
        ]
        accumulated_value_grads = [
            np.zeros(tensor.shape) for tensor in self.value_network_weights
        ]

        cummulative_policy_loss = []
        cummulative_value_loss = []
        for transition in transitions_batch[1:]:
            current_reward = transition.reward + discount * current_reward
            current_state = transition.previous_state
            policy_gradients, policy_loss = self.sess.run(
                [self.policy_network_grads, self.policy_loss],
                feed_dict={
                    self.policy_network_input: [current_state],
                    self.value_network_input: [current_state],
                    self.current_reward: current_reward
                }
            )
            value_gradients, value_loss = self.sess.run(
                [self.value_network_grads, self.value_loss],
                feed_dict={
                    self.value_network_input: [current_state],
                    self.current_reward: current_reward
                }
            )

            accumulated_policy_grads = [
                acc + grad for acc, grad in zip(accumulated_policy_grads, policy_gradients)
            ]

            accumulated_value_grads = [
                acc + grad for acc, grad in zip(accumulated_value_grads, value_gradients)
            ]
            cummulative_policy_loss.append(policy_loss)
            cummulative_value_loss.append(value_loss)

        self.sess.run(
            self.policy_optimizer,
            feed_dict={
                acc: grad for acc, grad in zip(self.total_policy_network_grads, accumulated_policy_grads)
            }
        )
        self.sess.run(
            self.value_optimizer,
            feed_dict={
                acc: grad for acc, grad in zip(self.total_value_network_grads, accumulated_value_grads)
            }
        )

        logger.add_tf_summary_with_last_episode(
            tf.Summary(
                value=[tf.Summary.Value(tag='loss/policy', simple_value=
                                        np.mean(cummulative_policy_loss)),
                       tf.Summary.Value(tag='loss/value', simple_value=
                                        np.mean(cummulative_value_loss))]),
        )

    def get_possible_q_values_for_board(self, board):
        actions_q_values = self.predict_function(
            [
                np.array([board.board_state()])
            ]
        )[0]
        return actions_q_values

    def save_Q(self, file_name):  # save table
        self.policy_network.save('policy_network.h5')
        self.value_network.save('value_network.h5')

    def load_Q(self, file_name):  # save table
        self.policy_network.set_weights(load_model('policy_network.h5'))
        self.value_network.set_weights(load_model('value_network.h5'))


if __name__ == '__main__':
    A2CPlayer()
