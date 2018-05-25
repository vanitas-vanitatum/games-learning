import numpy as np
import random
from src.game.nxn_tictactoe import Action
from collections import defaultdict
from src.game.player import Player


class QPlayer(Player):
    def __init__(self, q_initial_value=0.0):
        super().__init__('qplayer')
        self.Q = defaultdict(lambda:
                             defaultdict(lambda:
                                         float(q_initial_value)))
        self.q_initial_value = q_initial_value

        self.last_state_action_as_X = None, None
        self.last_state_action_as_O = None, None

    def move(self, board, rand=False):
        if rand:
            action = self.get_random_action_for_state(board)
        else:
            action = self.get_max_action_for_state(board)
        if board.moving_player == board.X:
            self.last_state_action_as_X = (board.board_state(), action)
        else:
            self.last_state_action_as_O = (board.board_state(), action)
        return action

    def get_max_action_for_state(self, state):
        Q_s = self.get_action_value_for_state(state)
        mx_q = max(Q_s.values())
        best_moves = [k for k in Q_s if Q_s[k] == mx_q]
        return random.choice(best_moves)

    def update_Q(self, reward, new_board, learning_rate, temporary_discount):
        if new_board.moving_player == new_board.X:
            previous_state, action = self.last_state_action_as_X
        else:
            previous_state, action = self.last_state_action_as_O

        q_previous = self.Q[previous_state][action]
        possible_qs_next = self.get_possible_q_values_for_board(new_board)
        if possible_qs_next:
            max_q_new = max(possible_qs_next)
        else:
            max_q_new = 0

        updated_value = (q_previous
                         + (learning_rate
                            * (reward
                               + (temporary_discount
                                  * max_q_new)
                               - q_previous)))

        self.Q[previous_state][action] = updated_value
        # self.temporary_discount *= self.discount

    def get_possible_q_values_for_board(self, board):
        possible_moves = board.get_legal_moves()
        actions_q_values = self.Q[board.board_state()]
        q_values = list(actions_q_values.values())
        if len(possible_moves) > len(actions_q_values):
            q_values.append(self.q_initial_value)
        return q_values

    def get_action_value_for_state(self, board):
        result = {Action(row, col): self.q_initial_value
                  for row, col in board.get_legal_moves()}
        result.update(self.Q[board.board_state()])
        return result

    def get_max_q_value_for_state(self, board):
        return self._get_max_min_q_value_for_state(board, max)

    def _get_max_min_q_value_for_state(self, board, maxmin=max):
        all_actions = [Action(row, col) for row, col in board.get_legal_moves()]
        max_value = -np.inf if maxmin == max else np.inf
        state = board.board_state()
        for act in all_actions:
            max_value = maxmin(max_value, self.Q[state][act])
        return max_value

    def get_random_action_for_state(self, board):
        all_actions = [Action(row, col) for row, col in board.get_legal_moves()]
        return random.choice(all_actions)

    def add_value(self, state, action, value):
        self.Q[state][action] += value

    def set_value(self, state, action, value):
        self.Q[state][action] = value

    def get_q_value(self, state, action):
        return self.Q[state][action]

    def reward(self, value, board):
        pass
