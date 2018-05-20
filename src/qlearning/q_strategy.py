from collections import defaultdict

import numpy as np


class QStrategy:
    def __init__(self, q_initial_value=0.0):
        self.state_action_mapping = defaultdict(lambda: float(q_initial_value))

    def get_max_action(self):
        best_value = -np.inf
        best_action = None
        for (action, state), value in self.state_action_mapping:
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

    def add_value(self, state, action, value):
        self.state_action_mapping[(state, action)] += value

    def set_value(self, state, action, value):
        self.state_action_mapping[(state, action)] = value
