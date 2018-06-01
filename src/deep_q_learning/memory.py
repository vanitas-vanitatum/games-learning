from collections import namedtuple, deque, defaultdict

import numpy as np
import random

Transition = namedtuple('Transition', ['previous_state', 'action', 'next_state', 'reward', 'is_terminal_state'])


class ReplayMemory:
    def __init__(self, size, seed=None):
        self.size = size
        self.global_memory = deque(maxlen=self.size)
        self.reward_occurrences = defaultdict(float)
        self.reward_to_transitions = defaultdict(list)
        self._rng = np.random.RandomState(seed)

    def add_result(self, previous_state, action, next_state, reward, is_terminal):
        transition = Transition(previous_state, action, next_state, reward, is_terminal)
        self.global_memory.append(transition)

    def add_transition(self, transition):
        self.global_memory.append(transition)
        self.reward_occurrences[transition.reward] += 1.0
        self.reward_to_transitions[transition.reward].append(transition)

    def sample(self, batch_size):
        buffer = sorted(self.global_memory, key=lambda replay: abs(replay.reward), reverse=True)
        p = np.array([0.99 ** i for i in range(len(buffer))])
        p /= p.sum()
        sample_ixs = np.random.choice(np.arange(len(buffer)), size=batch_size, p=p)
        return [buffer[idx] for idx in sample_ixs]
        # batch = []
        # i = 0
        # keys = list(self.reward_to_transitions.keys())
        # while len(batch) < batch_size:
        #     batch.append(
        #         random.choice(self.reward_to_transitions[keys[i % len(keys)]])
        #     )
        #     i += 1
        # return batch
        # return random.sample(self.global_memory, batch_size)

    def is_enough_memory_for_players(self, batch_size):
        return len(self.global_memory) >= batch_size
