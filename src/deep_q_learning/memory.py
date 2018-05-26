from collections import namedtuple, deque
from src.game.board import Board

import numpy as np

Transition = namedtuple('Transition', ['previous_state', 'action', 'next_state', 'reward', 'player', 'is_terminal_state'])


class ReplayMemory:
    def __init__(self, size, seed=None):
        self.size = size

        self.player_x = deque(maxlen=self.size)
        self.player_o = deque(maxlen=self.size)

        self._rng = np.random.RandomState(seed)

    def add_result(self, previous_state, action, next_state, reward, player, is_terminal):
        transition = Transition(previous_state, action, next_state, reward, player, is_terminal)
        if player == Board.X:
            self.player_x.append(transition)
        else:
            self.player_o.append(transition)

    def add_transition(self, transition):
        if transition.player == Board.X:
            self.player_x.append(transition)
        else:
            self.player_o.append(transition)

    def sample_for_player(self, batch_size, player):
        if player == Board.X:
            collection = self.player_x
        else:
            collection = self.player_o
        sampled_indices = self._rng.randint(0, len(collection), size=batch_size)
        batch = []
        for ind in sampled_indices:
            value = collection[ind]
            batch.append(value)
        return batch

    def is_enough_memory_for_players(self, batch_size):
        return len(self.player_x) >= batch_size and len(self.player_o) >= batch_size
