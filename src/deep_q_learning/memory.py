from collections import namedtuple, deque, Counter
from src.game.board import Board
from src.game.rewards import Reward
import numpy as np

Transition = namedtuple('Transition', ['previous_state', 'action', 'next_state', 'reward', 'player', 'is_terminal_state'])

PROBABS = {Reward.DRAW: 0.2,
           Reward.WIN: 0.3,
           Reward.LOOSE: 0.3,
           Reward.NONE: 0.1,
           Reward.ILLEGAL: 0.1}


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
        probs = np.asarray([PROBABS[x.reward] for x in collection])
        probs = probs/probs.sum()
        batch = np.random.choice(collection, size=batch_size, p=probs)
        return batch

    def is_enough_memory_for_players(self, batch_size):
        return len(self.player_x) >= batch_size and len(self.player_o) >= batch_size
