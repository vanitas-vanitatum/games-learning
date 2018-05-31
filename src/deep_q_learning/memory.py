from collections import namedtuple, deque

import numpy as np

from src.game.action import Action
from src.game.board import Board

Transition = namedtuple('Transition',
                        ['previous_state', 'action', 'next_state', 'reward', 'player', 'is_terminal_state'])


class ReplayMemory:
    def __init__(self, size, board_size=3, seed=None):
        self.size = size
        self.board_size = board_size

        self.player_x = deque(maxlen=self.size)
        self.player_o = deque(maxlen=self.size)

        self._rng = np.random.RandomState(seed)

    def add_result(self, previous_state, action, next_state, reward, player, is_terminal):
        transition = Transition(previous_state, action, next_state, reward, player, is_terminal)
        if player == Board.X:
            self.player_x.append(transition)
        else:
            self.player_o.append(transition)

    def add_transition(self, transition, with_transforms=True):
        player = transition.player
        transition = self.transform_transitions(transition) if with_transforms else [transition]
        if player == Board.X:
            for t in transition:
                self.player_x.append(t)
        else:
            for t in transition:
                self.player_o.append(t)

    def _rotate_action(self, action, rot90_counter):
        row, col = action // self.board_size, action % self.board_size
        rot90_counter %= 4
        if rot90_counter == 0:
            return row * self.board_size + col
        elif rot90_counter == 1:
            return col * self.board_size + self.board_size - row - 1
        elif rot90_counter == 2:
            return self.board_size * (self.board_size - row - 1) + self.board_size - col - 1
        elif rot90_counter == 3:
            return self.board_size * (self.board_size - col - 1) + row

    def _rotate_board(self, board, rot90_counter):
        board = board.reshape((self.board_size, self.board_size))
        board = np.rot90(board, rot90_counter, axes=(1, 0))
        return board.flatten()

    def transform_transitions(self, base_transition):
        output = []
        previous_state, action, next_state, reward, player, is_terminal_state = base_transition
        previous_state_board, previous_state_player_info = previous_state[:-1], previous_state[-1:]
        next_state_board, next_state_player_info = next_state[:-1], next_state[-1:]
        output.append(base_transition)
        for i in range(1, 4):
            rotated_previous_board = self._rotate_board(previous_state_board, i)
            rotated_next_board = self._rotate_board(next_state_board, i)

            rotated_action = self._rotate_action(action, i)
            rotated_previous_state = np.concatenate((rotated_previous_board, previous_state_player_info))
            rotated_next_state = np.concatenate((rotated_next_board, next_state_player_info))

            output.append(Transition(rotated_previous_state, rotated_action, rotated_next_state, reward, player,
                                     is_terminal_state))
        return output

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
