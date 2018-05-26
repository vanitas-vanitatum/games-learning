from src.game.tictactoe import TicTacToe
from src.game.board import Board
from src.game.player import HumanPlayer, RandomPlayer
from src.deep_q_learning import DeepQPlayer, DeepLearner
from src.deep_q_learning.model import get_model_3x3

qA = DeepQPlayer(get_model_3x3())
starters, winners = DeepLearner(TicTacToe(Board(3, 3), qA, qA), memory_size=100).fit(1000)
qA.save_Q('model.h5')
t = TicTacToe(Board(3, 3), HumanPlayer(), qA)
t.play_game()
