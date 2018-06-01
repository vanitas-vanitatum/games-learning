from src.game.tictactoe import TicTacToe
from src.game.board import Board
from src.game.player import HumanPlayer, RandomPlayer
from src.deep_q_learning import DeepQPlayer, DeepLearner
from src.deep_q_learning.model import get_model_3x3

qA = DeepQPlayer(get_model_3x3())
starters, winners = DeepLearner(TicTacToe(Board(3, 3), qA, qA), batch_size=128, discount=0.9, memory_size=3000).fit(50000)
qA.save_Q('model.h5')
import ipdb
ipdb.set_trace()
t = TicTacToe(Board(3, 3), HumanPlayer(), qA)
t.play_game()
