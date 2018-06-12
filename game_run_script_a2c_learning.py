from src.game.tictactoe import TicTacToe
from src.game.board import Board
from src.game.player import HumanPlayer
from src.a2c_learning import A2CLearner, A2CPlayer

qA = A2CPlayer()
qB = A2CPlayer()
starters, winners = A2CLearner(TicTacToe(Board(3, 3), qA, qB), qA, qB, discount=0.7).fit(50000)
import ipdb
ipdb.set_trace()
t = TicTacToe(Board(3, 3), HumanPlayer(), qB)
t.play_game()
