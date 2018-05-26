from src.game.tictactoe import TicTacToe
from src.game.board import Board
from src.game.player import HumanPlayer, RandomPlayer
from src.qlearning import QPlayer, Learner

qA = QPlayer(1)
starters, winners = Learner(TicTacToe(Board(3, 3), qA, qA)).fit(10000)
t = TicTacToe(Board(3, 3), HumanPlayer(), qA)
t.play_game()
