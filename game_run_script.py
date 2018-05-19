from src.game.tictactoe import TicTacToe
from src.game.player import HumanPlayer, RandomPlayer, MinMaxPlayer


t = TicTacToe(HumanPlayer(), MinMaxPlayer())
t.play_game()