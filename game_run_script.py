NxN = True
from pprint import pprint

if NxN:
    from src.game.nxn_tictactoe import TicTacToe, Board
    from src.game.player import HumanPlayer, RandomPlayer, MinMaxPlayer, QPlayer
    from src.qlearning import Learner

    q = QPlayer(0)
    Learner(TicTacToe(Board(3, 3), q, q), q).fit(10000)
    pprint(q.state_action_mapping)
    print(len(q.state_action_mapping.keys()))
    t = TicTacToe(Board(3, 3), HumanPlayer(), q)
    t.play_game()
else:
    from src.game.tictactoe import TicTacToe
    from src.game.player import HumanPlayer, RandomPlayer, MinMaxPlayer
    
    t = TicTacToe(HumanPlayer(), MinMaxPlayer())
    t.play_game()

