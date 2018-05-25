NxN = True

if NxN:
    from src.game.tictactoe import TicTacToe
    from src.game.board import Board
    from src.game.player import HumanPlayer, RandomPlayer
    from src.qlearning import QPlayer, Learner

    q = QPlayer(1)
    #q2 = QPlayer(1)
    Learner(TicTacToe(Board(3, 3), q, q), q).fit(10000)
    #pprint(q.Q)
    #print(len(q.Q.keys()))
    t = TicTacToe(Board(3, 3), HumanPlayer(), q)
    t.play_game()
else:
    from src.game.tictactoe import TicTacToe
    from src.game.player import HumanPlayer, RandomPlayer
    
    t = TicTacToe(HumanPlayer(), RandomPlayer())
    t.play_game()

