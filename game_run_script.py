NxN = True

if NxN:
    from src.game.nxn_tictactoe import TicTacToe, Board
    from src.game.player import HumanPlayer, RandomPlayer, MinMaxPlayer

    t = TicTacToe(HumanPlayer(), RandomPlayer())
    t.play_game()
else:
    from src.game.tictactoe import TicTacToe
    from src.game.player import HumanPlayer, RandomPlayer, MinMaxPlayer
    
    t = TicTacToe(HumanPlayer(), MinMaxPlayer())
    t.play_game()

