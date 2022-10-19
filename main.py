from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax

class GameController(TwoPlayerGame):
    def __init__(self, players, size):
        self.size = size
        num_pawns, len_board = size
        p = [[(i, j) for j in range(len_board)] for i in [0, num_pawns - 1]]

        for i, d, goal, pawns in [(0, 1, num_pawns - 1, p[0]), (1, -1, 0, p[1])]:
            players[i].direction = d
            players[i].goal_line = goal
            players[i].pawns = pawns

        # Define the players
        self.players = players

        # Define who starts first
        self.current_player = 1

        # Define the alphabets
        self.alphabets = 'ABCDEFGHIJ'

        # Convert B4 to (1, 3)
        self.to_tuple = lambda s: (self.alphabets.index(s[0]), int(s[1:]) - 1)

        # Convert (1, 3) to B4
        self.to_string = lambda move: ' '.join([self.alphabets[
                move[i][0]] + str(move[i][1] + 1)
                for i in (0, 1)])

    # definicja możliwych ruchów
    def possible_moves(self):
        moves = []
        opponent_pawns = self.opponent.pawns
        d = self.player.direction

        for i, j in self.player.pawns:
            if (i + d, j) not in opponent_pawns:
                moves.append(((i, j), (i + d, j)))

            if (i + d, j + 1) in opponent_pawns:
                moves.append(((i, j), (i + d, j + 1)))

            if (i + d, j - 1) in opponent_pawns:
                moves.append(((i, j), (i + d, j - 1)))

        return list(map(self.to_string, [(i, j) for i, j in moves]))

    # definicja wykonania ruchu
    def make_move(self, move):
        move = list(map(self.to_tuple, move.split(' ')))
        ind = self.player.pawns.index(move[0])
        self.player.pawns[ind] = move[1]

        if move[1] in self.opponent.pawns:
            self.opponent.pawns.remove(move[1])

    # defnicja przegranej
    def loss_condition(self):
        return (any([i == self.opponent.goal_line
                for i, j in self.opponent.pawns])
                or (self.possible_moves() == []) )

    # sprawdzenie czy gra się skończyła
    def is_over(self):
        return self.loss_condition()

    # wyświetlenie planszy
    def show(self):
        def f(x): return '1' if x in self.players[0].pawns else ('2' if x in self.players[1].pawns else '.')

        print("\n".join([" ".join([f((i, j)) for j in range(self.size[1])]) for i in range(self.size[0])]))


if __name__ == '__main__':
    # wynik: 0 przegrana, 1 gramy dalej
    def scoring(game): return 0 if game.loss_condition() else 1

    # definicja algorytmu, z wielkością drzewa wykonywanych ruchów
    algorithm = Negamax(9, scoring)

    board_size = 5
    # stworzenie gry
    game = GameController([AI_Player(algorithm), AI_Player(algorithm)], (board_size, board_size))
    game.play()

    print('\nPlayer', game.opponent_index, 'wins after', game.nmove-1, 'turns')
    input()

