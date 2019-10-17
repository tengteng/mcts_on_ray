import hashlib
import random


class State:
    NUM_TURNS = 10
    GOAL = 0
    MOVES = [2, -2, 3, -3]
    MAX_MOVE = len(MOVES)
    MAX_VALUE = (5.0 * (NUM_TURNS - 1) * NUM_TURNS) / 2
    num_moves = len(MOVES)

    def __init__(self, value=0, moves=[], turn=NUM_TURNS):
        self.value = value
        self.turn = turn
        self.moves = moves

    def next_state(self):
        nextmove = random.choice([x * self.turn for x in self.MOVES])
        next = State(self.value + nextmove, self.moves + [nextmove],
                     self.turn - 1)
        return next

    def terminal(self):
        if self.turn == 0:
            return True
        return False

    def reward(self):
        r = 1.0 - (abs(self.value - self.GOAL) / self.MAX_VALUE)
        return r

    def __hash__(self):
        return int(
            hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        s = "Value: %d; Moves: %s" % (self.value, self.moves)
        return s
