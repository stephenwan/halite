import numpy as np
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random
import math

BasicMoves = [np.array(m) for m in ((0, -1), (1, 0), (0, 1), (-1, 0))]

def move_from_dir(direction):
    if direction == STILL:
        return np.array((0, 0))
    return BasicMoves[direction]

def dir_from_move(move):
    for i, m in enumerate(BasicMoves):
        if np.array_equal(m, move):
            return i
    return STILL

def _idx(loc):
    return (loc[1], loc[0]);


class GameData:
    def __init__(self, myID, game_map, logger):
        self.logger = logger
        self.w, self.h = (game_map.width, game_map.height)
        self.me = myID
        self.shape = np.array((self.w, self.h))
        self.zeros = np.zeros((self.h, self.w), dtype=int)
        self.owns = np.empty((self.h, self.w), dtype=bool)
        self.prods = np.empty((self.h, self.w), dtype=int)
        self.strs = np.empty((self.h, self.w), dtype=int)
        self.bds = np.empty((self.h, self.w), dtype=bool)
        self.mustmoves = np.empty((self.h, self.w), dtype=bool)

        for square in game_map:
            x, y, owner, strength, production = square;
            self.owns[y, x] = owner == self.me
            self.prods[y, x] = production
            self.strs[y, x] = strength
            self.bds[y, x] = self.owns[y, x] and any([s.owner != self.me for s in game_map.neighbors(square)])

        self.n_owns = np.sum(self.owns)
        self.n_bds = np.sum(self.bds)
        self.mystrs = np.where(self.owns, self.strs, self.zeros)
        self.mustmoves = self.mystrs > 100
        self.n_mustmoves = np.sum(self.mustmoves)

    def nstrongest(self, n, exclude_bd=False):
        strs = self.mystrs.copy()
        if exclude_bd:
            strs[self.bds] = -1

        cutoff_mark = np.sort(strs.flatten())[-n]
        return [np.array([x, y]) for y, x in zip(*np.where(strs >= cutoff_mark)) if strs[y, x] > 0]




class GameMaster:
    def __init__(self, myID, game_map, logger=None):
        self.d = GameData(myID, game_map, logger)
        self.game_map = game_map
        self.logger = logger

    def neighbour_locs(self, loc):
        return [ (loc + m) % self.d.shape for m in BasicMoves]

    def stronger(self, loc1, loc2):
        return self.d.strs[_idx(loc1)] > self.d.strs[_idx(loc2)]

    def distance(self, loc1, loc2):
        interm = np.minimum( loc1 + self.d.shape - loc2, loc2 + self.d.shape - loc1)
        d = np.minimum(np.abs( loc1 - loc2), interm)
        return d[0] + d[1]

    def nearestbd(self, loc, factor=1):
        bds = np.array([np.array([x, y]) for y, x in zip(*np.where(self.d.bds))])[::factor]
        distances = np.array([ self.distance(loc, l) for l in bds])
        return bds[np.argmin(distances)]

    def get_move(self, loc_from, loc_to):
        d1 = loc_to - loc_from
        d2 = np.where(loc_to > loc_from, loc_to - (loc_from + self.d.shape), loc_to + self.d.shape - loc_from)
        move = np.where(np.abs(d1) < np.abs(d2), d1, d2)
        zero = np.array([0, 0])
        one = np.array([1, 1])
        mone = np.array([-1, -1])
        move = np.where(move > zero, one, move)
        move = np.where(move < zero, mone, move)
        return move

    def move_bd(self, loc):
        possible_locs = [n
                         for n in self.neighbour_locs(loc)
                         if (not self.d.owns[_idx(n)]) and self.stronger(loc, n)]
        if len(possible_locs) > 0:
            move_to_loc = random.choice(possible_locs)
            return dir_from_move(self.get_move(loc, move_to_loc))
        return STILL

    def move_in(self, loc, candidates, factor=1):
        if any([ np.array_equal(loc, c) for c in candidates]):
            move_to_loc = self.nearestbd(loc, factor)
            return dir_from_move(self.get_move(loc, move_to_loc))
        return STILL

    def get_interior_candidates_number(self):
        ni = self.d.n_owns - self.d.n_bds
        if ni == 0:
            return 0
        return math.floor( ni / 4.0) + 1


    def move_my_locs(self):
        moves = []
        n_ic = self.get_interior_candidates_number()
        n_bdguards = 8;

        factor = max(self.d.n_bds / n_bdguards, 1)

        if n_ic > self.d.n_mustmoves:
            candidates = self.d.nstrongest(n_ic, True)
        else:
            candidates = [np.array(x, y) for y, x in zip(*np.where(self.d.mustmoves))]

        for y, x in zip(*np.where(self.d.owns)):
            square = self.game_map.contents[y][x]
            loc = np.array([x, y])
            if self.d.bds[_idx(loc)]:
                move = Move(square, self.move_bd(loc))
            else:
                move = Move(square, self.move_in(loc, candidates, factor))
            moves.append(move)
        return moves
