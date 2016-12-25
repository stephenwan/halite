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
    if move[0] != 0 and move[1] != 0:
        move = random.choice([np.array([x, y]) for x in [0, move[0]] for y in [0, move[1]] if not (x == 0 and y == 0) ])
    for i, m in enumerate(BasicMoves):
        if np.array_equal(m, move):
            return i
    return STILL

def _idx(loc):
    return (loc[1], loc[0]);


S_INIT, S_MOVED, S_DECIDED = (0, 1, 2)
STR_THRESHOLD_MIN, STR_THRESHOLD_MAX = (10, 60)

class GameData:
    def __init__(self, myID, logger):
        self.logger = logger
        self.me = myID


    def load_game_map(self, game_map):
        self.w, self.h = (game_map.width, game_map.height)
        self.shape = np.array((self.w, self.h))
        self.zeros = np.zeros((self.h, self.w), dtype=int)
        self.owns = np.empty((self.h, self.w), dtype=bool)
        self.prods = np.empty((self.h, self.w), dtype=int)
        self.strs = np.empty((self.h, self.w), dtype=int)
        self.bds = np.empty((self.h, self.w), dtype=bool)
        self.states = np.zeros((self.h, self.w), dtype=int)
        for square in game_map:
            x, y, owner, strength, production = square;
            self.owns[y, x] = owner == self.me
            self.prods[y, x] = productionOg
            self.strs[y, x] = strength
            self.bds[y, x] = self.owns[y, x] and any([s.owner != self.me for s in game_map.neighbors(square)])

        self.n_owns = np.sum(self.owns)
        self.n_bds = np.sum(self.bds)
        self.mystrs = np.where(self.owns, self.strs, self.zeros)
        self.mustmoves = self.mystrs > STR_THRESHOLD_MAX
        self.n_mustmoves = np.sum(self.mustmoves)
        self.locs_bd = self.truthy_locs(self.bds)
        self.loc_mustmoves = self.truthy_locs(self.mustmoves)
        self.loc_owns = self.truthy_locs(self.owns)

    def truthy_locs(self, boolboard):
        return np.array([np.array([x, y]) for y, x in zip(* np.where(boolboard))])



class GameMaster:
    def __init__(self, myID, game_map, logger=None):
        self.d = GameData(myID, logger)
        self.d.load_game_map(game_map)
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

    def nearestbd(self, loc):
        factor = max(self.d.n_bds / 8, 1)
        bds = self.d.locs_bd[::factor]
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

    def conquer(self):
        moves = []
        for loc in self.d.locs_bd:
             possible_locs = [n
                              for n in self.neighbour_locs(loc)
                              if (not self.d.owns[_idx(n)]) and self.stronger(loc, n)]
             if len(possible_locs) > 0:
                 move_to_loc = random.choice(possible_locs)
                 moves.append(Move(self.get_square(loc), dir_from_move(self.get_move(loc, move_to_loc))))
                 self.d.states[_idx(loc)] = S_MOVED
        return moves

    def strengthern_border(self):
        moves = []
        prods_on_border = [ self.d.prods[_idx(loc)] for loc in self.d.locs_bd]
        sorted_indexes = np.argsort(prods_on_border, kind='mergesort')

        for i in sorted_indexes[::-1]:
            loc = self.d.locs_bd[i]
            if self.d.states[_idx(loc)] != S_INIT:
                continue
            neighbors = self.neighbour_locs(loc)
            for neighbor in neighbors:
                i = _idx(neighbor)
                if self.d.owns[i] and self.d.states[i] != S_MOVED :
                    if self.d.strs[i] > STR_THRESHOLD_MIN :
                        moves.append(Move(self.get_square(neighbor), dir_from_move(self.get_move(neighbor, loc))))
                        self.d.states[i] = S_MOVED
                    else:
                        self.d.states[i] = S_DECIDED

                    _neighbours = self.neighbour_locs(neighbor)
                    for _i in [ _idx(_n) for _n in _neighbours]:
                        if self.d.bds[_i] and self.d.states[_i] == S_INIT:
                            self.d.states[_i] = S_DECIDED
        return moves

    def call_to_arm(self):

        moves = []
        for loc in self.d.loc_mustmoves:
            i = _idx(loc)
            if self.d.states[i] == S_INIT:
                move_to_loc = self.nearestbd(loc)
                the_move = self.get_move(loc, move_to_loc)
                direction = dir_from_move(the_move)
                moves.append(Move(self.get_square(loc), direction))
                self.d.states[i] = S_MOVED

        return moves

    def farm(self):
        moves = []
        for loc in self.d.loc_owns:
            i = _idx(loc)
            if self.d.states[i] != S_MOVED:
                moves.append(Move(self.get_square(loc), STILL))
                self.d.states[i] = S_MOVED
        return moves


    def get_square(self, loc):
        return self.game_map.contents[loc[1]][loc[0]]


    def play(self):
        return self.conquer() + self.strengthern_border() + self.call_to_arm() + self.farm()
