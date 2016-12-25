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


S_INIT, S_MOVED, S_GATHER_HERE, S_NOT_GATHER_HERE = (0, 1, 2, 3)
STR_THRESHOLD_MIN, STR_THRESHOLD_MAX, STR_BD_MAX, STR_OVERSHOOT = (10, 60, 210, 50)
BF_RADIUS = 10

class GameData:
    def __init__(self, myID, logger):
        self.logger = logger
        self.me = myID


    def load_game_map(self, game_map):
        self.w, self.h = (game_map.width, game_map.height)
        self.shape = np.array((self.w, self.h))
        self.zeros = np.zeros((self.h, self.w), dtype=int)
        self.owns = np.empty((self.h, self.w), dtype=bool)
        self.owners = np.empty((self.h, self.w), dtype=int)
        self.prods = np.empty((self.h, self.w), dtype=int)
        self.strs = np.empty((self.h, self.w), dtype=int)
        self.bds = np.empty((self.h, self.w), dtype=bool)
        self.strategicspots = np.empty((self.h, self.w), dtype=bool)
        self.bdstrdiffs = np.zeros((self.h, self.w), dtype=int)
        self.hostilestrs = np.zeros((self.h, self.w), dtype=int)
        self.states = np.zeros((self.h, self.w), dtype=int)

        for square in game_map:
            x, y, owner, strength, production = square;
            self.owns[y, x] = owner == self.me
            self.prods[y, x] = production
            self.strs[y, x] = strength
            self.owners[y, x] = owner

        for square in game_map:
            x, y, owner, strength, production = square;
            unowned_neighbors = [ s for s in game_map.neighbors(square) if s.owner != self.me ]
            if self.owns[y, x]:
                self.bds[y, x] = len(unowned_neighbors) > 0
                self.strategicspots[y, x] = False
            else:
                self.bds[y, x] = False
                self.strategicspots[y, x] = len(unowned_neighbors) < 3

            if self.bds[y, x]:
                self.bdstrdiffs[y, x] = self.strs[y, x] - max([ self.strs[s.y, s.x] for s in unowned_neighbors ])

            nearby_enemies = [ s for s in game_map.neighbors(square, n=2) if s.owner != self.me and s.owner != 0 ]
            if len(nearby_enemies) > 0:
                self.hostilestrs[y, x] = max([ self.strs[s.y, s.x] for s in nearby_enemies ])

        self.mystrs = np.where(self.owns, self.strs, self.zeros)
        self.mustmoves = self.mystrs > STR_THRESHOLD_MAX

        self.loc_bds = self.truthy_locs(self.bds)
        self.loc_strategicspots = self.truthy_locs(self.strategicspots)
        self.loc_mustmoves = self.truthy_locs(self.mustmoves)
        self.loc_owns = self.truthy_locs(self.owns)
        self.loc_bfs = self.truthy_locs( self.hostilestrs > 0 )

        self.n_strategicspots = len(self.loc_strategicspots)
        self.n_mustmoves = len(self.loc_mustmoves)
        self.n_owns = len(self.loc_owns)
        self.n_bds = len(self.loc_bds)
        self.n_bfs = len(self.loc_bfs)

        factor = max(int(self.n_bds / 8), 1)
        self.loc_sampled_bds = self.loc_bds[::factor]

    def truthy_locs(self, boolboard):
        return np.array([np.array([x, y]) for y, x in zip(* np.where(boolboard))])



class GameMaster:
    def __init__(self, myID, game_map, logger=None):
        self.d = GameData(myID, logger)
        self.d.load_game_map(game_map)
        self.game_map = game_map
        self.logger = logger

    def neighbour_locs(self, loc, only_owned=False):
        locs = [ (loc + m) % self.d.shape for m in BasicMoves]
        if only_owned:
            return [ l for l in locs if self.d.owns[_idx(l)]]
        else:
            return locs

    def distance(self, loc1, loc2):
        interm = np.minimum( loc1 + self.d.shape - loc2, loc2 + self.d.shape - loc1)
        d = np.minimum(np.abs( loc1 - loc2), interm)
        return d[0] + d[1]

    def nearestbd(self, loc):
        distances = np.array([ self.distance(loc, l) for l in self.d.loc_sampled_bds])
        return self.d.loc_sampled_bds[np.argmin(distances)]

    def nearestbf(self, loc):
        if self.d.n_bfs == 0:
            return None

        distances = np.array([self.distance(loc, l) for l in self.d.loc_bfs])
        min_index = np.argmin(distances)
        if distances[min_index] <= BF_RADIUS:
            return self.d.loc_bfs[min_index]
        else:
            return None


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

    def capture_strategicspots(self):
        moves = []
        if self.d.n_strategicspots == 0:
            return moves

        for loc in self.d.loc_strategicspots:
            bds = [ l for l in self.neighbour_locs(loc) if self.d.owns[ _idx(l)]]

            bds_with_sufficient_str = [ l for l in bds if self.d.strs[_idx(l)] > self.d.strs[_idx(loc)]]
            strs = [ self.d.strs[_idx(l)] for l in bds_with_sufficient_str ]

            if len(bds_with_sufficient_str) > 0:
                t = bds_with_sufficient_str[np.argmin(strs)]
                moves.append(Move(self.get_square(t), dir_from_move(self.get_move(t, loc))))
                self.d.states[_idx(t)] = S_MOVED
            else:
                if sum([ self.d.strs[_idx(bd)] for bd in bds ]) > self.d.strs[_idx(loc)]:
                    for bd in bds:
                        moves.append(Move(self.get_square(bd), dir_from_move(self.get_move(bd, loc))))
                        self.d.states[_idx(bd)] = S_MOVED
                else:
                    for bd in bds:
                        self.d.states[_idx(bd)] = S_GATHER_HERE

        return moves

    def conquer(self):
        moves = []
        for loc in self.d.loc_bds:
            iloc = _idx(loc)
            if self.d.states[iloc] != S_INIT:
                continue

            targets = [n
                       for n in self.neighbour_locs(loc)
                       if not self.d.owns[_idx(n)]]

            if self.d.strs[iloc] > STR_BD_MAX:
                target = random.choice(targets)
                moves.append(Move(self.get_square(loc), dir_from_move(self.get_move(loc, target))))
                self.d.states[iloc] = S_MOVED
            else:
                beatable_targets = [ t for t in targets if self.d.strs[iloc] > self.d.strs[_idx(t)]]
                if len(beatable_targets) > 0:
                    if (self.d.hostilestrs[iloc] == 0
                        or self.d.strs[iloc] - self.d.hostilestrs[iloc] > self.d.bdstrdiffs[iloc]):
                        target = random.choice(beatable_targets)
                        moves.append(Move(self.get_square(loc), dir_from_move(self.get_move(loc, target))))
                        self.d.states[iloc] = S_MOVED
        return moves

    def gather_here(self, loc):
        moves = []
        for neighbor in self.neighbour_locs(loc, only_owned=True):
            i = _idx(neighbor)
            if self.d.states[i] == S_INIT:
                if self.d.strs[i] > STR_THRESHOLD_MIN and self.d.strs[i] + self.d.strs[_idx(loc)] < 250 + STR_OVERSHOOT:
                    moves.append(Move(self.get_square(neighbor), dir_from_move(self.get_move(neighbor, loc))))
                    self.d.states[i] = S_MOVED
                else:
                    self.d.states[i] = S_NOT_GATHER_HERE
                secondary_neighbours = self.neighbour_locs(neighbor, only_owned=True)
                for _i in [ _idx(_n) for _n in secondary_neighbours]:
                    if self.d.bds[_i] and self.d.states[_i] == S_INIT:
                        self.d.states[_i] = S_NOT_GATHER_HERE
        return moves

    def strengthern_border(self):
        moves = []

        for loc in self.d.loc_bds:
            if self.d.states[_idx(loc)] == S_GATHER_HERE:
                moves = moves + self.gather_here(loc)

        other_bds = [ l for l in self.d.loc_bds if self.d.states[_idx(l)] == S_INIT ]
        bddiffs = [ self.d.bdstrdiffs[_idx(l)] for l in other_bds]
        sorted_indexes = np.argsort(bddiffs, kind='mergesort')

        for i in sorted_indexes[::-1]:
            loc = other_bds[i]
            if self.d.states[_idx(loc)] != S_INIT:
                continue
            moves = moves + self.gather_here(loc)

        return moves

    def call_to_arm(self):
        moves = []
        for loc in self.d.loc_mustmoves:
            i = _idx(loc)
            if self.d.states[i] == S_INIT:
                nearestbf = self.nearestbf(loc)
                if nearestbf is not None:
                    move_to_loc = nearestbf
                else:
                    move_to_loc = self.nearestbd(loc)

                if self.distance(loc, move_to_loc) > 1 or self.d.strs[i] + self.d.strs[_idx(move_to_loc)] < 250 + STR_OVERSHOOT:
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
        return self.capture_strategicspots() + self.conquer() + self.strengthern_border() + self.call_to_arm() + self.farm()
