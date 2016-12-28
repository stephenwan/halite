import numpy as np
import math
from collections import namedtuple
import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square, GameMap

STR_MAX = 250
STR_NAN = -1
PROD_NAN = -1
IDX_NAN = -1
INT_TRUE = 1
INT_FALSE = 0

STATE_INIT, STATE_DONE = (1, 2)

class Geo:
    def __init__(self, h, w):
        self.h, self.w = (h, w)
        self.n = self.h * self.w
        self.init_geo()

    def add_logger(self, logger):
        self.logger = logger

    def init_geo(self):
        self.dim2d = np.array([ self.h, self.w])
        self.adjs = np.zeros((self.n, 9), dtype=int)
        _a = np.arange(self.n, dtype=int)
        self.adjs[:,0] = (_a - self.w + self.n) % self.n
        self.adjs[:,1] = np.where(_a % self.w == self.w - 1, _a - self.w + 1, _a + 1 )
        self.adjs[:,2] = (_a + self.w ) % self.n
        self.adjs[:,3] = np.where(_a % self.w == 0, _a + self.w - 1, _a - 1 )
        self.adjs[:,4] = _a
        self.adjs[:,5] = (self.adjs[:,1] - self.w + self.n) % self.n
        self.adjs[:,6] = (self.adjs[:,1] + self.w ) % self.n
        self.adjs[:,7] = (self.adjs[:,3] + self.w ) % self.n
        self.adjs[:,8] = (self.adjs[:,3] - self.w + self.n) % self.n
        self.locs = np.zeros((self.n, 2), dtype=int)
        self.locs[:,0] = _a // self.w
        self.locs[:,1] = _a % self.w

    def get_i(self, y, x):
        return y * self.w + x

    def get_container(self, t=int):
        return np.arange(self.n, dtype=t)

    def get_direction(self, loc_from, loc_to):
        if np.array_equal(loc_from, loc_to):
            return STILL

        d1 = loc_to - loc_from
        d2 = np.where(loc_to > loc_from, loc_to -(loc_from + self.dim2d), loc_to + self.dim2d - loc_from)
        (move_y, move_x) = np.where(np.abs(d1) < np.abs(d2), d1, d2)
        if abs(move_y) > abs(move_x):
            return [NORTH, SOUTH][ move_y > 0 ]
        else:
            return [EAST, WEST][ move_x < 0 ]


    def distance(self, p1, p2):
        loc1 = self.locs[p1]
        loc2 = self.locs[p2]
        size = np.array([self.h, self.w])
        tmp = np.minimum(loc1 + size - loc2, loc2 + size - loc1 )
        d = np.minimum(np.abs(loc1 - loc2), tmp)
        return sum(d)

    def move(ifrom, direction):
        return self.adjs[ifrom, direction]


class GameData:
    def __init__(self, myID, geo, owners, strengths, productions, game_map):
        self.me = myID
        self.owner_neutral = 0
        self.geo = geo
        self.owners = owners
        self.strs = strengths
        self.prods = productions
        self.squares = [ square for square in game_map ]
        self.logger = None


    def add_logger(self, logger):
         self.logger = logger

    def analyze(self):
        self.is_mine = self.owners == self.me
        self.is_neutual = self.owners == self.owner_neutral
        self.is_oppo = np.logical_and(self.owners != self.me, self.owners != self.owner_neutral)

        self.utils = {
            'lookup_str': np.vectorize(lambda i: self.strs[i] if i != IDX_NAN else STR_NAN),
            'lookup_prod': np.vectorize(lambda i: self.prods[i] if i != IDX_NAN else PROD_NAN),
            'lookup_is_not_mine': np.vectorize(lambda i: not self.is_mine[i]),
            'lookup_is_mine': np.vectorize(lambda i: self.is_mine[i])
        }

        self._build_boundary()
        self._build_sites_touch_mine()
        self._build_strength_difference_at_boundary()

        if self.logger is not None:
            self.logger.write("\nowners\n%s\n" % str( self.owners.reshape(self.geo.h, self.geo.w)))


    def _build_boundary(self):
        lookup_is_not_mine = self.utils['lookup_is_not_mine']
        t = lookup_is_not_mine(self.geo.adjs[:,0:4])
        self.is_bd = np.logical_and(np.any(t, axis=1), self.is_mine)
        self.is_interior = np.logical_and(np.logical_not(self.is_bd), self.is_mine)
        t_i = np.nonzero(self.is_bd)[0]
        self.sites_boundary = np.c_[t_i.T, t[t_i]]


    def _build_sites_touch_mine(self):
        lookup_is_mine = self.utils['lookup_is_mine']
        self.touches_mine = np.where(self.is_mine, 0, np.sum(lookup_is_mine(self.geo.adjs[:,0:4]), axis=1))
        t = np.nonzero(self.touches_mine)[0].T
        self.sites_touch_mine = np.c_[ t, self.touches_mine[t]]


    def _build_strength_difference_at_boundary(self):
        lookup_str = self.utils['lookup_str']
        boundary = self.sites_boundary[:, 0]
        t = lookup_str(np.where( self.sites_boundary[:,1:5] != 0, self.geo.adjs[:,0:4][boundary], IDX_NAN))
        other_str = np.min(np.where(t == STR_NAN, 999, t), axis=1)
        self.sites_str_diff = np.c_[ boundary, other_str - lookup_str(boundary)]


class Strategy:
    def __init__(self, gameData):
        self.logger = None
        self.gd = gameData
        self.geo = gameData.geo
        self.states = self.geo.get_container()
        self.states[:] = STATE_INIT
        self.gotos = self.geo.get_container()
        self.gotos[:] = IDX_NAN
        self.comefroms = self.geo.get_container()
        self.comefroms[:] = IDX_NAN
        self.MAX_STR_STILL = 60

    def add_logger(self, logger):
        self.logger = logger


    def move_strong(self):
        targets =  np.nonzero(np.logical_and(self.gd.strs > self.MAX_STR_STILL, self.gd.is_interior))[0].T
        moves = []

        if len(targets) == 0:
            return moves

        sample_step = max(self.gd.sites_boundary.shape[0] // 8, 1)
        sampled_boundary = self.gd.sites_boundary[::sample_step, 0]

        if self.logger is not None:
            self.logger.write("\nsampled boundary\n%s\n" % str( sampled_boundary ))

        calculate_distance = np.vectorize(lambda loc1, loc2: self.geo.distance(loc1, loc2))
        distances = calculate_distance(targets[:,np.newaxis], sampled_boundary)

        move_to_sites = sampled_boundary[np.argmin(distances, axis=1)]
        unique_move_to_sites = np.unique(move_to_sites)

        for s in unique_move_to_sites:
            target_group = targets[move_to_sites == s]
            group_center = np.average( self.geo.locs[target_group], axis=0)
            group_direction = self.geo.get_direction(group_center, self.geo.locs[s])
            moves += [ Move(self.gd.squares[i], group_direction) for i in target_group ]
            self.states[target_group] = STATE_DONE
            self.gotos[target_group] = self.geo.adjs[target_group, group_direction]
            self.comefroms[self.gotos[target_group]] = target_group

        return moves


    def expand(self):
        sort_type = [('breaths', int), ('production', int), ('site', int)]

        moves = []

        for b_info in self.gd.sites_boundary:
            target = b_info[0]
            move_to_sites = self.geo.adjs[target, 0:4][ b_info[1:5] == INT_TRUE]
            to_sort = np.array([(4 - self.gd.touches_mine[site],
                                 self.gd.prods[site],
                                 site) for site in move_to_sites], dtype=sort_type)
            to_sort.sort(order=['breaths', 'production'])
            move_to_site = None

            if self.comefroms[target] == IDX_NAN:
                defeatables = [ e for e in to_sort if self.gd.strs[e['site']] < self.gd.strs[target]]
                if len(defeatables) > 0:
                    move_to_site = defeatables[-1]['site']
            else:
                move_to_site = to_sort[-1]['site']

            if move_to_site is not None:
                direction = self.geo.get_direction(self.geo.locs[target], self.geo.locs[move_to_site])
                moves.append(Move(self.gd.squares[target], direction))
                self.states[target] = STATE_DONE
                self.gotos[target] = move_to_site
                self.comefroms[move_to_site] = target

        return moves

    def finish_not_moved(self):
        moves = []
        for target in np.where(self.states != STATE_DONE)[0]:
            moves.append(Move(self.gd.squares[target], STILL))
            self.states[target] = STATE_DONE

        return moves


    def execute(self):
        return self.move_strong() + self.expand() + self.finish_not_moved()




def load_game_map(myID, game_map):
    geo = Geo(game_map.height, game_map.width)

    owners = geo.get_container()
    strengths = geo.get_container()
    productions = geo.get_container()

    for square in game_map:
        x, y, owner, strength, production = square
        i = geo.get_i(y, x)
        owners[i] = owner
        strengths[i] = strength
        productions[i] = production


    return GameData(myID, geo, owners, strengths, productions, game_map)

# Map = namedtuple('Map', 'w h')

# test_game_map = GameMap("4 4", "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16", "2 0 1 1 2 0 3 1 1 0 3 1 4 0 71 96 93 157 101 141 63 93 157 93 96 71 93 63 141 101")

# d = load_game_map(1, test_game_map)
# d.analyze()
# s = Strategy(d)
# s.execute()


if __name__ == "__main__":
    myID, game_map = hlt.get_init()
    hlt.send_init("MyPythonBot")

    file_log = None
    #file_log = open("debug" + str(myID) + ".log", "w")

    while True:
        game_map.get_frame()

        gd = load_game_map(myID, game_map)
        gd.add_logger(file_log)

        gd.analyze()
        strategy = Strategy(gd)

        strategy.add_logger(file_log)


        moves = strategy.execute()

        hlt.send_frame(moves)
