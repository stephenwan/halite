import numpy as np
import math
from collections import namedtuple
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square, GameMap

STR_MAX = 250
STR_NAN = 999
IDX_NAN = -1

STATE_INIT, STATE_DONE = (1, 2)

class Geo:
    def __init__(self, h, w):
        self.h, self.w = (h, w)
        self.n = self.h * self.w
        self.init_geo()

    def add_logger(logger):
        self.logger = logger

    def init_geo(self):
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
    def __init__(self, myID, geo, owners, strengths, productions):
        self.me = myID
        self.owner_neutral = 0
        self.geo = geo
        self.owners = owners
        self.strs = strengths
        self.prods = productions
        self._analyze()

    def _analyze(self):
        self.is_mine = self.owners == self.me
        self.is_neutual = self.owners == self.owner_neutral
        self.is_oppo = np.logical_and(self.owners != self.me, self.owners != self.owner_neutral)

        self._build_boundary()
        self._build_sites_touch_mine()
        self._build_strength_difference_at_boundary()


    def _build_boundary(self):
        lookup_is_not_mine = np.vectorize(lambda i: not self.is_mine[i])
        t = lookup_is_not_mine(self.geo.adjs[:,0:4])
        self.is_bd = np.logical_and(np.any(t, axis=1), self.is_mine)
        self.is_interior = np.logical_and(np.logical_not(self.is_bd), self.is_mine)
        t_i = np.nonzero(self.is_bd)[0]
        self.sites_boundary = np.c_[t_i.T, t[t_i]]


    def _build_sites_touch_mine(self):
        lookup_is_mine = np.vectorize(lambda i: self.is_mine[i])
        touches_mine = np.where(self.is_mine, 0, np.sum(lookup_is_mine(self.geo.adjs[:,0:4]), axis=1))
        t = np.nonzero(touches_mine)[0].T
        self.sites_touch_mine = np.c_[ t, touches_mine[t]]

    def _build_strength_difference_at_boundary(self):
        lookup_str = np.vectorize(lambda i: self.strs[i] if i != IDX_NAN else STR_NAN)
        boundary = self.sites_boundary[:, 0]
        t = lookup_str(np.where( self.sites_boundary[:,1:5] != 0, self.geo.adjs[:,0:4][boundary], IDX_NAN))
        other_str = np.min(t, axis=1)
        self.sites_str_diff = np.c_[ boundary, other_str - lookup_str(boundary)]


class Strategy:
    def __init__(self, gameData):
        self.gd = gameData
        self.geo = gameData.geo
        self.states = self.geo.get_container()
        self.states[:] = STATE_INIT
        self.gotos = self.geo.get_container()

        self.MAX_STR_STILL = 60

        self.move_strong()


    def move_strong(self):
        targets =  np.nonzero(np.logical_and(self.gd.strs > self.MAX_STR_STILL, self.gd.is_interior))[0].T

        sample_step = max(self.gd.sites_boundary.shape[0] // 8, 1)
        sampled_boundary = self.gd.sites_boundary[::sample_step, 0]

        calculate_distance = np.vectorize(lambda loc1, loc2: self.geo.distance(loc1, loc2))

        distances = calculate_distance(targets[:,np.newaxis], sampled_boundary)

        move_to_sites = sampled_boundary[np.argsort(distances, axis=1)]

        self.target = move_to_sites
        return









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


    return GameData(myID, geo, owners, strengths, productions)

Map = namedtuple('Map', 'w h')

test_game_map = GameMap("4 4", "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16", "2 0 1 1 2 0 3 1 1 0 3 1 2 0 1 1 1 0 71 96 93 157 101 141 63 93 157 93 96 71 93 63 141 101")

d = load_game_map(1, test_game_map)
s = Strategy(d)
