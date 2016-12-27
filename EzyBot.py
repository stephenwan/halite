import numpy as np
import math
from collections import namedtuple
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square, GameMap

STR_MAX = 250
STR_NAN = 999

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
        self._init_states()

    def _analyze(self):
        self.is_mine = self.owners == self.me
        self.is_neutual = self.owners == self.owner_neutral
        self.is_oppo = np.logical_and(self.owners != self.me, self.owners != self.owner_neutral)
        lookup_is_mine = np.vectorize(lambda i: self.is_mine[i])
        lookup_is_not_mine = np.vectorize(lambda i: not self.is_mine[i])
        self.is_bd = np.logical_and(np.any(lookup_is_not_mine(self.geo.adjs[:,0:4]), axis=1), self.is_mine)
        self.contacts = np.where(self.is_mine, 0, np.sum(lookup_is_mine(self.geo.adjs[:,0:4]), axis=1))

        lookup_other_str = np.vectorize(lambda i: self.strs[i] if not self.is_mine[i] else STR_NAN)
        lookup_mine_str =  np.vectorize(lambda i: self.strs[i] if self.is_mine[i] else 0)
        self.dstr = np.where(self.is_bd,
                             np.min(lookup_other_str(self.geo.adjs[:,0:4]), axis=1) - self.strs,
                             STR_NAN)

        self.dstr_ms = np.where(self.contacts > 1,
                                self.strs - np.sum(lookup_mine_str(self.geo.adjs[:,0:4]), axis=1),
                                STR_NAN)

    def _init_states(self):
        self.states = self.geo.get_container()
        self.states[:] = STATE_INIT
        self.gotos = self.geo.get_container()





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

test_game_map = GameMap("4 4", "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16", "2 0 1 1 2 0 3 1 1 0 2 1 5 0 71 96 93 157 101 141 63 93 157 93 96 71 93 63 141 101")

d = load_game_map(1, test_game_map)
