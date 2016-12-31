import numpy as np
import math
from collections import namedtuple
import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square, GameMap

STR_MAX = 250
STR_NAN = -1
PROD_NAN = -1
IDX_NAN = -1
DIRECTION_NAN = -1
INT_TRUE = 1
INT_FALSE = 0

ROUNDS_INF = 999
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
        return np.zeros(self.n, dtype=t)

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

    def reverse_direction(self, direction):
        return [2, 3, 0, 1, 4, 7, 8, 5, 6][direction]

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
        self._build_nearby_enemies()


        #self.logger.write("\nowners\n%s\n" % str( self.owners.reshape(self.geo.h, self.geo.w)))


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
        self.str_diffs = self.geo.get_container()
        self.str_diffs[boundary] = other_str - lookup_str(boundary)

    def _build_nearby_enemies(self):
        def _any_enemy_in_sites(sites):
            return np.any( self.is_oppo[sites] )
        targets = self.sites_touch_mine[:,0]
        self.sites_nearby_enemy = targets[ np.apply_along_axis(_any_enemy_in_sites, 1, self.geo.adjs[targets])]
        self.nearby_enemy = self.geo.get_container(bool)
        self.nearby_enemy[ self.sites_nearby_enemy ] = True


class Strategy:
    def __init__(self, gameData):
        self.logger = None
        self.MAX_STR_STILL = 60
        self.INIT_MOMENTUM = 1
        self.gd = gameData
        self.geo = gameData.geo
        self.states = self.geo.get_container()
        self.states[:] = STATE_INIT
        self.gotos = self.geo.get_container()
        self.gotos[:] = IDX_NAN
        self.directions = self.geo.get_container()
        self.directions[:] = DIRECTION_NAN
        self.comefroms = np.zeros( (self.geo.n, 3), dtype=int) # strongest_site, str, number
        self.comefroms[:,0] = IDX_NAN
        self.momentums = np.zeros( (self.geo.n, 2), dtype=int)
        self.momentums[:,0] = DIRECTION_NAN

    def add_logger(self, logger):
        self.logger = logger

    def _create_move(self, strategy, from_s, to_s, direction, gain_momentum=False):
        self.states[from_s] = STATE_DONE
        self.gotos[from_s] = to_s
        self.directions[from_s] = direction

        strs_from = self.gd.strs[from_s]
        str_compared = self.comefroms[to_s, 1] < strs_from
        self.comefroms[to_s, 0] = np.where( str_compared, from_s  , self.comefroms[to_s, 0])
        self.comefroms[to_s, 1] = np.where( str_compared, strs_from, self.comefroms[to_s, 1])
        self.comefroms[to_s, 2] = self.comefroms[to_s, 2] + 1

        self.momentums[to_s, 1] = np.where(self.momentums[to_s, 1] > 0, self.momentums[to_s, 1] - 1, 0)
        if gain_momentum:
            self.momentums[to_s] = [direction, self.INIT_MOMENTUM]

    def _extract_move(self):
        moves = []
        for s in np.nonzero(self.gd.is_mine)[0]:
            direction = STILL
            if self.directions[s] != DIRECTION_NAN:
                direction = self.directions[s]
            square = self.gd.squares[s]
            moves.append(Move(square, direction))
        return moves


    def keep_momentum(self, targets):
        for target in targets:
            if self.momentums[target, 0] != DIRECTION_NAN and self.momentums[target, 1] > 0:
                move_to = self.geo.adjs[target, self.momentums[target, 0]]
                self._create_move("keep_momentum",  target, move_to, self.momentums[target, 0])

    def import_momentums(self, momentums):
        self.momentums = momentums

    def export_momentums(self):
        return self.momentums.copy()

    def move_strong(self):
        targets =  np.nonzero(np.logical_and(self.gd.strs > self.MAX_STR_STILL, self.gd.is_interior))[0].T

        if len(targets) == 0:
            return

        self.keep_momentum(targets)

        targets = np.array([ t for t in targets if self.states[t] == STATE_INIT ])

        if len(targets) == 0:
            return

        calculate_distance = np.vectorize(lambda loc1, loc2: self.geo.distance(loc1, loc2))
        surroundings = np.array([ s for s in self.gd.sites_touch_mine[:,0] if not self.gd.nearby_enemy[s]])

        if len(surroundings) == 0:
            return

        enemies = self.gd.sites_nearby_enemy
        enemy_alert_radius = 15
        if len(enemies) > 0:
            ds = calculate_distance(enemies[:, np.newaxis], surroundings)
            ds[ ds < enemy_alert_radius ] = -1
            surroundings = surroundings[ np.all( ds != -1, axis=0)]
            surroundings = np.concatenate((enemies, surroundings))

        if len(surroundings) == 0:
            return

        distances = calculate_distance(targets[:,np.newaxis], surroundings)

        move_to_sites = surroundings[np.argmin(distances, axis=1)]
        unique_move_to_sites = np.unique(move_to_sites)

        for s in unique_move_to_sites:
            target_group = targets[move_to_sites == s]
            group_center = np.average( self.geo.locs[target_group], axis=0)
            group_direction = self.geo.get_direction(group_center, self.geo.locs[s])

            for target in target_group:
                target_move_to = self.geo.adjs[target, group_direction]
                self._create_move("move_strong", target, target_move_to, group_direction, gain_momentum=True)




    def expand(self):
        sort_type = [('breaths', int), ('production', int), ('site', int)]

        for b_info in self.gd.sites_boundary:
            target = b_info[0]
            move_to_sites = self.geo.adjs[target, 0:4][ b_info[1:5] == INT_TRUE]
            to_sort = np.array([(4 - self.gd.touches_mine[site],
                                 self.gd.prods[site],
                                 site) for site in move_to_sites], dtype=sort_type)
            to_sort.sort(order=['breaths', 'production'])
            move_to_site = None

            defeatables = [ e for e in to_sort if self.gd.strs[e['site']] < self.gd.strs[target]]
            if len(defeatables) > 0:
                move_to_site = defeatables[-1]['site']


            if move_to_site is not None:
                direction = self.geo.get_direction(self.geo.locs[target], self.geo.locs[move_to_site])
                self._create_move("expand", target, move_to_site, direction)

    def converging_attack(self):
        converging_points = self.gd.sites_touch_mine[:,0][self.gd.sites_touch_mine[:,1] > 1]

        for point in converging_points:
            mine_not_moved = [ (direction, s) for (direction, s) in enumerate( self.geo.adjs[point, 0:4])
                               if self.states[s] == STATE_INIT and self.gd.is_mine[s]]

            if sum([ self.gd.strs[s] for (direction, s) in mine_not_moved]) > self.gd.strs[point]:
                for direction, s in mine_not_moved:
                    d = self.geo.reverse_direction(direction)
                    self._create_move("converging_attack",  s, point, d)

    def focus_force(self):
        bds = self.gd.sites_boundary[:,0]
        prods = self.gd.prods[bds]

        with np.errstate(divide='ignore'):
            rounds_to_conquer = self.gd.str_diffs[bds] / prods

        rounds_to_conquer[np.isnan(rounds_to_conquer)] = ROUNDS_INF
        min_rounds_to_conquer = np.min(np.where(prods > 0, self.gd.str_diffs[bds] / prods, ROUNDS_INF))

        if (min_rounds_to_conquer <= 2):
            return

        territory_of_interest = np.array( [ s for s in self.gd.sites_touch_mine[:,0] if self.gd.prods[s] > 0 ] )
        sort_type = [('payback_period', int), ('strs', int), ('site', int)]

        to_sort = np.array([ ( self.gd.strs[s] // self.gd.prods[s] , self.gd.strs[s] , s) for s in territory_of_interest ], dtype=sort_type)
        to_sort.sort(order=['payback_period', 'strs'])

        target = to_sort[0]['site']

        mine = [ s for s in self.geo.adjs[target, 0:4] if self.gd.is_mine[s]]
        focus_at = mine[np.argmin([ self.gd.str_diffs[s] for s in mine ])]

        move_tos = [focus_at]
        levels = min_rounds_to_conquer

        def _gather_at_focus(focus):
            min_str_to_move = 10
            targets = []
            if self.states[focus] == STATE_INIT:
                self.states[focus] = STATE_DONE

            for direction, s in enumerate(self.geo.adjs[focus, 0:4]):
                if self.gd.is_mine[s] and self.states[s] == STATE_INIT:
                    if self.gd.strs[s] >= min_str_to_move:
                        d = self.geo.reverse_direction(direction)
                        self._create_move("focus_force", s, focus, d)

                        targets.append(s)
            return targets

        move_tos = [focus_at]
        levels = min_rounds_to_conquer

        while len(move_tos) > 0 and levels > 0:
            for s in move_tos:
                move_tos += _gather_at_focus(s)
            levels -= 1


    def run_from_overstr(self):
        targets = np.where(self.gotos != IDX_NAN)[0]

        while len(targets) > 0:
            _targets = []
            for s in targets:
                goto = self.gotos[s]
                if self.gd.is_mine[goto] and (self.states[goto] == STATE_INIT) and (self.gd.strs[s] + self.gd.strs[goto] >= STR_MAX):
                    _targets.append(goto)
                    direction = self.geo.get_direction(self.geo.locs[s], self.geo.locs[goto])
                    move_to = self.geo.adjs[goto, direction]
                    moves.append(self._create_move("run_from_overstr", goto, move_to, direction))
            targets = _targets

    def avoid_overstr(self):
        sites = np.where(self.comefroms[:,2] > 1)[0]

        for s in sites:
            if not self.gd.is_mine[s]:
                continue

            origins = np.where(self.gotos == s)[0]
            if self.gotos[s] != IDX_NAN:
                site_str = 0
            else:
                site_str = self.gd.strs[s]

            if site_str + np.sum( self.gd.strs[origins]) > STR_MAX:
                direction_of_the_strongest = self.directions[self.comefroms[s, 0]]
                for o in origins:
                    move_to = self.geo.adjs[o, direction_of_the_strongest]
                    self._create_move("avoid_overstr", o, move_to, direction_of_the_strongest)

                    # has bug. didn't update comefroms

    def execute(self):
        self.focus_force()
        self.move_strong()
        self.expand()
        self.converging_attack()
        self.run_from_overstr()
        self.avoid_overstr()
        return self._extract_move()


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

#Map = namedtuple('Map', 'w h')

#test_game_map = GameMap("4 4", "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16", "1 0 15 1 71 96 93 157 151 141 63 93 157 93 96 71 93 63 141 101")

#test_game_map = GameMap("4 4", "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16", "1 0 2 1 7 0 1 2 5 0 71 96 93 157 151 141 63 93 157 93 96 71 93 63 141 101")

#test_game_map = GameMap("4 4", "1 2 3 4 5 20 7 8 9 10 11 12 13 14 15 16", "6 0 1 1 2 0 2 1 5 0 100 100 100 100 100 100 10 100 100 11 5 100 100 100 50 100")

#d = load_game_map(1, test_game_map)
#d.analyze()
#s = Strategy(d)
#s.execute()


if __name__ == "__main__":
    myID, game_map = hlt.get_init()
    hlt.send_init("MyPythonBot")

    file_log = None
    #file_log = open("debug" + str(myID) + ".log", "w")

    momentums = None

    while True:
        game_map.get_frame()

        gd = load_game_map(myID, game_map)
        gd.add_logger(file_log)

        gd.analyze()
        strategy = Strategy(gd)
        strategy.add_logger(file_log)

        if momentums is not None:
            strategy.import_momentums(momentums)

        moves = strategy.execute()

        momentums = strategy.export_momentums()

        hlt.send_frame(moves)
