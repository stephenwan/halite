import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random
from gm import GameMaster


myID, game_map = hlt.get_init()
hlt.send_init("MyPythonBot")

file_log = open("debug.log", "w")

rounds = 0

while True:
    game_map.get_frame()
    gm = GameMaster(myID, game_map, file_log)

    moves = gm.play()
    rounds += 1

    hlt.send_frame(moves)

file_log.close()
