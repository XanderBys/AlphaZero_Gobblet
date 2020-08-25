from Player import Human
from Environment import Environment
from play_game import play_matches
from Memory import Memory
import config

env = Environment(4, 4, 4)
memory = Memory()
human = Human('human', env, -1)

_, mem, _ = play_matches(human, human, 1, 0, memory=memory, verbose=True)
mem.clear_short_term()
for i in mem.long_term:
    print(str(i['state']), i['value'])