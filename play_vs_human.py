from Player import Human
from Environment import Environment
from play_game import play_matches

env = Environment(4, 4, 4)
human = Human('human', env, -1)

play_matches(human, human, 3, 10, verbose=True)