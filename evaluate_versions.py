from Player import Player
from Environment import Environment
from Model import Model
from play_game import play_matches
import config

env = Environment(4, 4, 4)
model = Model(config.INPUT_SHAPE, 12*16, config.HIDDEN_LAYERS, config.REG_CONST, config.LEARNING_RATE)
filepath = input("Type the path to the model: ")
model.load(filepath)
print("Model loaded and compiled")
agent = Player('agent', env, 100, config.CPUCT, model)
agent_random = Player('random', env, 5, config.CPUCT, model)
agent_random.is_random = True
#import pdb;pdb.set_trace()
play_matches(agent_random, agent, 25, 0, verbose=False)

