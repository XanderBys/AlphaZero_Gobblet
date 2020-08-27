from Player import Player, Raw_NN
from Environment import Environment
from Model import Model
from play_game import play_matches
import config

env = Environment(4, 4, 4)
model = Model(config.INPUT_SHAPE, 12*16, config.HIDDEN_LAYERS, config.REG_CONST, config.LEARNING_RATE)
filepath = input("Type the path to the model: ")
model.load(filepath)
print("Model loaded and compiled")
agent = Player('agent', env, 100, config.CPUCT/2, model)

#model2 = Model(config.INPUT_SHAPE, 12*16, config.HIDDEN_LAYERS, config.REG_CONST, config.LEARNING_RATE)
#filepath = input("Type the path to the model: ")
#if filepath != 'r':
#    model2.load(filepath)
#print("Model loaded and compiled")
agent_2 = Raw_NN('random', env, model)
if filepath == 'r':
    agent_2.is_random = True
#import pdb;pdb.set_trace()
play_matches(agent_2, agent, 100, 0, verbose=False)
print(agent.nn_diff[:100])
print(agent.nn_diff[-100:])