from Player import Player, Human
from Environment import Environment
from Model import Model
from play_game import play_matches
import config

env = Environment(4, 4, 4)
human = Human('human', env, -1)
model = Model((2, 64, 4), 12*16, config.HIDDEN_LAYERS, config.REG_CONST, config.LEARNING_RATE)
filepath = input("Type the path to the model: ")
model.load(filepath)
print("Model loaded and compiled")
agent = Player('agent', env, 150, config.CPUCT, model)

play_matches(human, agent, 1, 0, verbose=True)