from Player import Player, Human, Raw_NN
from Environment import Environment
from Model import Model
from play_game import play_matches
import config
import profile

env = Environment(4, 4, 4)
human = Human('human', env, -1)
model = Model(config.INPUT_SHAPE, 12*16, config.HIDDEN_LAYERS, config.REG_CONST, config.LEARNING_RATE)
filepath = "models/version2.h5"
model.load(filepath)
print("Model loaded and compiled")
agent = Player('agent', env, 100, 2.5, model)
agent.mcts = None

profile.run("agent.move(env, 1)")