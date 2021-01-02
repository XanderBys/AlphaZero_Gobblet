from Player import Player, Human, Raw_NN
from Environment import Environment
from Model import Model
from play_game import play_matches
import config
import profile

env = Environment(4, 4, 4)
human = Human('human', env, -1)

print("Creating neural network . . .")
model = Model(config.INPUT_SHAPE, 12*16, config.HIDDEN_LAYERS, config.REG_CONST, config.LEARNING_RATE)
iteration = 0
filepath = "models/version2.h5"

try:
    print("Loading model with filename {} . . .".format(filename))
    model.load(filename)
    print("Model loaded and compiled")
except:
    print("Starting a new model from scratch . . .")

agent = Player('agent', env, 100, 2.5, model)
agent.mcts = None

profile.run("agent.move(env, 1)")