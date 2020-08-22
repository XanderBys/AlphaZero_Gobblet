print("Loading external modules . . .")
import os
import time
import random
import pickle
import config
from Environment import Environment
from Player import Player
from Memory import Memory
from Model import Model
from play_game import play_matches

env = Environment(4, 4, 4)
memory = Memory()

# create NN
print("Creating neural network . . .")
curr_model = Model(config.INPUT_SHAPE, 12*16, config.HIDDEN_LAYERS, config.REG_CONST, config.LEARNING_RATE)
iteration = 0
try:
    iteration = int(open('initialization.txt', 'r').read())
    filename = "models/version{}.h5".format(iteration)
    print("Loading model with filename {} . . .".format(filename))
    curr_model.load(filename)
except:
    pass

print("Initializing agent . . .")
curr_agent = Player('curr_agent', env, config.MCTS_SIMS, config.CPUCT, curr_model)

print("Initialization done. Starting main training loop.")
#import pdb;pdb.set_trace()
try:
    LOG_DIR = "data/{}/".format(time.strftime("%m-%d-%y_%H:%M:%S"))
    curr_model.log_dir = LOG_DIR
    while True:
        iteration += 1

        ### PART 1: SELF PLAY ###
        print("\n\n==================== SELF-PLAY ====================")

        _, memory, _ = play_matches(curr_agent, curr_agent, config.EPISODES, config.TAU_COUNTER, memory, verbose=False)
        memory.clear_short_term()
        
        ### PART 2: RETRAINING ###
        print("\n\n==================== RETRAINING ====================")
        curr_agent.train(memory.long_term)
        
        curr_model.save('', iteration)
        with open("initialization.txt", "w+") as fout:
            fout.write(str(iteration))
        
except KeyboardInterrupt:
    pass