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
curr_model = Model((2, 64, 4), 12*16, config.HIDDEN_LAYERS, config.REG_CONST, config.LEARNING_RATE)
iteration = 0
try:
    iteration = int(open('initialization.txt', 'r').read())
    filename = "models/version{}.h5".format(iteration)
    print("Loading model with filename {} . . .".format(filename))
    curr_model.load(filename)
except (FileNotFoundError, OSError):
    pass

print("Initializing agent . . .")
curr_agent = Player('curr_agent', env, config.MCTS_SIMS, config.CPUCT, curr_model)

print("Initialization done. Starting main training loop.")

try:
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
        
except KeyboardInterrupt:
    pass

<<<<<<< HEAD
=======
except Exception as err:
    print("The following error occurred at {}:\n{}".format(time.strftime("%H:%M:%S"), str(err)))

>>>>>>> c04eca8fe848170ed7fd1c6d821366c36cc40f26
finally:
    print("Saving data and exiting training loop . . .")
    folder = "data/{}".format(time.strftime("%m-%d-%y_%H:%M:%S"))
    os.mkdir(folder)
    pickle.dump(curr_agent.train_loss, open("{}/train_overall_loss.p".format(folder), 'wb'))
    pickle.dump(curr_agent.train_value_loss, open("{}/train_value_loss.p".format(folder), 'wb'))
    pickle.dump(curr_agent.train_policy_loss, open("{}/train_policy_loss.p".format(folder), 'wb'))
    
    fout = open('initialization.txt', 'w+')
    fout.write(str(iteration))
    fout.close()