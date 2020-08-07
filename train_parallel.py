print("Loading external modules . . .")
import os
import multiprocessing
from functools import partial
import time
import random
import pickle
import config
from Environment import Environment
from Player import Player
from Memory import Memory
from Model import Model
from play_game import play_matches

NUM_PROCESSES = 3

# create NN
print("Creating neural network . . .")
curr_model = Model(config.INPUT_SHAPE, 12*16, config.HIDDEN_LAYERS, config.REG_CONST, config.LEARNING_RATE)
LOG_DIR = "data/{}/".format(time.strftime("%m-%d-%y_%H:%M:%S"))
curr_model.log_dir = LOG_DIR

env = Environment(4, 4, 4)
iteration = 0
try:
    iteration = int(open('initialization.txt', 'r').read())
    filename = "models/version{}.h5".format(iteration)
    print("Loading model with filename {} . . .".format(filename))
    curr_model.load(filename)
except (FileNotFoundError, OSError, ValueError):
    pass

print("Initializing agents . . .")
agents = [Player('agent_'+str(i), env, config.MCTS_SIMS, config.CPUCT, curr_model) for i in range(NUM_PROCESSES)]

print("Initialization done. Starting main training loop.")
            
pool = multiprocessing.Pool(processes=NUM_PROCESSES)
memory = Memory()

while True:
    iteration += 1

    ### PART 1: SELF PLAY ###
    print("\n\n==================== SELF-PLAY ====================")
    args = [(agents[i], agents[i], config.EPISODES // NUM_PROCESSES, config.TAU_COUNTER, memory, False) for i in range(NUM_PROCESSES)]
    results = [pool.apply_asynch
    memory.clear_short_term()
        
    ### PART 2: RETRAINING ###
    print("\n\n==================== RETRAINING ====================")
    for i in range(config.TRAINING_LOOPS):
        logging.info("Formatting data . . .")
        batch = random.sample(memory, min(config.BATCH_SIZE, len(memory)))
        states = np.array([sample['state'].binary for sample in batch])
        targets = {'value_head': np.array([sample['value'] for sample in batch]),
                   'policy_head': np.array([sample['AV'] for sample in batch]).reshape(len(batch), 12*16)}
        
        logging.info("Training neural network . . . ")
        hist = self.model.train_batch(states.reshape(len(batch), 64, 4, 2), targets, epochs=config.EPOCHS).history

curr_model.save('', iteration)
