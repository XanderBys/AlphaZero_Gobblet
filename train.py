print("Loading external modules . . .")
import random
import pickle
import initialization
import config
from Environment import Environment
from Player import Player
from Memory import Memory
from Model import Model
from play_game import play_matches

env = Environment(4, 4, 4)
memory = Memory()

# create 2 NNs
print("Creating neural networks . . .")
curr_model = Model((2, 64, 4), 12*16, config.HIDDEN_LAYERS, config.REG_CONST, config.LEARNING_RATE)
best_model = Model((2, 64, 4), 12*16, config.HIDDEN_LAYERS, config.REG_CONST, config.LEARNING_RATE)

best_player_version = 0
best_model.nn.set_weights(curr_model.nn.get_weights()) # ensure the weights are the same

print("Initializing agents . . .")
curr_agent = Player('curr_agent', env, config.MCTS_SIMS, config.CPUCT, curr_model)
best_agent = Player('best_agent', env, config.MCTS_SIMS, config.CPUCT, best_model)

print("Initialization done. Starting main training loop.")
iteration = 0
while True:
    iteration += 1
    
    ### PART 1: SELF PLAY ###
    _, memory, _ = play_matches(best_agent, best_agent, config.EPISODES, config.TAU_COUNTER, memory, verbose=True)
    memory.clear_short_term()
    
    if len(memory.long_term) >= config.MEMORY_CAP:
        ### PART 2: RETRAINING ###
        curr_agent.train(memory.long_term)
        
        if iteration % 20 == 0:
            print("Saving memory . . .")
            pickle.dump(memory, open('./memory/' + str(iteration).zfill(4) + '.p'))
            
        ### PART 3: EVALUATION ###
        scores, _, points = play_matches(best_agent, curr_agent, config.EVAL_EPISODES, 0, None)
        
        if scores['curr_agent'] > scores['best_agent'] * config.SCORING_THRESHOLD:
            best_player_version += 1
            best_model.nn.set_weights(curr_model.nn.get_weights())
            print("Saving the new best model . . .")
            best_model.save(env.name, best_player_version)
    print("{} iterations completed".format(iteration))
