print("Loading external modules . . .")
import os
import time
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
    print("\n\n==================== SELF-PLAY ====================")
    _, memory, _ = play_matches(best_agent, best_agent, config.EPISODES, config.TAU_COUNTER, memory, verbose=False)
    memory.clear_short_term()
    
    if iteration % 2 == 0:
        print("Saving memory . . .")
        pickle.dump(memory, open("memory/{}.p".format(str(iteration).zfill(4)), 'wb+'))
    
    if len(memory.long_term) >= config.MEMORY_CAP:
        ### PART 2: RETRAINING ###
        print("\n\n==================== RETRAINING ====================")
        curr_agent.train(memory.long_term)
            
        ### PART 3: EVALUATION ###
        print("\n\n==================== EVALUATION ====================")
        scores, _, points = play_matches(best_agent, curr_agent, config.EVAL_EPISODES, 0, None)
        
        if scores['curr_agent'] > scores['best_agent'] * config.SCORING_THRESHOLD:
            best_player_version += 1
            best_model.nn.set_weights(curr_model.nn.get_weights())
            print("Saving the new best model . . .")
            best_model.save(env.name, best_player_version)
            
    print("""
          \n\n========================================
          \n{} iterations completed
          \n========================================\n\n
          """.format(iteration))
    
    try:
        print("Program continuing in 10 seconds . . .")
        time.sleep(10)
    except KeyboardInterrupt:
        print("Saving data and exiting training loop . . .")
        
        folder = "data/{}".format(time.strftime("%m_%d_%y_%H_%M_%S"))
        os.mkdir(folder)
        pickle.dump(curr_agent.train_loss, open("{}/train_overall_loss.p".format(folder), 'wb'))
        pickle.dump(curr_agent.train_value_loss, open("{}/train_value_loss.p".format(folder), 'wb'))
        pickle.dump(curr_agent.train_policy_loss, open("{}/train_policy_loss.p".format(folder), 'wb'))
        
        curr_model.save('', "final")
        
        break