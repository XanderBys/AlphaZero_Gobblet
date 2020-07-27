from Environment import Environment
import time
import logging
#logging.basicConfig(filename="logs/play_game.log", level=logging.INFO)
def play_matches(player1, player2, EPISODES, tau_counter, memory=None, verbose=False, single_match=False):
    players = [player1, player2]
    scores = {player1.name:0, "drawn":0, player2.name:0}
    points = {player1.name:[], player2.name:[]}
    t = time.time()
    for i in range(EPISODES):
        env =  Environment(4, 4, 4)
        
        player1.mcts = None
        player2.mcts = None
        
        turn_counter = 0
        
        states_seen = set()
        duplicate_states = set()

        while True:
            turn_counter += 1
            
            if turn_counter < tau_counter:
                # act stochastically
                action, pi, tree_val, nn_val, next_state, result, complete = players[env.pieces_idx].move(env, 1)
            else:
                # act deterministically
                action, pi, tree_val, nn_val, next_state, result, complete = players[env.pieces_idx].move(env, 0)
    
            if memory is not None:
                memory.add_sample((env.copy(), pi))
            #logging.info("Move {} chosen".format(turn_counter))

            env = next_state
            if env.id in states_seen:
              duplicate_states.add(env.id)
            elif env.id in duplicate_states:
              complete = True
              result = 0

            if verbose:
                print(env)
                
            if complete:
                # the game is over here
                if memory is not None:
                    for sample in memory.short_term:
                        if sample['turn'] == env.turn:
                            sample['value'] = result
                        else:
                            sample['value'] = -1*result
                    memory.update_long_term()
                
                if result == 1:
                    scores[players[env.pieces_idx].name] += 1
                    points[players[env.pieces_idx].name].append(1)
                    points[players[-1*env.pieces_idx].name].append(-1)
                    
                elif result == -1:
                    scores[players[-1*env.pieces_idx].name] += 1
                    points[players[-1*env.pieces_idx].name].append(1)
                    points[players[env.pieces_idx].name].append(-1)
                else:
                    scores['drawn'] += 1
                    points[players[env.pieces_idx].name].append(0)
                    points[players[-1*env.pieces_idx].name].append(0)
                
                # switch who starts the game
                players = players[::-1]
                #logging.info("Total time: {}\n Time on simulations: {} ({}%)".format(time.time()-t, player1.time_sims, (player1.time_sims/(time.time()-t))*100))
                break
        if not single_match:
            print("{} out of {} games complete".format(i+1, EPISODES))
            logging.info("{} out of {} games complete".format(i+1, EPISODES))
    
    if not single_match:
        if memory is None:
            print("Final scores: {}".format(scores))
        return scores, memory, points
    else:
        return scores[player1.name]
