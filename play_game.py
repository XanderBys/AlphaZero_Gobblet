from Environment import Environment
import time
def play_matches(player1, player2, EPISODES, tau_counter, memory=None, verbose=False):
    players = [player1, player2]
    scores = {player1.name:0, "drawn":0, player2.name:0}
    points = {player1.name:[], player2.name:[]}
    t = time.time()
    for i in range(EPISODES):
        env =  Environment(4, 4, 4)
        
        player1.mcts = None
        player2.mcts = None
        
        turn_counter = 0
        
        while True:
            turn_counter += 1
            
            if turn_counter < tau_counter:
                # act stochastically
                action, pi, tree_val, nn_val, next_state, result, complete = players[env.turn].move(env, 1)
            else:
                # act deterministically
                action, pi, tree_val, nn_val, next_state, result, complete = players[env.turn].move(env, 0)
    
            if memory is not None:
                memory.add_sample((env.copy(), pi))
            
            env = next_state
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
                    scores[players[env.turn].name] += 1
                    points[players[env.turn].name].append(1)
                    points[players[-1*env.turn].name].append(-1)
                    
                elif result == -1:
                    scores[players[-1*env.turn].name] += 1
                    points[players[-1*env.turn].name].append(1)
                    points[players[env.turn].name].append(-1)
                else:
                    scores['drawn'] += 1
                    points[players[env.turn].name].append(0)
                    points[players[-1*env.turn].name].append(0)
                
                # switch who starts the game
                players = players[::-1]
                print("Total time: {}\n Time on simulations: {} ({}%)".format(time.time()-t, player1.time_sims, (player1.time_sims/(time.time()-t))*100))
                break
        print("{} out of {} games complete".format(i+1, EPISODES))
    
    return scores, memory, points