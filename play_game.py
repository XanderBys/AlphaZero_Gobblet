from Environment import Environment
import time, sys
import logging
import config
#logging.basicConfig(filename="logs/play_game.log", level=logging.INFO)
def play_matches(player1, player2, EPISODES, tau_counter, memory=None, verbose=False, single_match=False):
    players = [player1, player2]

    scores = {player1.name:0, "drawn":0, player2.name:0}
    points = {player1.name:[], player2.name:[]}
    
    t = time.time()
    for i in range(EPISODES):
        env =  Environment(4, 4, 4)
        turn = int(env.turn)
        
        player1.mcts = None
        player2.mcts = None
        
        turn_counter = 0
        t=time.time()
        while True:
            turn_counter += 1
            
            if players[env.pieces_idx].is_random:
                action, pi, tree_val, nn_val, next_state, result, complete = players[env.pieces_idx].take_random_action(env)
            elif turn_counter < tau_counter:
                # act stochastically
                action, pi, tree_val, nn_val, next_state, result, complete = players[env.pieces_idx].move(env, 1)
            else:
                # act deterministically
                action, pi, tree_val, nn_val, next_state, result, complete = players[env.pieces_idx].move(env, 0)
                
            env = next_state
            turn *= -1
            env.turn = int(turn)
            
            if memory is not None:
                memory.add_sample((env.copy(), pi))
                
            if turn_counter >= config.MAX_GAME_LENGTH:
              complete = True
              result = 0

            if verbose:
                print(env, file=sys.stdout)
                
            if complete:
                # the game is over here
                if memory is not None:
                    if result == 0:
                        fout = open("log.txt", "a+")
                    for sample in memory.short_term:
                        if sample['turn'] == env.turn:
                            sample['value'] = result
                        else:
                            sample['value'] = -1*result
                        if result == 0:
                            fout.write(str(sample['state'])+'\n')
#                        print(sample['state'])
#                        print("Value: " + str(sample['value'])+"\n\n")
                    memory.update_long_term()
                    if result == 0:
                        fout.close()
                    
                if result == 1:
                    scores[players[env.pieces_idx].name] += 1
                    points[players[env.pieces_idx].name].append(1)
                    points[players[env.pieces_idx-1].name].append(-1)
                    
                elif result == -1:
                    scores[players[int(not env.pieces_idx)].name] += 1
                    points[players[int(not env.pieces_idx)].name].append(1)
                    points[players[env.pieces_idx].name].append(-1)
                else:
                    scores['drawn'] += 1
                    points[players[env.pieces_idx].name].append(0)
                    points[players[int(not env.pieces_idx)].name].append(0)
                
                # switch who starts the game
                players = players[::-1]
#                print("Total time: {}\n Time on simulations: {} ({}%)".format(time.time()-t, player2.time_sims, (player2.time_sims/(player2.total_time))*100))
#                player2.time_sims=0
#                plyaer2.total_time = 0
                break
        if not single_match:
            t_elapsed = time.time()-t
            print("{} out of {} games complete in {} moves and {:.1f}s ({:.3f}s/move)".format(i+1, EPISODES, turn_counter, t_elapsed, (t_elapsed/turn_counter)))
            logging.info("{} out of {} games complete in {} moves".format(i+1, EPISODES, turn_counter))
    
    if not single_match:
        if memory is None:
            print("Final scores: {}".format(scores))
        
        return scores, memory, points
    else:
        return scores[player1.name]
