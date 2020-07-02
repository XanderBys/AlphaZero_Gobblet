from Environment import Environment
import config

def play_matches(player1, player2, EPISODES, tau_counter, memory=None, turn=0):
    players = [player1, player2]
    env =  Environment(4, 4, 4)
    scores = {player1.name:0, "drawn":0, player2.name:0}
    points = {player1.name:[], player2.name:[]}
    
    for i in range(EPISODES):
        env.reset()
        
        player1.mcts = None
        player2.mcts = None
        
        turn_counter = 0
        
        while True:
            turn_counter += 1
            
            if turn_counter < tau_counter:
                # act stochastically
                action, pi, tree_val, nn_val = players[env.turn].move(env, 0)
            else:
                # act deterministically
                action, pi, tree_val, nn_val = players[env.turn].move(env, 1)
            print('move chosen')
            if memory is not None:
                memory.add_sample((env.copy(), pi))
            
            state, result, complete = env.update(action, env.turn)
            
            if complete:
                # the game is over here
                if memory is not None:
                    for sample in memory.short_term:
                        if sample['turn'] == env.turn:
                            move['value'] = result
                        else:
                            move['value'] = -1*result
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
                print("One game complete")
    return scores, memory, points