import sys
import pickle
import Environment
from Player import Player
from Player import Human
import State
from Model import Model

env = Environment.Environment(4, 4, 4)
prefix = input("Input the prefix for the model to be loaded: ")
model_name = input("Input the name of the model you want: ") if len(sys.argv) != 2 else sys.argv[1]
print("Loading model with filename {}{} . . .".format(prefix, model_name))
agent = Player(model_name, env, 1, 0)
agent.load_policy(model_name, prefix)

human = Human('human', env, -1)
players = [human, agent]
while True:
    while True:
        for player in players:
            if env.turn != player.symbol:
                env.turn *= -1
            game_over = False
            
            # choose action
            action = player.choose_action(env.state)
            
            # take action
            next_state, result = env.update(action, player, check_legal=True)
            
            env.display()
            if result != None:
                # the game is over here
                game_over = True
                if result == human.symbol:
                    print("You win!")
                elif result == agent.symbol:
                    print("The Q Agent wins!")
                else:
                    print("Draw")
                break
            
        if game_over:
            break
        
    cont = input("Continue(y/n)? ")
    if cont.lower() != 'y':
        break
    
    players = players[::-1]
    env.reset()
