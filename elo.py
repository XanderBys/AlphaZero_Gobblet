import numpy as np
import config
from play_game import play_matches
from Environment import Environment
from Model import Model
from Player import Player

def predict_match(ratings):
    return 1/(1+pow(10, (ratings[1]-ratings[0])/400))

env = Environment(4, 4, 4)
INIT_RATING = 1500.
mods = [27, 1]
versions = range(1, len(mods)+1)
n = len(versions) # number of models in tournament
K = 32

models = np.array(list(map(lambda x: Model((2, 64, 4), 12*16, config.HIDDEN_LAYERS, config.REG_CONST, config.LEARNING_RATE), versions)))
# load the weights
for idx, model in zip(mods, models):
    filename = "models/version{}.h5".format(idx)
    print("Loading model with filepath ", filename)
    model.load(filename)

agents = np.array(list(map(lambda x: Player(x, env, config.MCTS_SIMS, config.CPUCT, models[x-1]), versions)))

ratings = np.array([INIT_RATING for i in versions])

print("Initializing list of matches . . .")
matches_seen = set()
matches = np.zeros((n*(n-1)//2, 2), dtype=np.int32)
counter = 0

for i in versions:
    for j in versions:
        pair = [i-1, j-1]
        match = (min(pair), max(pair))
        if i==j or match in matches_seen:
            continue
        else:
            matches_seen.add(match)
            matches[counter] = match
            counter += 1

error = np.full(1, 100)
while abs(np.sum(error) / len(matches)) > 0.01:
    print("Predicting matches . . .")
    predictions = np.fromiter(map(predict_match, ratings[matches]), dtype=np.float32)
    
    print("Playing matches . . .")
    outcomes = np.fromiter(map(lambda x: play_matches(x[0], x[1], 1, 0, single_match=True), agents[matches]), dtype=np.float32)
    error = outcomes - predictions

    # update the ratings
    print("Updating ratings . . .")
    ratings[matches[:, 0]] += K * error
    ratings[matches[:, 1]] -= K * error
    print(predictions, outcomes)
    print(ratings)
    print(np.sum(error)/len(matches))
    
    # switch the matches so that each player can go first
    matches = matches[:, ::-1]