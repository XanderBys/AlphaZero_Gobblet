import random, time
import pickle
import multiprocessing, functools
import numpy as np
from State import State
import MCTS
import config
import logging

class Player:
    def __init__(self, name, env, num_sims, cpuct, model):
        self.name = name
        self.env = env
        self.samples = []
        self.num_sims = num_sims
        self.cpuct = cpuct
        self.model = model
        self.is_random = False
        
        self.positions_cache = {}
        #self.NUM_WORKERS = 2
        
        self.nn_diff = []
    
    def run_simulation(self, t):
        leaf, value, done, edges = self.mcts.go_to_leaf()
        logging.info("Navigated to leaf")
        
        value = self.evaluate_state(leaf, value, done)
        logging.info("Evaluated state")

        self.mcts.update_nodes(leaf, value, edges)
        logging.info("Updated nodes")
        
    def evaluate_state(self, state, values, complete):
        if not complete:
            logging.info("Predicting state . . .")
            probs, values, legal_moves = self.predict_state(state)
            logging.info("Iterating through legal moves . . .")
            for action in legal_moves.T:
                new_state, val, complete = state.env.update(action)
                state.env.undo_move()
                if new_state.id not in self.mcts.tree.keys():
                    new_node = MCTS.Node(new_state.copy())
                    self.mcts.add_node(new_node)
                else:
                    new_node = self.mcts.tree[new_state.id]
                new_edge = MCTS.Edge(state, new_node, probs[action[0], action[1]], action)
                state.edges.append((action, new_edge))     
        return values
    
    def move(self, state, tau):
        #import pdb;pdb.set_trace()
        if self.mcts is None or state.id not in self.mcts.tree:
            self.build_MCTS(state)
        else:
            self.change_MCTS_root(state)
        
        for sim in range(self.num_sims):
            self.run_simulation(state.turn)
            
        logging.info("Simulations complete")
        pi, vals = self.get_action_vals(1)
        action, value = self.choose_action(pi, vals, tau)
        tree_state = self.mcts.root.env
        
        try:
            next_state, result, complete = tree_state.update(action)
            tree_state.undo_move()
        except ValueError as err:
            print(state)
            print(tree_state)
            raise err
        
        nn_probs, nn_values, lm = self.predict_state(MCTS.Node(state))
        self.nn_diff.append(np.sum(np.abs(nn_probs-pi)))
        
        return (action, pi, value, nn_values, next_state, result, complete)
    
    def choose_action(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == np.amax(pi))
            action = tuple(random.choice(actions))
        else:
            action_idx = np.random.multinomial(1, pi.reshape(-1)/sum(pi.reshape(-1))).reshape(pi.shape)
            loc = np.where(action_idx == 1)
            action = (loc[0][0], loc[1][0])
        
        value = values[action]
        
        return (action, value)
    
    def get_action_vals(self, tau):
        pi = np.zeros((12, 16), dtype=np.float32)
        vals = np.zeros((12, 16), dtype=np.float32)
        for action, edge in self.mcts.root.edges:
            pi[tuple(action)] = pow(edge.data['N'], 1/tau)
            vals[tuple(action)] = edge.data['Q']
        
        pi = pi.astype(float)
        pi /= np.sum(pi)
        return pi, vals
    
    def train(self, memory, single_loop=False):
        # train the model based on the reward
        num_loops = config.TRAINING_LOOPS if not single_loop else 1
        for i in range(config.TRAINING_LOOPS):
            logging.info("Formatting data . . .")
            batch = random.sample(memory, min(config.BATCH_SIZE, len(memory)))
            states = np.array([sample['state'].binary for sample in batch])
            targets = {'value_head': np.array([sample['value'] for sample in batch]),
                       'policy_head': np.array([sample['AV'] for sample in batch]).reshape(len(batch), 12*16)}
            
            logging.info("Training neural network . . . ")
            hist = self.model.train_batch(states.reshape(len(batch), 4, 16, 2), targets, epochs=config.EPOCHS).history
        
        # clear the cache now that the model has been updated
        self.posistions_cache = {}
        
    def predict_state(self, state):
        inp = state.env.binary
        state_id = state.env.id
        
        moves = state.env.get_legal_moves_idxs()
        legal_moves = np.array(moves).T
        
        if state_id in self.positions_cache:
            probs, values = self.positions_cache[state_id]
            return probs, values, legal_moves
        
        else:
            values, logits = self.model.predict_one(inp)
            
            values = values[0]
            logits = logits[0].reshape(12, 16)
            
            # make sure illegal moves aren't chosen
            mask = np.ones(logits.shape, dtype=bool)
            mask[legal_moves[0], legal_moves[1]] = False
            logits[mask] = -100
            
            # put probabilities through softmax
            exps = np.exp(logits)
            probs = exps / np.sum(exps)
            
            # cache the values
            self.positions_cache[state_id] = (probs, values)
            
            return probs, values, legal_moves
    
    def build_MCTS(self, state):
        root = MCTS.Node(state.copy())
        self.mcts = MCTS.MCTS(root, self.cpuct)
        
    def change_MCTS_root(self, new_root):
        self.mcts.root = self.mcts.tree[new_root.id]
    
    def take_random_action(self, env):
        action = random.choice(env.get_legal_moves_idxs())
        next_state, result, complete = env.update(action)
        env.undo_move()
        return (action, 0, 0, 0, next_state, result, complete)
    
    def save_policy(self, prefix):
        fout = open("{}policy_{}".format(prefix, self.name), 'wb')
        pickle.dump(self.model, fout)
        fout.close()
    
    def load_policy(self, name, prefix=None):
        self.model = pickle.load(open("{}policy_{}".format(prefix, name), 'rb'))
    
    def get_metrics(self):
        return {'loc_loss': self.loc_loss,
                'piece_loss': self.piece_loss,
                'loc_accuracy': self.loc_accuracy,
                'piece_accuracy': self.piece_accuracy,
                'reward': self.total_rewards,
                'average_reward': self.average_reward,
                'regret': self.regret,
                'invalid_moves': self.invalid_moves}
    
    def __str__(self):
        return "{}: {}".format(self.name, self.pieces)
    
class Human(Player):
    def __init__(self, name, env, symbol):
        super().__init__(name, env, symbol, 0, 0)
    
    def move(self, state, tau=None):
        action = None
        
        human_pieces = list(filter(lambda piece: piece["stack_number"]==0, state.pieces[state.pieces_idx]))
        print("Your pieces are: {}".format(list(map(lambda x: "{} @ {}".format(x["size"], x["location"]), human_pieces))))
        row = int(input("Type a row to move to: "))
        col = int(input("Type a col to move to: "))
        idx = int(input("Type the index of the piece: "))
        action = (human_pieces[idx], (row, col))
        next_state, result, complete = state.update(action)
        
        return (action, 0, 0, 0, next_state, result, complete)

class Raw_NN(Player):
    def __init__(self, name, env, model):
        super().__init__(name, env, 0, 0, model)
    
    def move(self, state, tau=None):
        probs, value, legal_moves = self.predict_state(MCTS.Node(state))
        values = np.zeros(len(legal_moves.T))
        
        for idx, action in enumerate(legal_moves.T):
            new_state, val, complete = state.update(action)
            prob, val, lm = self.predict_state(MCTS.Node(new_state))
            state.undo_move()
            values[idx] = val
            
        action = legal_moves.T[np.argmax(values)]
        print(probs)
        next_state, result, complete = state.update(action)
        
        return (action, None, None, values, next_state, result, complete)

class Raw_MCTS(Player):
    def __init__(self, name, env, num_sims, cpuct):
        super().__init__(name, env, num_sims, cpuct, None)
    
    def predict_state(self, state):
        moves = state.env.get_legal_moves_idxs()
        legal_moves = np.array(moves).T
        
        