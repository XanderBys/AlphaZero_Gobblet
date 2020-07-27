import random, time
import pickle
import numpy as np
from State import State
import MCTS
import config
import logging
#logging.basicConfig(filename="logs/Player.log", level=logging.INFO)
class Player:
    def __init__(self, name, env, num_sims, cpuct, model):
        self.GAMMA = 0.9
        
        self.name = name
        self.env = env
        self.samples = []
        self.num_sims = num_sims
        self.cpuct = cpuct
        self.model = model
        self.time_sims = 0
        
        self.train_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
    
    def run_simulation(self):
        leaf, value, done, edges = self.mcts.go_to_leaf()
        logging.info("Navigated to leaf")
        value, edges = self.evaluate_state(leaf, value, done, edges)
        logging.info("Evaluated state")
        self.mcts.update_nodes(leaf, value, edges)
        logging.info("Updated nodes")
        
    def evaluate_state(self, state, value, complete, edges):
        if not complete:
            logging.info("Predicting state . . .")
            value, probs, legal_moves = self.predict_state(state.env)
            
            logging.info("Iterating through legal moves . . .")
            for action in legal_moves.T:
                new_state, val, complete = state.env.update(action)
                state.env.undo_move()
                if new_state.id not in set(self.mcts.tree):
                    node = MCTS.Node(new_state.copy())
                    self.mcts.add_node(node)
                else:
                    node = self.mcts.tree[new_state.id]
                new_edge = MCTS.Edge(state, node, probs[action[0], action[1]], action)
                state.edges.append((action, new_edge))
        
        return (value, edges)
    
    def predict_state(self, state):
        values, logits = self.model.predict_one(state.binary)
        values = values[0]
        logits = logits[0].reshape(12, 16)
        
        # make sure illegal moves aren't chosen
        moves = state.get_legal_moves_idxs()
        legal_moves = np.array(moves).T
        if len(legal_moves)==0:
            print(state)
        mask = np.ones(logits.shape, dtype=bool)
        mask[legal_moves[0], legal_moves[1]] = False
        logits[mask] = -100
        
        # put probabilities through softmax
        exps = np.exp(logits)
        probs = exps / np.sum(exps)
        
        return (values, probs, legal_moves)
    
    def move(self, state, tau):
        if self.mcts is None or state.id not in self.mcts.tree:
            self.build_MCTS(state)
        else:
            self.change_MCTS_root(state)
        
        for sim in range(self.num_sims):
            self.run_simulation()
        logging.info("Simulations complete")
        pi, vals = self.get_action_vals(1)
        action, value = self.choose_action(pi, vals, tau)
        tree_state = self.mcts.root.env
        #action = (tree_state.pieces[tree_state.pieces_idx][action[0]], (action[1] // tree_state.NUM_COLS, action[1] % tree_state.NUM_COLS))
        try:
            next_state, result, complete = tree_state.update(action)
            tree_state.undo_move()
        except ValueError as err:
            print(state)
            print(tree_state)
            raise err
        
        predicted_value = -1*self.predict_state(next_state)
        
        return (action, pi, value, predicted_value, next_state, result, complete)
    
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
    
    def train(self, memory):
        # train the model based on the reward
        logging.info("Beginning training . . . ")
        for i in range(config.TRAINING_LOOPS):
            logging.info("Formatting data . . .")
            batch = random.sample(memory, min(config.BATCH_SIZE, len(memory)))
            states = np.array([sample['state'].binary for sample in batch])
            targets = {'value_head': np.array([sample['value'] for sample in batch]),
                       'policy_head': np.array([sample['AV'] for sample in batch]).reshape(len(batch), 12*16)}
            
            logging.info("Training neural network . . . ")
            hist = self.model.train_batch(states.reshape(len(batch), 64, 4, 2), targets, epochs=config.EPOCHS).history

            self.train_loss.extend(map(lambda x: round(x,4), hist['loss']))
            self.train_value_loss.extend(map(lambda x: round(x,4), hist['value_head_loss']))
            self.train_policy_loss.extend(map(lambda x: round(x,4), hist['policy_head_loss']))
        
    def build_MCTS(self, state):
        root = MCTS.Node(state.copy())
        self.mcts = MCTS.MCTS(root, self.cpuct)
        
    def change_MCTS_root(self, new_root):
        self.mcts.root = self.mcts.tree[new_root.id]
    
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
        
        human_pieces = list(filter(lambda piece: piece.is_top_of_stack, state.pieces[state.pieces_idx]))
        print("Your pieces are: {}".format(list(map(str, human_pieces))))
        row = int(input("Type a row to move to: "))
        col = int(input("Type a col to move to: "))
        idx = int(input("Type the index of the piece: "))
        action = (human_pieces[idx], (row, col))
        next_state, result, complete = state.update(action)
        
        return (action, 0, 0, 0, next_state, result, complete)
