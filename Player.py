import random
import pickle
import math
import copy
import numpy as np
from State import State
from Model import Model
from Memory import Memory
import MCTS
import config

class Player:
    def __init__(self, name, env, num_sims, cpuct, model):
        self.GAMMA = 0.9
        
        self.name = name
        self.env = env
        self.samples = []
        self.num_sims = num_sims
        self.cpuct = cpuct
        self.model = model
    
    def run_simulation(self):
        leaf, value, done, edges = self.mcts.go_to_leaf()
        value, edges = self.evaluate_state(leaf, value, done, edges)
        self.mcts.update_nodes(leaf, value, edges)
    
    def evaluate_state(self, state, value, complete, edges):
        if not complete:
            value, probs, legal_moves = self.predict_state(state.env)
            probs = probs[legal_moves[0], legal_moves[1]]
            
            for idx, action in enumerate(legal_moves.T):
                new_state, val, complete = state.env.update(action, state.env.turn)
                state.env.undo_move()
                if new_state.id not in self.mcts.tree:
                    node = MCTS.Node(state.env.copy())
                    self.mcts.add_node(node)
                else:
                    node = self.mcts.tree[new_state.id]
                
                new_edge = MCTS.Edge(state, node, probs[idx], action)
                state.edges.append((action, new_edge))
        
        return (value, edges)
    
    def predict_state(self, state):
        values, logits = self.model.predict_one(state.binary)
        values = values[0]
        logits = logits[0].reshape(12, 16)
        
        # make sure illegal moves aren't chosen
        moves = state.get_legal_moves_idxs(state.turn)
        legal_moves = np.array(moves[0]).T
        illegal_moves = np.array(moves[1]).T
        
        logits[illegal_moves[0], illegal_moves[1]] = -100
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
        
        pi, vals = self.get_action_vals(1)
        
        action, value = self.choose_action(pi, vals, tau)
        next_state, _, complete = state.update(action, state.turn)
        state.undo_move()
        predicted_value = -1*self.predict_state(next_state)
        
        return (action, pi, value, predicted_value)
    
    def choose_action(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == np.amax(pi))
            action = tuple(random.choice(actions))
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]
        
        value = values[action]
        
        return (action, value)
    
    def get_action_vals(self, tau):
        pi = np.zeros((12, 16), dtype=np.float32)
        vals = np.zeros((12, 16), dtype=np.int32)
        for action, edge in self.mcts.root.edges:
            pi[tuple(action)] = pow(edge.data['N'], 1/tau)
            vals[tuple(action)] = edge.data['Q']
        pi /= (np.sum(pi) * 1.0)
        return pi, vals
    
    def train(self, memory):
        # train the model based on the reward
        for i in range(config.TRAINING_LOOPS):
            batch = random.sample(memory, min(config.BATCH_SIZE, len(memory)))
            
            states = np.array([sample['state'].binary for sampmle in batch])
            targets = {'values': np.array([sample['value'] for sample in batch]),
                       'policy': np.array([sample['AV'] for sample in batch])}
            hist = self.model.train_batch(states, training, epochs=config.EPOCHS).history
            self.train_loss.append(round(hist['loss'][config.EPOCHS-1],4))
            self.train_value_loss.append(round(hist['value_head_loss'][config.EPOCHS-1],4))
            self.train_policy_loss.append(round(hist['policy_head_loss'][config.EPOCHS-1],4))
        
    def build_MCTS(self, state):
        root = MCTS.Node(state)
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
        super().__init__(name, env, symbol, 0)
    
    def choose_action(self, state, moves=None):
        action = None
        
        human_pieces = list(filter(lambda piece: piece.is_top_of_stack, self.pieces))
        print("Your pieces are: {}".format(list(map(str, human_pieces))))
        row = int(input("Type a row to move to: "))
        col = int(input("Type a col to move to: "))
        idx = int(input("Type the index of the piece: "))
        action = (human_pieces[idx], (row, col))
        return action
