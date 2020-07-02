import numpy as np
import config

class Node:
    def __init__(self, env):
        self.env = env
        self.turn = env.turn
        self.id = env.id
        self.edges = []
    
    @property
    def is_leaf(self):
        return len(self.edges) == 0
    
class Edge:
    def __init__(self, in_node, out_node, prior, action):
        self.id = in_node.env.id + '|' + out_node.env.id
        self.in_node = in_node
        self.out_node = out_node
        self.turn = in_node.env.turn
        # N: number of times the action from the state has been chosen
        # W: total value of next state
        # Q: mean valeu of next state
        # P: probability of selecting this action
        self.data = {'N':0, 'W':0, 'Q':0, 'P': prior}

class MCTS:
    def __init__(self, root, cpuct):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.add_node(root)
    
    def __len__(self):
        return len(self.tree)
    
    def go_to_leaf(self):
        edges = []
        curr_node = self.root
        complete = False
        value = 0
        
        while not curr_node.is_leaf:
            maxQU = -999999
            if curr_node == self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(curr_node.edges))
            else:
                epsilon = 0
                nu = [0] * len(curr_node.edges)
            
            nb = 0
            for action, edge in curr_node.edges:
                nb += edge.data['N']
            
            for idx, (action, edge) in enumerate(curr_node.edges):
                # choose teh leaf that is most confident according to the formula
                U = self.cpuct * ((1-epsilon)*edge.data['P'] + epsilon*nu[idx]) * np.sqrt(nb) / (1+edge.data['N'])
                Q = edge.data['Q']
                if Q + U > maxQU and curr_node.env.is_legal(action, curr_node.env.turn):
                    maxQU = Q + U
                    sim_action = action
                    sim_edge = edge
            
            next_state, value, complete = curr_node.env.update(sim_action, curr_node.env.turn)
            curr_node = sim_edge.out_node
            edges.append(sim_edge)
        
        return curr_node, value, complete, edges
    
    def update_nodes(self, leaf, value, edges):
        curr_turn = leaf.env.turn
        
        for edge in edges:
            turn = edge.turn
            direction = 1 if turn == curr_turn else -1
            
            edge.data['N'] += 1
            edge.data['W'] += value * direction
            edge.data['Q'] = edge.data['W'] / edge.data['N']
    
    def add_node(self, node):
        self.tree[node.id] = node