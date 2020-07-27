import numpy as np
import config
import logging
#logging.basicConfig(filename="logs/MCTS.log", level=logging.INFO)
class Node:
    def __init__(self, env):
        self.env = env
        self.turn = env.turn
        self.id = env.id
        self.edges = []
        logging.info("Node created")
    
    @property
    def is_leaf(self):
        return len(self.edges) == 0

class Edge:
    def __init__(self, in_node, out_node, prior, action):
        self.id = "{}|{}".format(in_node.env.id, out_node.env.id)
        self.in_node = in_node
        self.out_node = out_node
        self.turn = in_node.env.turn
        self.action = action
        # N: number of times the action from the state has been chosen
        # W: total value of next state
        # Q: mean value of next state
        # P: probability of selecting this action
        self.data = {'N':0, 'W':0, 'Q':0, 'P': prior}
        logging.info("Edge created")

class MCTS:
    def __init__(self, root, cpuct):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.add_node(root)
        logging.info("New Monte Carlo Tree created")
    
    def __len__(self):
        return len(self.tree)
    
    def go_to_leaf(self):
        edges = []
        nodes_visited = {}
        curr_node = self.root
        complete = False
        value = 0
        logging.info("Navigating to leaf . . .")
        while (not curr_node.is_leaf):
            maxQU = -999999
            if curr_node == self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(curr_node.edges))
            else:
                epsilon = 0
                nu = [0] * len(curr_node.edges)
            
            nb = sum(map(lambda edge: edge[1].data['N'], curr_node.edges))
            
            idx = np.random.choice(range(len(curr_node.edges)))
            (sim_action, sim_edge) = curr_node.edges[idx]
            for idx, (action, edge) in enumerate(curr_node.edges):
                # choose the leaf that is most confident according to the formula
                U = self.cpuct * ((1-epsilon)*edge.data['P'] + epsilon*nu[idx]) * np.sqrt(nb) / (1+edge.data['N'])
                Q = edge.data['Q']
                if Q + U > maxQU and not np.isnan(Q+U):
                    maxQU = Q + U
                    sim_action = action
                    sim_edge = edge
            
            logging.info("Updating environment . . .")
            next_state, value, complete = curr_node.env.update(sim_action)
            curr_node.env.undo_move()
            curr_node = sim_edge.out_node
            
            num_visits = nodes_visited.get(curr_node.id, 0)
            if num_visits == 0:
                nodes_visited[curr_node.id] = 1
            else:
                nodes_visited[curr_node.id] += 1

                if nodes_visited[curr_node.id] >= 3:
                    complete = True
                    value = 0
                    break
            
            logging.info("Current node:\n{}".format(str(curr_node.env)))
            edges.append(sim_edge)
        
        return curr_node, value, complete, edges
    
    def update_nodes(self, leaf, value, edges):
        logging.info("Updating nodes . . .")
        curr_turn = leaf.env.turn
        
        for edge in edges:
            turn = edge.turn
            direction = 1 if turn == curr_turn else -1
            
            edge.data['N'] += 1
            edge.data['W'] += value * direction
            edge.data['Q'] = edge.data['W'] / edge.data['N']
    
    def add_node(self, node):
        self.tree[node.id] = node
