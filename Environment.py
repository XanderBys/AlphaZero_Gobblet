import copy
import numpy as np
from numba import jit
from State import State
import logging
#logging.basicConfig(filename="logs/Environment.log", level=logging.INFO)
class Environment:
    id_cache={}
    
    def __init__(self, NUM_ROWS, NUM_COLS, DEPTH):
        self.state = None
        self.prev_states = set()
        self.duplicate_states = set()
        self.draw_flag = False
        self.turn = None
        self.temp_cp = None
        self.pieces = []
        self.name = 'Gobblet'
        
        self.NUM_ROWS = NUM_ROWS
        self.NUM_COLS = NUM_COLS
        self.DEPTH = DEPTH
        
        self.reset()
        
    def reset(self):
        # resets the board to be empty and the turn to be 'X'
        self.state = State(np.array([[0 for i in range(self.NUM_COLS)] for j in range(self.NUM_COLS)]))
        self.moves_made = set()
        self.duplicate_moves = set()
        self.draw_flag = False
        self.turn = 1
        self.initialize_pieces()

#   @jit(nopython=False)
    def update(self, action, turn=0, check_legal=True):
        # updates the board given an action represented as 2 indicies e.g. [0, 2]
        # returns [next_state, result]
        # where next_state is the board after action is taken
        logging.info("Storing copy of board . . .")
        self.temp_cp = self.copy()
        piece, location = action
        if type(piece) != dict:
            piece = self.pieces[self.pieces_idx][piece]
            location = (location // self.NUM_COLS, location % self.NUM_COLS)
        
        logging.info("Ensuring the move is legal . . .")
        if not self.is_legal((piece, location)):
            if check_legal:
                print(self.state)
                raise ValueError("The action {} is not legal".format((str(piece), location)))
            else:
                return (self.state, 10*self.turn)
        
        if turn == 0:
            turn = self.turn
        
        logging.info("Accounting for special rule . . .")
        if isinstance(piece["location"], tuple):
            # if the piece was on the board, set its origin to be empty
            self.state.board[piece["location"]] = 0
            
            # if it was covering another piece, propogate the change upwards
            lower_loc = self.state.lower_layers[0][tuple(piece["location"])]
            if lower_loc != 0:
                self.undo_lower_layers(tuple(piece["location"]))
           
        # update the board and the player
        logging.info("Updating the board and pieces . . .")
        prev_occupant = int(self.state.board[location])
        self.state.board[location] = self.turn * piece["size"]
        piece["location"] = location

        self.update_pieces()
        
        logging.info("Updating lower layers . . .")
        if prev_occupant != 0:
            self.update_lower_layers((piece, location), prev_occupant)
#        
#        if self["id"] in self.duplicate_states:
#            self.draw_flag=True
#        elif self.id in self.moves_made:
#            self.duplicate_states.add(self.id)
#        else:
#            self.moves_made.add(self.id)

        result = self.get_result(self.state)
        
        # update the turn tracker
        self.turn *= -1
        
        return (self.copy(), result, result is not None)
    
    def undo_move(self):
        # undo the last move made
        self.state = self.temp_cp.state
        self.pieces = self.temp_cp.pieces
        self.turn = self.temp_cp.turn
    
    def undo_lower_layers(self, location, i=-1):
        if i < 0:
            layer = self.state.board
        else:
            layer = self.state.lower_layers[i]
        next_layer = self.state.lower_layers[i+1]    
        dest = next_layer[location]
        layer[location] = int(dest)
        next_layer[location] = 0
#        if i+2 > self.DEPTH-1 and self.state.lower_layers[i+2][location] != 0:
#            self.undo_lower_layers(location, i+1)
#            
        exit_flag = False
        for idx, stack in enumerate(self.pieces):
            if exit_flag:
                break
            for p in stack:
                if p["location"] == location and p["size"] == abs(dest):
                    if (idx == 0 and np.sign(dest) == 1) or (idx == 1 and np.sign(dest) == -1):
                        p["stack_number"] -= 1
                        exit_flag = True
                        
    def update_lower_layers(self, action, prev_occupant, i=0):
        # propogate changes throughout lower layers recursively
        piece, location = action
        layer = self.state.lower_layers[i]
        dest = layer[location]
        if dest != 0 and i < 2:
            try:
                self.update_lower_layers(action, dest, i+1)
            except IndexError:
                print("IndexError in update_lower_layers")
        dest = self.turn * piece["size"]
        self.state.lower_layers[i, location[0], location[1]] = prev_occupant
        exit_flag = False
        for idx, stack in enumerate(self.pieces):
            if exit_flag:
                break
            for p in stack:
                if p["location"] == location and p["size"] == abs(prev_occupant):
                    if (idx == 0 and np.sign(prev_occupant) == 1) or (idx == 1 and np.sign(prev_occupant) == -1):
                        p["stack_number"] += 1
                        exit_flag = True
                        break
    #@jit       
    def get_result(self, state):
        # returns None if the game isn't over, 1 if white wins and -1 if black wins
        
        # check rows
        for row in state.board:
            ones = np.sign(row)
            if abs(sum(ones)) == self.NUM_ROWS:
                return sum(ones) / self.NUM_ROWS
            
        # check columns
        cols = state.board.copy()
        cols.transpose()
        for col in cols:
            ones = np.sign(col)
            if abs(sum(ones)) == self.NUM_COLS:
                return sum(ones) / self.NUM_COLS
        
        # check diagonals
        diags = [state.board.diagonal(), np.fliplr(state.board).diagonal()]
        for diag in diags:
            ones = np.sign(diag)
            if abs(sum(ones)) == self.NUM_ROWS:
                return sum(ones) / self.NUM_ROWS
        
        # check for draws
        # that is, if three identical moves have been made, it's a draw
        if self.draw_flag or len(self.prev_states) > 50: 
            #print("DRAW BY {}".format("REPETITION" if self.draw_flag else "NO MOVES"))
            return 0
            
        return None
    
    def is_legal(self, action, verbose=True):
        piece, location = action
        if type(piece) != dict:
            piece = self.pieces[self.pieces_idx][piece]
            location = (location // self.NUM_COLS, location % self.NUM_COLS)
            
        curr_piece = self.state.board[location]

        # the piece has to be bigger than the one currently there
        if not piece["stack_number"] == 0 or piece["size"] <= abs(curr_piece):
            return False
        
        # implement the rule that a new gobblet on the board must be on an empty space
        if not isinstance(piece["location"], tuple) and curr_piece != 0:
            # exception: if there is three in a row through the desired location, the move is valid
            row = self.state.board[location[0]]
            col = self.state.board[:, location[1]]
            diag = [0 for i in range(self.NUM_ROWS)]
            if location[0]==location[1]:
                diag = self.state.board.diagonal()
            elif location[0]+location[1] == self.NUM_ROWS-1:
                diag = np.fliplr(self.state.board).diagonal()
            
            flag = False
            
            for i in [row, col, diag]:
                if flag:
                    break
                counter = 0
                for j in np.squeeze(i):
                    if j != 0:
                        counter += 1
                    if counter==3:
                        flag = True
                        break
            
            if not flag:
                return False
        
        return True
    
    def get_legal_moves_idxs(self):
        # returns the legal moves that can be taken
        moves = []
        illegal_moves = []
        add_move = moves.append
        is_valid_move = self.is_legal
        for idx, i in enumerate(self.state.board):
            for jIdx, j in enumerate(i):
                for piece in self.pieces[self.pieces_idx]:
                    move = (piece, (idx, jIdx))
                    if is_valid_move(move):
                        add_move((piece["id"], idx*self.NUM_COLS + jIdx))
                    
        return moves
    
    def initialize_pieces(self):
        self.pieces = [[{"location":0, "size":4, "stack_number":0, "id":0}, {"location":1, "size":3, "stack_number":1, "id":1}, {"location":2, "size":2, "stack_number":2, "id":2}, {"location":3, "size":1, "stack_number":3, "id":3},
                        {"location":0, "size":4, "stack_number":0, "id":4}, {"location":1, "size":3, "stack_number":1, "id":5}, {"location":2, "size":2, "stack_number":2, "id":6}, {"location":3, "size":1, "stack_number":3, "id":7},
                        {"location":0, "size":4, "stack_number":0, "id":8}, {"location":1, "size":3, "stack_number":1, "id":9}, {"location":2, "size":2, "stack_number":2, "id":10}, {"location":3, "size":1, "stack_number":3, "id":11}],
                       [{"location":0, "size":4, "stack_number":0, "id":0}, {"location":1, "size":3, "stack_number":1, "id":1}, {"location":2, "size":2, "stack_number":2, "id":2}, {"location":3, "size":1, "stack_number":3, "id":3},
                        {"location":0, "size":4, "stack_number":0, "id":4}, {"location":1, "size":3, "stack_number":1, "id":5}, {"location":2, "size":2, "stack_number":2, "id":6}, {"location":3, "size":1, "stack_number":3, "id":7},
                        {"location":0, "size":4, "stack_number":0, "id":8}, {"location":1, "size":3, "stack_number":1, "id":9}, {"location":2, "size":2, "stack_number":2, "id":10}, {"location":3, "size":1, "stack_number":3, "id":11}]
                       ]
    @property
    def pieces_idx(self):
        return 0 if self.turn == 1 else 1
    
    def update_pieces(self):
        for player in self.pieces:
            for piece in player:
                if piece["id"] % 4 == 0:
                    counter = 0
                if isinstance(piece["location"], tuple):
                    continue
                piece["location"] = counter
                piece["stack_number"] = counter
                counter += 1
    
    @property
    def binary(self):
        # convert the state to a binary matrix
	# final format will be 4 x 4 x 9 (ignore hidden pieces and use only current state for now)
        plane_dim = self.state.board.shape

        color = np.zeros(plane_dim) if self.turn == -1 else np.ones(plane_dim)

        board = self.state.board.reshape(-1) #np.append(self.state.board.reshape(-1), self.state.lower_layers.reshape(-1))

        p1_pos = np.zeros((plane_dim[0], plane_dim[1], 4))
        p1_locs = np.argwhere(np.sign(board.reshape(plane_dim))==1)
        p1_pos[4-np.abs(self.state.board[p1_locs[:,0], p1_locs[:,1]]), p1_locs[:,0], p1_locs[:,1]] = 1

        p2_pos = np.zeros((plane_dim[0], plane_dim[1], 4))
        p2_locs = np.argwhere(np.sign(board.reshape(plane_dim))==-1)
        p2_pos[4-np.abs(self.state.board[p2_locs[:,0], p2_locs[:,1]]), p2_locs[:,0], p2_locs[:,1]] = 1

        p2_pos = np.append(p2_pos, color).reshape((-1, plane_dim[0], plane_dim[1]))
        positions = np.concatenate((p1_pos, p2_pos))
        #player1_positions = np.zeros((len(board), self.DEPTH), dtype=np.int)
	#player1_positions[np.sign(board)==self.turn, np.abs(board[np.sign(board)==self.turn])-1] = 1

        #player2_positions = np.zeros((len(board), self.DEPTH), dtype=np.int)
        #player2_positions[np.sign(board)==-1*self.turn, np.abs(board[np.sign(board)==-1*self.turn])-1] = 1

        #positions = np.stack((player1_positions, player2_positions), axis=-1)
        return positions
    
    @property
    def id(self):
        return str(self.turn)+''.join(map(str, np.append(self.state.board, self.state.lower_layers)))
    
    def copy(self):
        cp = Environment(self.NUM_ROWS, self.NUM_COLS, self.DEPTH)
        cp.state = State(copy.deepcopy(self.state.board))
        cp.state.lower_layers = copy.deepcopy(self.state.lower_layers)
        cp.pieces = [[i.copy() for i in self.pieces[0]], [i.copy() for i in self.pieces[1]]]
        cp.turn = int(self.turn)
        return cp
    
    def display(self):
        for i in self.state.board:
            print(i)
        print()
    
    def __str__(self):
        return "Current state:\n{}\n Pieces: {}\n Turn: {}\n".format(str(self.state), str(list(map(lambda x: "{} @ {}".format(x["size"], x["location"]), self.pieces[0])))+str(list(map(lambda x: "{} @ {}".format(x["size"], x["location"]), self.pieces[1]))), str(self.turn))

if __name__ == '__main__':
    env = Environment(4, 4, 4)
    cp = env.copy()
    print(np.array_equal(env.pieces, cp.pieces))
