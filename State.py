import numpy as np

class State:  
    def __init__(self, board, lower_layers=[]):
        self.board = board
        # dimensions of lower_layers = NUM_ROWS x NUM_COLS x DEPTH-1
        if len(lower_layers) > 0:
            self.lower_layers = lower_layers
        else:
            self.lower_layers = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    
    def get_empty_lower_layer(self):
        return State(self.board.copy())
    
    def __str__(self):
        return str([self.board, self.lower_layers])
    
    def deepcopy(self):
        st = State(self.board.copy())
        st.lower_layers = self.lower_layers.copy()
        return st

if __name__ == '__main__':
    st1 = State([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    st2 = State([[1, 0, 0], [0, 0, 0], [0, 0, 0]])

    print(st1==st2)