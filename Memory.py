import numpy as np
from collections import deque
import config
class Memory:
    def __init__(self):
        self.max_memory = config.MEMORY_CAP
        self.long_term = deque(maxlen=self.max_memory)
        self.short_term = deque(maxlen=self.max_memory)

    def add_sample(self, sample):
        state, AVs = sample
        self.short_term.append({'board': state.state, 'state': state, 'id': state.id, 'AV': AVs, 'turn': state.turn})
        
    def update_long_term(self):
        for i in self.short_term:
            self.long_term.append(i)
            
        self.clear_short_term()
        
    def clear_short_term(self):
        self.short_term = deque(maxlen=self.max_memory)