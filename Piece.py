class Piece:
    counter = 0
    def __init__(self, location, size, stack_number, idx, owner):
        self.location = location
        self.size = size
        self.stack_number = stack_number
        self.idx = idx
        self.owner = owner
        self.id = Piece.counter
        Piece.counter += 1
        Piece.counter %= 12
        
    @property
    def is_on_board(self):
        return isinstance(self.location, tuple)
    
    @property
    def is_top_of_stack(self):
        return self.stack_number == 0
    
    def copy(self):
        return Piece((int(self.location[0]), int(self.location[1])) if self.is_on_board else int(self.location), int(self.size), int(self.stack_number), int(self.idx), int(self.owner))
    
    def __eq__(self, other):
        return self.location == other.location and self.size == other.size and self.stack_number == other.stack_number and self.id == other.id
    
    def __str__(self):
        return "{} @ {}".format(self.size, self.location)

if __name__ == '__main__':
    p = [Piece(0, 0, 0, 0, 0) for i in range(100)]
    print(list(map(lambda x: x.id, p)))
    