# SELF-PLAY
EPISODES = 30
MCTS_SIMS = 30
TAU_COUNTER = 20
CPUCT = 4
EPSILON = 0.2
ALPHA = 0.8
MEMORY_CAP = 15000

# RETRAINING
<<<<<<< HEAD
MINIBATCH_SIZE = 8
BATCH_SIZE = 256
EPOCHS = 2
REG_CONST = 0.0001
LEARNING_RATE = 0.1
=======
BATCH_SIZE = 256
EPOCHS = 2
REG_CONST = 0.0001
LEARNING_RATE = 0.05
>>>>>>> c04eca8fe848170ed7fd1c6d821366c36cc40f26
MOMENTUM = 0.9
TRAINING_LOOPS = 50

# MODEL
NUM_FILTERS = 75
KERNEL_SIZE = (4, 4)
<<<<<<< HEAD
=======
INPUT_SHAPE = (None, 64, 4, 2)
>>>>>>> c04eca8fe848170ed7fd1c6d821366c36cc40f26
NUM_HIDDEN = 8
HIDDEN_LAYERS = [{'filters':NUM_FILTERS, 'kernel_size': KERNEL_SIZE} for i in range(NUM_HIDDEN)]

# EVALUATION
EVAL_EPISODES = 25
SCORING_THRESHOLD = 1.22