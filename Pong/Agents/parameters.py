
# Neural Network Parameters
INIT_WEIGHT = 1                 # Initial multiplier for weights
INIT_TYPE = 'sigmoid'           # Default activation function of each layer

# Optimization Parameters
GAMMA = 0.99                    # training discount
INIT_OPT = 'Adam'               # Optimization used (names in init under Training Optimization)
DECAY_RATE = 0.99               # Decay rate for RMSProp
LEARNING_RATE_DEFAULT = 1e-3    # Default Learning Rate of optimizer

# Policy Gradient Parameters
REWARD_CORRECTION = 0           # Correction of reward before running reward correction
BATCH_SIZE_DEFAULT = 50         # Default batch size for training
RENDER_MOD_DEFAULT = 100         # Default When to Render Game
RAND_ACTION_PROB = 0.2         # Default probability random action is taken
RAND_LOWEST_PROB = 1e-6         # Default lowest probability of random action
RAND_REDUCTION_DIV = 2        # Default diviser for probability of random action
