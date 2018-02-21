# NN optimization -> batch size=60 learning rate=1e-3
# CNN optimization -> batch size=50 learning rate=1e-4


# Neural Network Parameters
INIT_WEIGHT = 1                 # Initial multiplier for weights
INIT_TYPE = 'sigmoid'           # Default activation function of each layer

# Optimization Parameters
GAMMA = 0.99                    # training discount
INIT_OPT = 'Adam'               # Optimization used (names in init under Training Optimization)
DECAY_RATE = 0.99               # Decay rate for RMSProp
LEARNING_RATE_DEFAULT = 1e-4    # Default Learning Rate of optimizer
DROPOUT_PERCENT = 0.8          # Percent Dropout of elements (1 = no dropout)
DROPOUT_INCREASE_PERCENT = 1  # Percent increase of Dropout every batch
                                # (1 for no increase, >1 for increase)
CLIPPING_DEFAULT = 100.0         # Clipping value

# Policy Gradient Parameters
REWARD_CORRECTION = 0           # Correction of reward before running reward correction
BATCH_SIZE_DEFAULT = 50         # Default batch size for training
RENDER_MOD_DEFAULT = 100         # Default When to Render Game
