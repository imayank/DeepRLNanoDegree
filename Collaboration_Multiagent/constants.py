BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_QNET = 1e-4          # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NOISE_SCALE = 0.1       # scale the noise to be added
UPDATE_AFTER = 1000     # Minimum number of samples in the buffer after which learning can start
UPDATE_FREQ = 50        # Frequency of update (per 50 steps). NOT USED
## Related to priority buffer. NOT USED AT PRESENT
ALPHA = 0.6             # factor controling amount of prioritization
REPLAY_EPS=1e-6         # added to priority to facilitate exploration
BETA = 1                # for weights decay

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
