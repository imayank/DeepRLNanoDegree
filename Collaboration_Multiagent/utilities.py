import torch
import numpy as np
import random
import copy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def np_to_tensor(arr):
    """ Convert numpy array to tensor and placing it on device
    """
    return torch.from_numpy(arr).to(dtype=torch.float32, device= device)

def tensor_to_np(t):
    """Convert tensor to numpy array"""
    return t.cpu().detach().numpy()



class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state