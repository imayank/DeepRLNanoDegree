import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class QNetwork(nn.Module):   ### Critic --- estimates state-action-value 
    
    def __init__(self,state_size, action_size, seed):
        super(QNetwork, self).__init__()
        
        torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_size,256)
        self.linear2 = nn.Linear(256+action_size,256)
        self.linear3 = nn.Linear(256,128)
        self.linear4 = nn.Linear(128,1)
        self.reset_parameters()
        
    def forward(self, state, action):
        x = F.leaky_relu(self.linear1(state))
        inp = torch.cat((x,action),dim = 1)
        x = F.leaky_relu(self.linear2(inp))
        x = F.leaky_relu(self.linear3(x))
        z = self.linear4(x)
        return z
    
    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(*hidden_init(self.linear3))
        self.linear4.weight.data.uniform_(-3e-3, 3e-3)
    

class ActorNetwork(nn.Module):   ### actor --- estimates action 
                                       ### that maximizes state-action value
                                       ### also it is greedy policy
    
    def __init__(self,state_size, action_size, seed):
        super(ActorNetwork, self).__init__()
        
        torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_size,256)
        self.linear2 = nn.Linear(256,action_size)
        
        self.activation = nn.Tanh()
        self.reset_parameters()
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        z = self.linear2(x)
        
        return self.activation(z)
    
    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(-3e-3, 3e-3)