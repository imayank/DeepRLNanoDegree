import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def hidden_init(layer):
    """ initializing the hidden layer"""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class QNetwork(nn.Module):   ### Critic --- estimates state-action-value 
    
    def __init__(self,state_size, action_size, seed):
        super(QNetwork, self).__init__()
        
        torch.manual_seed(seed)
        self.linear1 = nn.Linear(2*state_size,512)
        self.linear2 = nn.Linear(512+action_size,256)
        self.linear3 = nn.Linear(256+action_size,256)
        self.linear4 = nn.Linear(256,128)
        self.linear5 = nn.Linear(128,1)
        self.reset_parameters()
        
    def forward(self, state, action):
        x = F.leaky_relu(self.linear1(state))
        #print("x: ",x.shape)
        #print("action[0]: ",action[:,0].shape)
        #print("action: ",action[:0].shape)
        inp = torch.cat((x,action[:,0]),dim = 1)
        x = F.leaky_relu(self.linear2(inp))
        inp = torch.cat((x,action[:,1]),dim = 1)
        x = F.leaky_relu(self.linear3(inp))
        x = F.leaky_relu(self.linear4(x))
        z = self.linear5(x)
        return z
    
    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(*hidden_init(self.linear3))
        self.linear4.weight.data.uniform_(*hidden_init(self.linear4))
        self.linear5.weight.data.uniform_(-3e-3, 3e-3)
    

class ActorNetwork(nn.Module):   ### actor --- estimates action 
                                       ### that maximizes state-action value
                                       ### also it is greedy policy
    
    def __init__(self,state_size, action_size, seed):
        super(ActorNetwork, self).__init__()
        
        torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_size,128) 
        self.linear2 = nn.Linear(128,action_size)
        
        self.activation = nn.Tanh()
        self.reset_parameters()
        
    def forward(self, state):
        #print(state.shape)
        x = F.relu(self.linear1(state))
        z = self.linear2(x)
        
        return self.activation(z)
    
    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)