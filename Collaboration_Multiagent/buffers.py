import numpy as np
from collections import namedtuple, deque
from segment_tree import SegmentTree
from constants import *
import random

#### DO NOT STORE TENSORS
#### IT RETURNS ARRAY NOT TENSORS ON SAMPLING
#### Input interface, output interface and functionality should be clear
class ReplayBuffer:
    
    def __init__(self, buffer_size, batch_size, seed):
        """Initializes the Replay buffer
        buffer_size: the maximum size of the replay memory
        batch_size: number of instances to be sampled
        seed: for random uniform sampling
        """
        
        self.buffer_size = buffer_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        
        self.interaction = namedtuple("Experience", 
                                      field_names=["states", "actions", 
                                                   "rewards", "next_states",
                                                   "dones"])
        self.buffer = deque(maxlen=buffer_size)
        
    def __len__(self):
        ### returns current length
        return len(self.buffer)
    
    def add(self, states, actions, rewards, next_states, dones):
        """ method to add to the replay buffer
        """
        self.buffer.append( self.interaction(states, actions, 
                                             rewards, next_states, dones) )
        
    def sample_batch(self, agent_num):
        """ Method to sample from the buffer.
        agent_num: agent for which to sample
        returns: states, actions, rewards, next_states, dones
        """
        batch = random.sample(self.buffer, k=self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch) 
        ## above line creates tuple of states etc.
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones).astype(np.uint8)
        
        #print(states.shape, actions.shape, rewards.shape,next_states.shape,dones.shape)
        #### if error due to none include if clause to skip none
        return (states, actions, rewards[:,agent_num], next_states, dones[:,agent_num])

    

### Incomplete implementation of priority buffer. PLEASE IGNORE AT PRESENT
### Uses segment tree
class PriorityReplayBuffer:
    
    def __init__(self, buffer_size,batch_size, alpha):
        
        self.max_size = buffer_size
        self.batch_size = batch_size
        self.memory = []
        self.next_ind = 0
        self.experience = namedtuple("Experience", 
                                     field_names=["states", "actions", 
                                                  "rewards", "next_states",
                                                  "dones"])
        self.alpha = alpha
        
        
        capacity = 1
        while capacity < buffer_size:
            capacity *= 2
        
        self.segment_tree = SegmentTree(capacity)
        self.max_priority = 1.0
        
    def __len__(self):
        return len(self.memory)
    
    def add(self, states, actions, rewards, next_states, dones, prs):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        ind = self.next_ind
        
        if self.next_ind >= len(self.memory):
            self.memory.append(e)
        else:
            self.memory[self.next_ind] = e
        self.next_ind = (self.next_ind + 1) % self.max_size
        #print(ind)
        self.segment_tree.set_value(ind, prs**self.alpha)
        
    def get_samples(self,sample_inds):
        batch = [self.memory[ind] for ind in sample_inds]
        
        states, actions, rewards, next_states, dones = zip(*batch) 
        ## above line creates tuple of states etc.
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones).astype(np.uint8)
        
        #### if error due to none include if clause to skip none
        return (states, actions, rewards, next_states, dones)

        
    def proportional_sampling(self):
        
        total_pri = self.segment_tree.interval_sum(0,len(self.memory))
        interval_length = total_pri/self.batch_size
        
        sample_inds = []
        
        for i in range(self.batch_size):
            cumulative_val = random.random()*interval_length + i * interval_length
            sample_inds.append(self.segment_tree.find_prefixsum_idx(cumulative_val))
            
        return sample_inds, total_pri
    
    def sample_batch(self, beta):
        
        sample_inds, total_pri = self.proportional_sampling()
        
        ## maximum weight is needed for normalizing the IS weights
        ## w_j = (N * P_j)^(-beta)
        ## maximum weight therefore corresponds to minimum priority
        max_w = (len(self.memory) * self.segment_tree.get_minimum()) ** (-beta)
        #print("max_w: ",max_w)
        #print("total_pri: ", total_pri)
        ## Calculate IS weights for sampled indices
        weights=[]
        for ind in sample_inds:
            ## normalize the priority
            norm_p = self.segment_tree.get_value(ind)/total_pri
            #print(norm_p)
            ##calculate the IS weight
            w = (len(self.memory) * norm_p) ** (-beta)
            ## append in the list
            weights.append(w)
        #print(type(weights))
        #weights  = torch.from_numpy(np.array(weights)).float().to(device)
        sample = self.get_samples(sample_inds)
        
        return tuple(list(sample)+[weights, sample_inds])
    
    def update_priority(self, batch_inds, priority):
        
        assert len(batch_inds) == len(priority)
        
        for ind, p in zip(batch_inds, priority):
            assert 0 <= ind < len(self.memory)
            assert p > 0
            self.segment_tree.set_value(ind,p**self.alpha)
            self.max_priority = max(self.max_priority, p)