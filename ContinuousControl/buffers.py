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
                                      field_names=["state", "action", 
                                                   "rewards", "next_state",
                                                   "done"])
        self.buffer = deque(maxlen=buffer_size)
        
    def __len__(self):
        ### returns current length
        return len(self.buffer)
    
    def add(self, state, action, rewards, next_state, done):
        """ method to add to the replay buffer
        """
        self.buffer.append( self.interaction(state, action, 
                                             rewards, next_state, done) )
        
    def sample_batch(self):
        """ Method to sample from the buffer.
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
        
        #### if error due to none include if clause to skip none
        return (states, actions, rewards, next_states, dones)
    
## Implements Replay Memory with priority based sampling    
class PriorityReplayBuffer:
    
    def __init__(self, buffer_size,batch_size, alpha):
        """ Method initializes the Priority replay buffer.
        buffer_size: the maximum size of the replay memory
        batch_size: number of instances to be sampled
        alpha: Controls how much prioritization is used, alpha=0 means uniform sampling and 1 means pure priority based sampling
        """
        
        # size of the buffer
        self.max_size = buffer_size
        # size of the batch
        self.batch_size = batch_size
        # memory as a list
        self.memory = []
        # the index where to store the experience
        self.next_ind = 0
        # experience as named tuple
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action", 
                                                  "rewards", "next_state",
                                                  "done"])
        self.alpha = alpha
        
        ### Initialize a segement tree.
        capacity = 1
        while capacity < buffer_size:
            capacity *= 2
        
        self.segment_tree = SegmentTree(capacity)
        self.max_priority = 1.0
        
    def __len__(self):
        return len(self.memory)
    
    def add(self, state, action, rewards, next_state, done,pr):
        """Add a new experience to memory."""
        e = self.experience(state, action, rewards, next_state, done)
        ind = self.next_ind
        
        if self.next_ind >= len(self.memory):
            self.memory.append(e)
        else:
            self.memory[self.next_ind] = e
        self.next_ind = (self.next_ind + 1) % self.max_size
        #print(ind)
        self.segment_tree.set_value(ind, pr**self.alpha)
        
    def get_samples(self,sample_inds):
        """Given the indexes of the samples get the samples from the buffer
        """
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
        """ Method implements proportional priority based sampling.
        """
        total_pri = self.segment_tree.interval_sum(0,len(self.memory))
        interval_length = total_pri/self.batch_size
        
        sample_inds = []
        
        for i in range(self.batch_size):
            cumulative_val = random.random()*interval_length + i * interval_length
            sample_inds.append(self.segment_tree.find_prefixsum_idx(cumulative_val))
            
        return sample_inds, total_pri
    
    def sample_batch(self, beta):
        """method implements the sampling of batch from the buffer based on prioriy
        beta: contors the amount importance sampling weights.
        Importance sampling weights are needed to account for priority based sampling
        """
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
        """ method to update the priorities of the experiences stored at
        batch_inds with the given priority as input.
        """
        assert len(batch_inds) == len(priority)
        
        for ind, p in zip(batch_inds, priority):
            assert 0 <= ind < len(self.memory)
            assert p > 0
            self.segment_tree.set_value(ind,p**self.alpha)
            self.max_priority = max(self.max_priority, p)