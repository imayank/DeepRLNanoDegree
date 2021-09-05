## importing necessary scripts and packages
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from utilities import *
from buffers import PriorityReplayBuffer, ReplayBuffer
from constants import *
from networks import *

## setting the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### class that initializes an actor
class Actor():
    
    def __init__(self, state_size, action_size, seed):
        """Initialize an actor
        state_size: size of the state
        action_size: size of action
        seed: for random noise process
        """
        self.seed = random.seed(seed)
        
        #### Initialize Actor network
        self.local = ActorNetwork(state_size, action_size, seed).to(device)
        self.target = ActorNetwork(state_size, action_size, seed).to(device)
        self.optim = optim.Adam(self.local.parameters(),lr=LR_ACTOR)
        
        ### Initialize stochastic process for noise
        self.noise_process = np.random.RandomState(seed)
        
        
    def act(self, state, add_noise):
        """ Method that returns action given a state
        state: is a numpy array
        add_noise: whether to add noise for exploration
        returns action as numpy array"""
   
        ### Select greedy action(one that maximizes the state-action value)
        self.local.eval() ### Necessary when model contains BN or 
                                ### dropout layer
        with torch.no_grad():
            action = self.local(np_to_tensor(state).unsqueeze(0)).squeeze(0)
        self.local.train()
        
        ### add noise for exploration
        #print(tensor_to_np(action))
        if add_noise:
            action = tensor_to_np(action) + NOISE_SCALE * self.noise_process.standard_normal(2)
            
        return np.clip(action,-1, 1) ## clipping action value in valid range
    
    def gradient_step(self, critic, states, pred_actions):
        """Method the performs one step of gradient descent on the actor network
        and performs Polyak update on target network
        critic: corresponding local critic network for the actor
        states: sampled states
        pred_actions: actions predicted by the local actor network
        """
        actor_loss = - critic(states.reshape((BATCH_SIZE,-1)), pred_actions).mean() ####critic is QNetwork_local
        
        self.optim.zero_grad()
        actor_loss.backward()
        self.optim.step()
        
        #### Polyak Updates
        self.soft_update(self.local, self.target, TAU)
    
    
    def soft_update(self, local_model, target_model, tau):
        """ Soft/Polyak update on the target network
        local_model: local network
        target_model: target network
        tau: Proportion update factor"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
### class that initializes a critic
class Critic():
    
    def __init__(self, state_size, action_size, priority, seed):
        """Method to initialize a critic
        state_size: size of the state
        action_size: size of the action
        priority: Whether to use priority buffer. NOT IMPLEMENTED AT PRESENT
        seed: seed for random process
        """
        
        #### Initializing Q-value networks
        self.local = QNetwork(state_size, action_size,seed).to(device)
        self.target = QNetwork(state_size, action_size,seed).to(device)
        self.optim = optim.Adam(self.local.parameters(), lr=LR_QNET, weight_decay=WEIGHT_DECAY)
        
        self.priority = priority
        

    
    def soft_update(self, local_model, target_model, tau):
        """ Soft/Polyak update on the target network
        local_model: local network
        target_model: target network
        tau: Proportion update factor"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def gradient_step(self, states, actions, rewards, next_states, dones, gamma_multipliers, greedy_actions):
        """Method to perform one step of gradient descent on local critic network
        states: sampled states
        actions: sampled actions
        rewards: sampled rewards
        next_states: sampled next_states
        dones: sampled array of done or not
        gamma_multipliers: value of GAMMA (discount)
        greedy_actions: array of action values by actors target networks for sampled next_states
        """
        
        #### Updating Qnet
        
        targets = rewards.unsqueeze(1) + torch.mul( torch.mul(gamma_multipliers , self.target(next_states.reshape((BATCH_SIZE,-1)), greedy_actions)) , (1-dones).unsqueeze(1))
        Q_sa = self.local(states.reshape((BATCH_SIZE,-1)), actions)
        
        #print(targets.shape)
        #print(Q_sa.shape)
        #td_error = targets - Q_sa
        
        if self.priority:
            self.buffer.update_priority(sample_inds,(td_error).detach().abs().squeeze().data.numpy()+REPLAY_EPS)
        
        huber_loss = torch.nn.SmoothL1Loss()
        #loss = ((td_error).pow(2)).mean()
        loss = huber_loss(Q_sa, targets.detach())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        ##### Polyak Update
        self.soft_update(self.local, self.target, TAU)
        
### class for multi-agent DDPG    
class MADDPG():
    
    def __init__(self, state_size, action_size, seeds, priority=False):
        """Initialize aactors and critics for MultiAgent DDPG algorithm.
        state_size: size of the state
        action_size: size of the action
        priority: Whether to use priority buffer. NOT IMPLEMENTED AT PRESENT
        seeds: seed for random processes
        """
        #### Initialize critics
        self.critics = [Critic(state_size, action_size, priority, seeds[0]),
                      Critic(state_size, action_size, priority, seeds[1]) ]
 
        #### Initialize Actors
        self.actors = [Actor(state_size, action_size, seeds[0]),
                      Actor(state_size, action_size, seeds[1])] 
        
        ##### Initializing Buffer
        if priority:
            self.buffer = PriorityReplayBuffer(BUFFER_SIZE, BATCH_SIZE, ALPHA)
        else:
            self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seeds[2])
            
        
        
        self.priority = priority
        self.action_size = action_size
        print("Agent Initialized")
        
        
    def learn(self):
        """
        Method that implements the core learning algorithm of MultiAgent DDPG
        """
        
        ## Update every critic and actor
        for ind in range(len(self.critics)):
            ##Sample from priority replay buffer. NOT IMPLEMENTED
            if self.priority:
                states, actions, rewards, next_states, dones, weights, sample_inds = self.buffer.sample_batch(BETA)
            else: ## sample from uniform buffer
                states, actions, rewards, next_states, dones = self.buffer.sample_batch(ind)
            
           
            gamma_multipliers = np.array(GAMMA)
            
            ##convert sampled arrays to tensors
            states = np_to_tensor(states)
            actions = np_to_tensor(actions)
            rewards = np_to_tensor(rewards)
            gamma_multipliers = np_to_tensor(gamma_multipliers)
            next_states = np_to_tensor(next_states)
            dones = np_to_tensor(dones)
            
            ## Action predicted by target actor networks
            greedy_action1 = (self.actors[0].target(next_states[:,0])).unsqueeze(1)
            greedy_action2 = (self.actors[1].target(next_states[:,1])).unsqueeze(1)
                  
            greedy_actions = torch.cat((greedy_action1, greedy_action2), dim = 1) #assuming the shape is  batch,action_shape
            ## Updating critic network
            self.critics[ind].gradient_step(states, actions, rewards, next_states, dones, gamma_multipliers, greedy_actions)
            
            ## actions prediceted by local actor network
            if ind==0:
                actions1 = (self.actors[0].local(states[:,0])).unsqueeze(1)
                self.actors[1].local.eval()
                with torch.no_grad():
                    actions2 = (self.actors[1].local(states[:,1])).unsqueeze(1)
                self.actors[1].local.train()
            else:
                actions2 = (self.actors[1].local(states[:,1])).unsqueeze(1)
                self.actors[0].local.eval()
                with torch.no_grad():
                    actions1 = (self.actors[0].local(states[:,0])).unsqueeze(1)
                self.actors[0].local.train()
            pred_actions = torch.cat((actions1, actions2), dim = 1) #assuming the shape is  batch,action_shape
            ## Updating actor network
            self.actors[ind].gradient_step(self.critics[ind].local, states, pred_actions)
        
    def get_random_action(self,num_agents=2, action_size=2):
        """ Method to choose random action for the agents"""
        actions = np.random.RandomState().standard_normal((num_agents,action_size))
        return np.clip(actions, -1, 1)
    
    def act(self, states,add_noise=True):
        """Method returns actions chosen by the actors
        states: states as seen by the two agents
        add_noise: whther to add noise"""
        
        actions = np.zeros((len(self.actors),self.action_size))
        #print(actions.shape)
        #print(actions[0].shape)
        for ind in range(len(self.actors)):
            actions[ind] = self.actors[ind].act(states[ind],add_noise)
            
        return actions
        