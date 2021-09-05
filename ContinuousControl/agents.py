## Make necessary imports
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from utilities import *
from buffers import PriorityReplayBuffer
from constants import *
from networks import *

## Setting the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Class implemetns the agent
class DDPGAgent():
    
    def __init__(self, state_size, action_size, seed):
        """ Initialize the agent.
        state_size: size of the state-vector
        action_size: size of the action-vector
        seed: seed for random process
        """
        self.seed = random.seed(seed)
        #### Initializing Q-value networks
        self.QNetwork_local = QNetwork(state_size, action_size,seed).to(device)
        self.QNetwork_target = QNetwork(state_size, action_size,seed).to(device)
        self.QNet_optim = optim.Adam(self.QNetwork_local.parameters(), lr=LR_QNET, weight_decay=WEIGHT_DECAY)
        
        #### Initialize Actor network
        self.actor_local = ActorNetwork(state_size, action_size, seed).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(),lr=LR_ACTOR)
        
        ### Initialize OU stochastic process for noise
        self.noise_process = OUNoise(action_size, seed)
        
        ### Initialize Priority Buffer
        self.buffer = PriorityReplayBuffer(BUFFER_SIZE, BATCH_SIZE, ALPHA)
        self.t = 1
        np.random.seed(2)
        print("Agent Initialized")
        
    def act(self, state, add_noise=True):
        """Method returns action given the state
        state: is a numpy array
        returns action as numpy array"""
        
        ### Select greedy action(one that maximizes the state-action value)
        self.actor_local.eval() ### Necessary when model contains BN or 
                                ### dropout layer
        with torch.no_grad():
            action = self.actor_local(np_to_tensor(state).unsqueeze(0))
        self.actor_local.train()
        ### add noise for exploration
        if add_noise:
            action = tensor_to_np(action).squeeze(0) + NOISE_SCALE * np.random.randn(4)
            
        return np.clip(action,-1, 1)
    
    def get_random_action(self,num_agents=1, action_size=4):
        """Method returns random action"""
        actions = np.random.randn(num_agents, action_size)
        return np.clip(actions, -1, 1).squeeze(0)
    
    def get_initial_priority(self,state, action, rewards, next_state, done):
        """Method that computes the priority for a 5 step interaction before 
        adding it to the priority buffer. Priority is Temporal difference.
        state: initial state of an interaction
        action: action taken in the initial state
        rewards: list of rewards obtained in 5 step interaction
        next_state: final state reached after 5 step interaction
        done: Whether the episode was over.
        
        returns priority for the given interaction
        """
        ## obtain the discounted sum of rewards from reward list
        ## also obtain final gamma multiplier
        reduced_reward, gamma_multiplier = self.reduce_rewards([rewards])
        
        ## Convert to tensors
        state = np_to_tensor(state).unsqueeze(0)
        action = np_to_tensor(action).unsqueeze(0)
        reduced_reward = np_to_tensor(reduced_reward)
        gamma_multiplier = np_to_tensor(gamma_multiplier)
        next_state = np_to_tensor(next_state).unsqueeze(0)
        done = np_to_tensor(np.array(done).astype(np.uint8))
        
        ## Obtain the action from actor target network
        greedy_action = self.actor_target(next_state)
        ## Compute the temporal difference
        target = reduced_reward + torch.mul( torch.mul(gamma_multiplier , self.QNetwork_target(next_state, greedy_action)) , (1-done))
        with torch.no_grad():
            Q_sa = self.QNetwork_local(state, action)
            td = Q_sa-target
        ## return absolute value of temporal difference as priority    
        return tensor_to_np(td.abs().squeeze())
        
        
    
    def reduce_rewards(self,rewards):
        """ Method computes discounted sum of rewards and final gamma multiplier
        rewards: list of rewards obtained in a multi step interaction
        """
        reduced_rewards = np.zeros((len(rewards),1))
        gamma_multipliers = np.zeros((len(rewards),1))
        
        for i in range(len(rewards)):
            gammas = [GAMMA**t for t in range(len(rewards[i]))]
            
            reduced_rewards[i][0] = np.sum(np.multiply(gammas, rewards[i]))
            gamma_multipliers[i][0] = GAMMA**(len(rewards[i]))
            
        return reduced_rewards, gamma_multipliers
    
    
    def learn(self):
        """ Method that implments the core learning algorithm DDPG"""
        ## obtain sample batch using priority based sampling.
        states, actions, rewards, next_states, dones, weights, sample_inds = self.buffer.sample_batch(BETA)
        
        ## obtain the discounted sum of rewards from reward list
        ## also obtain final gamma multiplier
        reduced_rewards, gamma_multipliers = self.reduce_rewards(rewards)
        
        ## convert to tensors
        states = np_to_tensor(states)
        actions = np_to_tensor(actions)
        reduced_rewards = np_to_tensor(reduced_rewards)
        gamma_multipliers = np_to_tensor(gamma_multipliers)
        next_states = np_to_tensor(next_states)
        dones = np_to_tensor(dones)
        weights = np_to_tensor(np.array(weights))
     
        #### Updating Qnet
        
        ## actions from the target actor network
        greedy_actions = self.actor_target(next_states)
        ## compute temporal difference
        targets = reduced_rewards + torch.mul( torch.mul(gamma_multipliers , self.QNetwork_target(next_states, greedy_actions)) , (1-dones).unsqueeze(1))
        Q_sa = self.QNetwork_local(states, actions)
        
        td_error = targets - Q_sa
        
        ## update the priorities using temporal differences
        self.buffer.update_priority(sample_inds,
                                    (td_error).detach().abs().squeeze().cpu().data.numpy()+REPLAY_EPS)
        
        ## compute the loss, importance sampling weights are used
        loss = ((td_error).pow(2)*weights).mean()
        
        self.QNet_optim.zero_grad()
        loss.backward()
        self.QNet_optim.step()
        
        ### Updating Actor
        pred_actions = self.actor_local(states)
        actor_loss = - self.QNetwork_local(states, pred_actions).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        #### Polyak Updates
        self.soft_update(self.QNetwork_local, self.QNetwork_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
    
    
    def soft_update(self, local_model, target_model, tau):
        """ Soft/Polyak update on the target network
        local_model: local network
        target_model: target network
        tau: Proportion update factor"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)