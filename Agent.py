"""
Agent
Created on 10-03-2025
Reinforcemnt Learning 2025 A1, Leiden University

Adapted from assignment A0

@author(s): Sam Beckers
"""
import numpy as np
from Helper import softmax, argmax
from Neural_Network import NeuralNetwork, ConvNeuralNetwork
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size) # Use deque for fast appends
        self.batch_size = batch_size
    
    def add_experience_to_buffer(self, s, a, r, s_next, done):
        """
        Add a tuple of an experience to the buffer
        """
        self.buffer.append((s, a, r, s_next, done))
    
    def sample(self):
        """
        Sample a random batch from the buffer
        """
        batch = random.sample(self.buffer, k=self.batch_size)
        s, a, r, s_next, done = map(np.array(), zip(*batch)) # Unzip the batch and map to arrays
        # s, s_next = np.vstack(s), np.vstack(s_next) # Stack state arrays 

        return s, a, r, s_next, done

class DQN_BaseAgent:
    def __init__(self, n_states, n_actions, learning_rate, gamma, neurons, UTDR, buffer_size, batch_size):
        """ Base class for DQN agents

        Args:
            n_states (int): number of states
            n_actions (int): number of actions
            learning_rate (float): learning rate
            gamma (float): discount factor
            neurons (int): number of neurons in the hidden layers
            UTDR (int): Update-to-Data Ratio
            buffer_size (int): size of the replay buffer
            batch_size (int): size of the batch
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.neurons = neurons
        self.UDTR = UTDR
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Initialize Neural Networks for function approximation of Q(s,a)
        self.network = NeuralNetwork(n_states, n_actions, neurons)
        self.optim = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Experience Replay (ER)
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # Target Network (TN)
        self.target_network = NeuralNetwork(n_states, n_actions, neurons)
        self.target_network.load_state_dict(self.network.state_dict()) # Copy the state paramaters of the NN to the TN
        self.target_network.eval()
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        """Select an action based on the policy

        Args:
            s (tensor): state
            policy (str): policy to use for selecting actions
            epsilon (float): epsilon for epsilon-greedy policy
            temp (float): temperature for softmax policy

        Returns:
            a (int): action
        """
        if isinstance(s, torch.Tensor):
             s_tensor = s.clone().detach().float()
        else:
            s_tensor = torch.tensor(s).float()
        with torch.no_grad():
            Q_sa = self.network(s_tensor).numpy().squeeze()

        if policy == 'greedy':
            a = argmax(Q_sa)
            
        elif policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            if np.random.rand() < epsilon: #epsilon = 1 gives random action
                a = np.random.randint(0,self.n_actions)
            else: #epsilon = 0 gives greedy action
                a = argmax(Q_sa)
                 
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            a = argmax(softmax(Q_sa,temp))
              
        return a
        
    def update(self):
        raise NotImplementedError('For each agent you need to implement its specific back-up method') # Leave this and overwrite in subclasses in other files

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=500):
        """Evaluate the agent

        Args:
            eval_env (environment): environment to evaluate the agent
            n_eval_episodes (int): number of episodes to evaluate
            max_episode_length (int): maximum length of an episode

        Returns:
            mean_return (float): mean return
        """
        self.network.eval() # Set network to evaluation mode
        returns = []  # list to store the reward per episode

        for i in range(n_eval_episodes):
            s = eval_env.reset()[0] 
            R_ep = 0 # Reward per episode
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy') 
                s_prime, r, done, *_ = eval_env.step(a.item())
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
    
        mean_return = np.mean(returns)
        self.network.train() # Set network back to training mode
        return mean_return
