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

class DQN_BaseAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, neurons, buffer_size, batch_size):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.neurons = neurons
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Initialize Neural Networks for function approximation of Q(s,a)
        self.network = ConvNeuralNetwork()
        self.optim = optim.Adam(self.network.parameters(), lr=learning_rate)
        # self.target_network = NeuralNetwork(n_states, n_actions, neurons, architecture)
        # self.target_network.load_state_dict(self.network.state_dict())
        # self.target_network.eval()
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
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

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        self.network.eval()
        returns = []  # list to store the reward per episode

        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy')
                s_prime, r, done = eval_env.step(a.item())
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        self.network.train()
        return mean_return
