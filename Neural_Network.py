"""
Neural_Network
Created on 10-03-2025
Reinforcemnt Learning 2025 A1, Leiden University

@author(s): Sam Beckers
"""
import torch
import torch.nn as nn
import torch.nn.functional as f

class NeuralNetwork(nn.Module):
    """
    Simple Neural Network with 3 layers
    """
    def __init__(self, s_dim, a_dim, neurons):
        """
        Args:
            s_dim (int): dimension of the state space
            a_dim (int): dimension of the action space
            neurons (int): number of neurons in the hidden layers
        """
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(s_dim, neurons) 
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, a_dim)
    
    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return self.fc3(x).float()

class DuelingNeuralNetwork(nn.Module):
    """
    Neural Network for Dueling DQN
    """
    def __init__(self, s_dim, a_dim, neurons):
        super(DuelingNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(s_dim, neurons)
        self.relu = nn.ReLU()

        self.fc_value = nn.Linear(neurons, neurons)
        self.fc_adv = nn.Linear(neurons, neurons) # Advantage

        self.value = nn.Linear(neurons, 1) 
        self.adv = nn.Linear(neurons, a_dim) 

    def forward(self, state):
        x = self.relu(self.fc1(state))
        
        value = self.relu(self.fc_value(x))
        adv = self.relu(self.fc_adv(x))
        
        value = self.value(value) 
        adv = self.adv(adv)
        
        adv_mean = torch.mean(adv)
        Q = value + adv - adv_mean

        return Q

class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x