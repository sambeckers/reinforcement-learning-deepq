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

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, neurons, architecture=1):
        super(DQN, self).__init__()
        self.architecture = architecture

        print(architecture)
        
        if architecture == 1:
            self.input = nn.Linear(input_dim, neurons)
            self.output = nn.Linear(neurons, output_dim)
            print(neurons)
        elif architecture == 2:
            self.input = nn.Linear(input_dim, neurons)
            self.layer = nn.Linear(neurons, neurons)
            self.output = nn.Linear(neurons, output_dim)
            print(neurons)

    def forward(self, x):
        if self.architecture == 1:
            x = f.relu(self.input(x))
            return self.output(x)
        elif self.architecture == 2:
            x = f.relu(self.input(x))
            x = f.relu(self.layer(x))
            return self.output(x)

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