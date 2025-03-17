"""
Environment
Created on 10-03-2025
Reinforcemnt Learning 2025 A1, Leiden University

@author(s): Sam Beckers
"""
import gymnasium as gym

class CartPole:
    def __init__(self):
        self.env = gym.make("CartPole-v1")

class VectorizedCartPole:
    def __init__(self):
        self.env = gym.make_vec("CartPole-v1", num_envs=4)

class LunarLander:
    def __init__(self):
        self.env = gym.make("LunarLander-v3")