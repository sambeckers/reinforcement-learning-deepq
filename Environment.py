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


class LunarLander:
    def __init__(self):
        self.env = gym.make("LunarLander-v3")



# episode_over = False
# while not episode_over:
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)

#     episode_over = terminated or truncated

# env.close()