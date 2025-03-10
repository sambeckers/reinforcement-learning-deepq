"""
DQN
Created on 10-03-2025
Reinforcemnt Learning 2025 A1, Leiden University

@author(s): Sam Beckers
"""
from Agent import DQN_BaseAgent
from Environment import CartPole
from Environment import LunarLander
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm

class DQNAgent(DQN_BaseAgent):
    def update(self, s, a, r, s_next, done):
        """
        Update the Q-values
        """
        # check if squeezing is necessary
        s_tensor = torch.FloatTensor(s).unsqueeze(0)
        s_next_tensor = torch.FloatTensor(s_next).unsqueeze(0)
        a_tensor = torch.LongTensor([a]).unsqueeze(1)
        r_tensor = torch.FloatTensor([r]).unsqueeze(0)
        done_tensor = torch.FloatTensor([done]).unsqueeze(0)
        
        # Get Q-values for state and next state
        Q_sa = self.network(s_tensor).gather(1, a_tensor) # gather all Q-values for the state and select the one corresponding to the action in 1 (dim)
        Q_sa_next = self.network(s_next_tensor).max(1)[0].detach() # detach (return new tensor) to prevent backpropagation
        
        # Calculate target
        target_Q_sa = r_tensor + self.gamma * Q_sa_next * (1 - done_tensor)
        
        # Calculate loss
        loss = f.mse_loss(Q_sa, target_Q_sa)

        # Optimize
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


def dqn(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval = 500, 
        buffer_size=10000, batch_size=64, neurons=128):
    ''' runs DQN on a gym environment
    Return: rewards, a vector with the observed rewards at each timestep ''' 

    # Initialize environment and agent
    env = CartPole().env
    eval_env = CartPole().env
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, learning_rate, gamma, neurons, buffer_size, batch_size)

    # Store rewards and evaluation results
    r_tot = 0
    eval_timesteps = []
    eval_returns = []

    for t in tqdm(range(n_timesteps), total=n_timesteps):
        s = env.reset()[0] #s=s_0
        done = False
        while not done:
            a = agent.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
            s_next, r, done, *_ = env.step(a)
            agent.update(s, a, r, s_next, done)
            s = s_next
            r_tot += r
            if done:
                break
        if t % eval_interval == 0 and t != 0:
            eval_return = agent.evaluate(eval_env)
            eval_returns.append(eval_return)
            eval_timesteps.append(t)
            if plot:
                env.render()
    return np.array(eval_returns), np.array(eval_timesteps)   

def test():  
    buffer_size = 10000
    batch_size = 64   
    n_episodes = 1000
    gamma = 0.98
    learning_rate = 0.001
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.01
    temp = 1.0

    plot = True

    dqn(n_episodes, learning_rate, gamma, policy, epsilon, temp, plot, buffer_size, batch_size)

if __name__ == '__main__':
    test()