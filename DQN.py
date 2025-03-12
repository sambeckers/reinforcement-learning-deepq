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
    def update(self, s, a, r, s_next, done) -> None:
        """Update the Q-values of the agent

        Args:
            s (array): state
            a (int): action
            r (float): reward
            s_next (array): next state
            done (bool): whether the episode is done
        """
        # Convert to tensors
        s_tensor = torch.FloatTensor(s).unsqueeze(0)
        a_tensor = torch.LongTensor([a]).unsqueeze(1)
        r_tensor = torch.FloatTensor([r]).unsqueeze(0)
        s_next_tensor = torch.FloatTensor(s_next).unsqueeze(0)
        done_tensor = torch.FloatTensor([done]).unsqueeze(0)
        
        # Get Q-values for state and next state
        Q_sa = self.network(s_tensor).gather(1, a_tensor) # Gather all Q-values for the state and select the one corresponding to the action in dim 1
        Q_sa_next = self.network(s_next_tensor).max(1)[0].detach() # Detach (return new tensor)
        
        # Calculate target
        target_Q_sa = r_tensor + self.gamma * Q_sa_next * (1 - done_tensor) 
        
        # Calculate loss
        loss = f.mse_loss(Q_sa, target_Q_sa)

        # Optimize
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


def dqn(n_episodes, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval = 500, 
        neurons=128, UTDR = 1, buffer_size=10000, batch_size=64):
    """Runs DQN on a gym environment

    Args:
        n_episodes (int): number of episodes to run the agent
        learning_rate (float): learning rate
        gamma (float): discount factor
        policy (str): policy to use for selecting actions
        epsilon (float): epsilon for epsilon-greedy policy
        temp (float): temperature for softmax policy
        plot (bool): whether to plot the environment
        eval_interval (int): interval at which to evaluate the agent
        neurons (int): number of neurons in the hidden layers
        UTDR (int): Update-to-Data Ratio
        buffer_size (int): size of the replay buffer
        batch_size (int): batch size for training
    
    Returns:
        np.array: evaluation returns
        np.array: evaluation episodes
    """
    # Initialize environment and agent
    env = CartPole().env
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, learning_rate, gamma, neurons, UTDR, buffer_size, batch_size)

    # Store rewards and evaluation results
    r_tot = 0
    eval_episodes = []
    eval_returns = []

    for e in tqdm(range(n_episodes), total=n_episodes):
        s = env.reset()[0] #s=s_0
        done = False
        steps = 0

        while not done:
            a = agent.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
            s_next, r, done, *_ = env.step(a)
            steps += 1

            if steps % UTDR == 0:
                agent.update(s, a, r, s_next, done)

            s = s_next
            r_tot += r
    
        if e % eval_interval == 0 and e != 0:
            eval_return = agent.evaluate(env)
            eval_returns.append(eval_return)
            eval_episodes.append(e)
            if plot:
                env.render()

    return np.array(eval_returns), np.array(eval_episodes)   

def test():
    n_episodes = 1000
    gamma = 0.98
    learning_rate = 0.001
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.01
    temp = 1.0
    plot = False

    dqn(n_episodes, learning_rate, gamma, policy, epsilon, temp, plot)

if __name__ == '__main__':
    test()