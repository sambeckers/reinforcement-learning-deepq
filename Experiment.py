"""
Experiment
Created on 10-03-2025
Reinforcemnt Learning 2025 A1, Leiden University

@author(s): Sam Beckers
"""
import numpy as np
import time
from tqdm import tqdm

from DQN import dqn

from Helper import LearningCurvePlot, smooth

def average_over_repetitions(n_repetitions, n_timesteps, learning_rate, gamma, policy='egreedy', 
                    epsilon=None, temp=None, smoothing_window=None, plot=False, n=5, eval_interval=500, buffer_size=10000, batch_size=64, neurons=128):

    returns_over_repetitions = []
    now = time.time()
    
    for rep in tqdm(range(n_repetitions), total=n_repetitions): # Loop over repetitions
        returns, timesteps = dqn(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval, buffer_size, batch_size, neurons)
        returns_over_repetitions.append(returns)
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(np.array(returns_over_repetitions),axis=0) # average over repetitions  
    if smoothing_window is not None: 
        learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve, timesteps  

def experiment():
    # Set hyperparameters
        # Experiment      
    n_repetitions = 5
    smoothing_window = None # Must be an odd number. Use 'None' to switch smoothing off!
    plot = False # Plotting is very slow, switch it off when we run repetitions
    
    # MDP    
    n_timesteps = 501 # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 100
    # max_episode_length = 100
    gamma = 1.0

    # Parameters we will vary in the experiments, set them to some initial values: 
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.05
    temp = 1.0
    learning_rate = 0.1
    n = 5 # only used when backup = 'nstep'

    # Learning Rate
    learning_rates = [0.01, 0.1, 0.5]
    Plot = LearningCurvePlot(title = 'Exploration: Learning Rate')
    for learning_rate in learning_rates:
        learning_curve, timesteps = average_over_repetitions(n_repetitions, n_timesteps, learning_rate, gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
        Plot.add_curve(timesteps, learning_curve, label = 'Learning Rate: {}'.format(learning_rate))
    Plot.save('Learning_Rate.png')

if __name__ == '__main__':
    experiment()

