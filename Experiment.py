"""
Experiment
Created on 10-03-2025
Reinforcemnt Learning 2025 A1, Leiden University

@author(s): Sam Beckers
"""
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

def average_over_repetitions(n_repetitions, n_episodes, learning_rate, gamma, policy='egreedy', 
                    epsilon=None, temp=None, smoothing_window=None, plot=False, eval_interval=500, 
                    neurons=128, UTDR=1, len_buffer=10000, len_batch=64, ER=False, TN=False):

    returns_over_repetitions = []
    now = time.time()
    
    for rep in tqdm(range(n_repetitions), total=n_repetitions): # Loop over repetitions
        returns, episodes = dqn(n_episodes, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval, 
                                neurons, UTDR, len_buffer, len_batch, ER, TN)
        returns_over_repetitions.append(returns)
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(np.array(returns_over_repetitions),axis=0) # average over repetitions  
    if smoothing_window is not None: 
        learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve, episodes  

def experiment():
    # Set hyperparameters
        # Experiment      
    n_repetitions = 5
    smoothing_window = 9 # Must be an odd number. Use 'None' to switch smoothing off!
    plot = False # Plotting is very slow, switch it off when we run repetitions
    
    # MDP    
    n_episodes = 1000 # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 10
    max_episode_length = 500
    gamma = 0.99
    

    # Parameters we will vary in the experiments, set them to some initial values: 
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.01
    temp = 1.0
    learning_rate = 0.001
    neurons = 128
    UTDR = 1
    len_buffer = 10000
    len_batch = 64
    ER = False
    TN = False

    # Learning Rate
    if LR_explore:
        print('Exploring Learning Rate')
        learning_rates = [0.001, 0.01, 0.1]
        Plot = LearningCurvePlot(title = 'Exploration: Learning Rate')
        for lr in learning_rates:
            learning_curve, episodes = average_over_repetitions(n_repetitions, n_episodes, lr, gamma, policy, epsilon, temp, 
                                                                smoothing_window, plot, eval_interval)
            Plot.add_curve(episodes, learning_curve, label = 'Learning Rate: {}'.format(lr))
        Plot.save('Learning_Rate.png')

    # Network Size
    if NS_explore:
        print(learning_rate)
        print('Exploring Network Size')
        neurons_size = [32, 64, 128]
        Plot = LearningCurvePlot(title = 'Exploration: Network Size')
        for ns in neurons_size:
            learning_curve, episodes = average_over_repetitions(n_repetitions, n_episodes, learning_rate, gamma, policy, epsilon, temp,
                                                                smoothing_window, plot, eval_interval, ns)
            Plot.add_curve(episodes, learning_curve, label = 'Network Size: {}'.format(ns))
        Plot.save('Network_Size.png')

    # Update-to-Data Ratio
    if UTDR_explore:
        print('Exploring Update-to-Data Ratio')
        UTDRs = [1, 2, 4]
        Plot = LearningCurvePlot(title = 'Exploration: Update-to-Data Ratio')
        for u in UTDRs:
            learning_curve, episodes = average_over_repetitions(n_repetitions, n_episodes, learning_rate, gamma, policy, epsilon, temp,
                                                                smoothing_window, plot, eval_interval, neurons, u)
            Plot.add_curve(episodes, learning_curve, label = 'UTDR: {}'.format(u))
        Plot.save('UTDR.png')

    # Exploration Factor
    if EF_explore:
        print('Exploring Exploration Factor')
        policy = 'egreedy'
        epsilons = [0.03,0.1,0.3]
        Plot = LearningCurvePlot(title = 'Exploration: Exploration Factor (greedy)')
        for ep in epsilons:
            learning_curve, episodes = average_over_repetitions(n_repetitions, n_episodes, learning_rate, gamma, policy, ep, temp,
                                                                smoothing_window, plot, eval_interval)
            Plot.add_curve(episodes, learning_curve, label = 'Epsilon: {}'.format(ep))
        Plot.save('Epsilon.png')

    if configurations:
        print('Exploring Configurations')
        configs = [('DQN - Naive', False, False), ('DQN - TN', False, True), ('DQN - ER', True, False), ('DQN - TN + ER', True, True)]
        Plot = LearningCurvePlot(title = 'Exploration: Configurations')
        for config in configs:
            learning_curve, episodes = average_over_repetitions(n_repetitions, n_episodes, learning_rate, gamma, policy, epsilon, temp,
                                                                smoothing_window, plot, eval_interval, neurons, UTDR, len_buffer, len_batch, config[1], config[2])
            Plot.add_curve(episodes, learning_curve, label = config[0])
        Plot.save('Configurations.png')


if __name__ == '__main__':
    global LR_explore, NS_explore, UTDR_epxlore, EF_explore, configurations
    LR_explore, NS_explore, UTDR_explore, EF_explore, configurations = False, False, False, False, True

    experiment()