# Assignment 1 for Reinforcement Learning 2025 (Leiden University): Deep Q-learning
## Submitted by Sam Beckers
The python files in this repository are:
1. DQN.py
2. Agent.py
3. Environment.py
4. NeuralNetwork.py
5. Helper.py
6. Experiment.py
7. requirements.txt

To run the experiments,
1. Install python
2. Install the required libraries using the following command:
   ```bash
   pip install -r requirements.txt
3. Run Experiment.py with any of these arguments:
--LR_explore (help='Explore learning rates')
--NS_explore (help='Explore network sizes')
--UTDR_explore (help='Explore update-to-data ratio')
--EF_explore (help='Explore exploration factors')
--configurations (help='Explore different DQN configurations')
--lunar_bonus (help='Run the Lunar Lander environment')
--dueling_bonus (help='Compare Dueling DQN to Naive DQN')
e.g.
   ```bash
   python Experiment.py --LR_explore
will run the Learning Rate exploration.
