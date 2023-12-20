# From Age to Value of Information

## Requirements
Code for simulation of the MDPs. The code is written in Python 3.10 and uses the following packages:
- numpy
- scipy 
- matplotlib 
- tikzplotlib

## Problem definition
First define a ``parameters.json`` file with the parameters of the MDP. The file should contain the following fields:
- mdp:
    - N_s (int): number of states
    - N_a (int): number of actions
    - gamma (float): discount factor
    - r_seed (int): random seed to get reproducible results
    - reward_decay (float): reward decay
- horizon (int): horizon of the MDP
- t_max (int): maximum time the system can wait before communicating
- runs_per_instance (int): number of runs per instance
-beta:
    - min (float): minimum value of beta
    - max (float): maximum value of beta
    - n (int): number of values of beta to consider
- n_test_episodes (int): number of episodes to test the policy
- path (str): path to save the results
- name (str): name of the file to save the results
- strategy (str): either "push" or "pull"

## Run the code
To run the code, use the following command:
```
python pull_updates_test.py \
--density_idx IDX \
--beta BETA \
--param_file parameters.json
```
To get results for the pull-based strategy. IDX (int) is the index of the MDP to train with. BETA (float) is the value of beta to use (for consistency use a value between the range specified in the ``parameters.json`` file). To get results for the push-based strategy, use the following command:
```
python push_updates_test.py \
--density_idx IDX \
--beta BETA \
--param_file parameters.json
```

## Plot the results
To plot the results, use the following command:
```
python get_plots.py
```
This script will automatically look for all the configuration files in the ``params`` folder and plot the results in the folder ``figures``.

## Test for the Peak AoI
To obtain results for the peak AoI, use the following command:
```
python test_PeakAOI.py \
--density_idx IDX \
--beta BETA \
--param_file parameters.json
```
