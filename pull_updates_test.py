import numpy as np
import os
from joblib import Parallel, delayed
import argparse

from ASoI.mdp import MDP, create_randomized_mdps
from ASoI.smart_agent import PolicyIterationWithSampling
from utils import load_json

parser = argparse.ArgumentParser()
parser.add_argument('--density_idx', type=int, default=0, help='index of the density to be used (default: 0, deterministic MDP)')
parser.add_argument('--beta', type=float, default=0.5, help='beta (cost of communication) (default: 0.5)')
parser.add_argument('--param_file', type=str, default='parameters.json', help='path to the parameter file (default: parameters.json)')

command_line_params = parser.parse_args()

def run_parallel(mdp, beta, params, i):
    policy_iteration = PolicyIterationWithSampling(mdp, params.horizon, params.horizon)
    policy_iteration.run(beta)
    r_q, c_q = policy_iteration.eval_perf(params.n_test_episodes)

    # save results as [average reward, average cost, beta, density, run_id]
    res = np.array([r_q, c_q, beta, mdp.density, i])

    # get density with 2 decimals
    d = str(mdp.density)
    d = d[:d.find('.')+3]

    folder_name = params.path + 'density_' + d + '/beta_' + str(beta) + '/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # save results
    np.save(folder_name + 'run_' + str(i) + '.npy', res)

    try:
        # save the policy
        np.save(folder_name + 'policy_' + str(i) + '.npy', policy_iteration.pi)
    except:
        print('Policy not saved!')

    print('Run ' + str(i) + ' finished')

# Load parameters
params = load_json(command_line_params.param_file)

# Create a random MDP
mdps = create_randomized_mdps(params.mdp.N_s, params.mdp.N_a, params.mdp.gamma, params.mdp.r_seed, params.mdp.reward_decay)

# number of available CPUs
n_cpus = os.cpu_count()

# make folder for results 
if not os.path.exists(params.path):
    os.makedirs(params.path)

# run parallel
Parallel(n_jobs=n_cpus)(delayed(run_parallel)(mdps[command_line_params.density_idx], command_line_params.beta, params, i) for i in range(params.runs_per_instance))