from ASoI.mdp import MDP, create_randomized_mdps
from ASoI.smart_agent import PolicyIterationWithSampling
from ASoI.smart_sensor import RemotePolicyIteration

import numpy as np
from utils import load_json

def test_a_solution(params_file: str, density_idx: int, density: str, beta: str, type: str)->None:
    params = load_json(params_file)
    mdps = create_randomized_mdps(params.mdp.N_s, 
                                  params.mdp.N_a, 
                                  params.mdp.gamma, 
                                  params.mdp.r_seed, 
                                  params.mdp.reward_decay)
    
    mdp = mdps[density_idx]
    if type == 'push':
        smart_sensor = RemotePolicyIteration(mdp, params.horizon, params.horizon)
        try:
            smart_sensor.pi = np.load(params.path + 'density_' + density + '/beta_' + beta + '/policy_0.npy')
        except:
            print('Policy not loaded!')
            return 
        smart_sensor.eval_perf(params.n_test_episodes)
    elif type == 'pull':
        smart_agent = PolicyIterationWithSampling(mdp, params.horizon, params.horizon)
        smart_agent.eval_perf(params.n_test_episodes)
    else:
        raise ValueError('type must be either push or pull')