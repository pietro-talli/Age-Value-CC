import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ASoI.mdp import MDP, create_randomized_mdps
from ASoI.smart_agent import PolicyIterationWithSampling
from ASoI.smart_sensor import RemotePolicyIteration
from ASoI.RL_algorithm import RL_Algorithm

class obj:
    def __init__(self, d):
        self.__dict__.update(d)

def load_json(filename: str = 'parameters.json'):
    with open(filename) as file_object:
        data = json.load(file_object, object_hook=obj)
    return data

def load_and_save_results(params):
    results_df = pd.DataFrame(columns=['density', 'beta', 'avg_c', 'avg_r', 'std_r'])
    # List all directories in the path
    dirs = os.listdir(params.path)
    # For each directory
    for dir in dirs:
        # List all subdirectories in the directory
        subdirs = os.listdir(params.path + '/' + dir)
        # For each subdirectory
        for subdir in subdirs:
            r = []
            c = []

            for i in range(params.runs_per_instance):
                filename = params.path + '/' + dir + '/' + subdir + '/run_' + str(i) + '.npy'
                try:
                    res = np.load(filename)
                    r.append(res[0])
                    c.append(res[1])

                except:
                    print(filename, " not found!")
                    continue

            avg_r = np.mean(r)
            std_r = np.std(r)
            avg_c = np.mean(c)

            # get density with 2 decimals
            density = round(res[3], 2)

            # get beta
            beta = res[2]

            # Add results to the dataframe
            results_df = results_df.append({'density': density, 'beta': beta, 'avg_c': avg_c, 'avg_r': avg_r, 'std_r': std_r}, ignore_index=True)

    # Save the dataframe
    results_df.to_csv('./csv/'+params.name + '.csv')
    # Return the dataframe
    return results_df

def load_agent(args, params):
    mdps = create_randomized_mdps(params.mdp.N_s,
                                params.mdp.N_a,
                                params.mdp.gamma,
                                params.mdp.r_seed)



    mdp = mdps[args.d_idx]

    if params.strategy == "pull":
        agent = PolicyIterationWithSampling(mdp, params.horizon, params.t_max)
        filename = params.path + 'density_' +args.density + '/beta_' + args.beta + '/policy_0.npy'
        agent.pi = np.load(filename)

    elif params.strategy == "push":
        agent = RemotePolicyIteration(mdp, params.horizon, params.t_max)
        filename = params.path + 'density_' + args.density + '/beta_' + args.beta + '/policy_actuator_0.npy'
        agent.pi_actuator = np.load(filename)
        filename = params.path + 'density_' + args.density + '/beta_' + args.beta + '/policy_sensor_0.npy'
        agent.pi_sensor = np.load(filename)
    else:
        raise ValueError("Strategy must be either 'pull' or 'push'")

    return agent

def get_peakAOI_dist(agent_push: RL_Algorithm, agent_pull: RL_Algorithm, d: str, beta: str, n_iter: int = 1000, reward: str = 'sparse'):
    import tikzplotlib
    push = []
    pull = []

    bins = 10

    heatmap_push = np.zeros((agent_push.mdp.N_s, bins))
    heatmap_pull = np.zeros((agent_pull.mdp.N_s, bins))
    for s in range(agent_push.mdp.N_s):
        
        plt.figure()
        PAOI_push = agent_push.get_PeakAoI(s, n_episodes=n_iter)

        n_bins = max(PAOI_push)

        

        PAOI_pull = agent_pull.get_PeakAoI(s, n_episodes=n_iter)

        heatmap_push[s,:] = np.histogram(PAOI_push, bins=range(0,bins+1), density=True)[0]
        heatmap_pull[s,:] = np.histogram(PAOI_pull, bins=range(0,bins+1), density=True)[0]

        plt.hist(PAOI_pull, bins=range(0,n_bins), density=True, alpha=0.5, label='State '+str(s))
        plt.hist(PAOI_push, bins=range(0,n_bins), density=True, alpha=0.5, label='State '+str(s))

        plt.xlabel('Peak AoI')
        plt.ylabel('Frequency')
        
        
        if not os.path.exists('AOI_figs/'+reward+'/'+d+'/'+beta):
            os.makedirs('AOI_figs/'+reward+'/'+d+'/'+beta)
        
        tikzplotlib.save('AOI_figs/'+reward+'/'+d+'/'+beta+'/state_'+str(s)+'.tex')
        plt.legend(['Push', 'Pull'])
        plt.savefig('AOI_figs/'+reward+'/'+d+'/'+beta+'/state_'+str(s)+'.png')

        for item_pull, item_push in zip(PAOI_pull, PAOI_push):
            pull.append(item_pull)
            push.append(item_push)

        plt.close()

        np.histogram(PAOI_push, bins=range(0,n_bins), density=True)

    plt.figure()
    plt.hist(pull, bins=range(0,20), density=True, alpha=0.5, label='Pull')
    plt.hist(push, bins=range(0,20), density=True, alpha=0.5, label='Push')
    plt.xlabel('Peak AoI')
    plt.ylabel('Frequency')
    

    if not os.path.exists('AOI_figs/'+reward+'/'+d+'/'+beta):
        os.makedirs('AOI_figs/'+reward+'/'+d+'/'+beta)
    
    tikzplotlib.save('AOI_figs/'+reward+'/'+d+'/'+beta+'/all_states.tex')
    plt.legend()
    plt.savefig('AOI_figs/'+reward+'/'+d+'/'+beta+'/all_states.png')
    plt.close()

    plt.figure()
    plt.imshow(heatmap_push.T, vmin=0, vmax=1)    
    plt.colorbar()
    plt.xlabel('State')
    plt.ylabel('Peak AoI')

    if not os.path.exists('AOI_figs/'+reward+'/'+d+'/'+beta):
        os.makedirs('AOI_figs/'+reward+'/'+d+'/'+beta)

    tikzplotlib.save('AOI_figs/'+reward+'/'+d+'/'+beta+'/heatmap_push.tex')
    plt.savefig('AOI_figs/'+reward+'/'+d+'/'+beta+'/heatmap_push.png')
    plt.close()

    plt.figure()
    plt.imshow(heatmap_pull.T, vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel('State')
    plt.ylabel('Peak AoI')
    
    if not os.path.exists('AOI_figs/'+reward+'/'+d+'/'+beta):
        os.makedirs('AOI_figs/'+reward+'/'+d+'/'+beta)
        
    tikzplotlib.save('AOI_figs/'+reward+'/'+d+'/'+beta+'/heatmap_pull.tex')
    plt.savefig('AOI_figs/'+reward+'/'+d+'/'+beta+'/heatmap_pull.png')
