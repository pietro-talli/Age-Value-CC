import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tikzplotlib
from matplotlib.ticker import StrMethodFormatter

from utils import load_and_save_results

def load_data(filename):
    try:
        data = pd.read_csv(filename)
    except:
        print(filename," not found!")
        return None
    return data

def plot_curves(filenames):
    df = []
    for filename in filenames:
        df.append(load_data('csv/'+filename))

    assert df[0] is not None, "Data not found!"
    assert 'density' in df[0].columns, "Density column not found!"
    assert 'beta' in df[0].columns, "Beta column not found!"

    densities = df[0]['density'].unique()
    print(densities)

    for i in range(len(densities)):
        plt.figure()

        for ii, name in enumerate(filenames):

            curr_df = df[ii].loc[df[ii]['density'] == densities[i]]

            # Sort by beta
            curr_df = curr_df.sort_values(by=['beta'])

            avg_c = 'avg_c'
            avg_r = 'avg_r'
            std_r = 'std_r'

            # nromalize rewards
            curr_df[avg_r] = curr_df[avg_r]/np.max(curr_df[avg_r])

            # nromalize stds
            curr_df[std_r] = curr_df[std_r]/np.max(curr_df[avg_r])

            plt.plot(curr_df[avg_c], curr_df[avg_r], label=name[:-4])
            plt.fill_between(curr_df[avg_c], curr_df[avg_r]-curr_df[std_r], curr_df[avg_r]+curr_df[std_r], alpha=0.2)

        plt.ylabel('average reward')
        plt.xlabel('average communication rate')

        try:
            tikzplotlib.save('./figures/latex/reward_over_comm_rate_d_' + str(densities[i]) + '.tex')
        except:
            print("Error saving tikz for density ", densities[i])

        plt.legend()
        plt.savefig('./figures/png/reward_over_comm_rate_d_' + str(densities[i]) + '.png')
        plt.close()

def plot_images(filename): 
    df = load_data('csv/'+filename)
    assert df is not None, "Data not found!"
    assert 'density' in df.columns, "Density column not found!"
    assert 'beta' in df.columns, "Beta column not found!"

    densities = df['density'].unique()
    # Sort densities in descending order
    densities = np.sort(densities)[::-1]
    print(densities)

    

    mat_r = np.zeros((14,21))
    mat_c = np.zeros((14,21))
    for i in range(0,14):
        curr_df = df.loc[df['density'] == densities[i]]
        curr_df = curr_df.sort_values(by=['beta'])
        rewards = np.array(curr_df['avg_r'])
        rewards = np.nan_to_num(rewards, nan=0)
        mat_r[i] = rewards#/np.max(rewards)
        costs = np.array(curr_df['avg_c'])
        costs = np.nan_to_num(costs, nan=0)
        mat_c[i] = costs

    d_s = [str(round(float(label), 2)) for label in densities]
    b_s = [str(round(float(label), 2)) for label in curr_df['beta'].unique()]

    plt.figure()
    plt.title('REWARD MATRIX (PI)')
    ax = plt.imshow(mat_r, vmin=0.1)
    plt.colorbar()
    plt.yticks(np.arange(0.1, 14, 1), d_s[:-1])
    plt.xticks(np.arange(0, 21, 2), [b_s[i] for i in range(0,21,2)])
    plt.ylabel('density')
    plt.xlabel('beta')

    tikzplotlib.save('./figures/latex/reward_matrix_'+filename[:-4]+'.tex')
    plt.savefig('./figures/png/reward_matrix_'+filename[:-4]+'.png')

    plt.figure()
    plt.title('COMMUNICATION COST MATRIX (PI)')
    plt.imshow(mat_c)
    plt.colorbar()
    
    plt.yticks(np.arange(0.1, 14, 1), d_s[:-1])
    plt.xticks(np.arange(0, 21, 2), [b_s[i] for i in range(0,21,2)])
    plt.ylabel('density')
    plt.xlabel('beta')

    tikzplotlib.save('./figures/latex/cost_matrix_'+filename[:-4]+'.tex')
    plt.savefig('./figures/png/cost_matrix_'+filename[:-4]+'.png')


#filename = 'results.csv'

#load_and_save_results('./triangle_random_mat', 10)

#plot_curves([filename])
#plot_images(filename)