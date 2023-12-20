import numpy as np

class MDP(object):
    def __init__(self, N_s, N_a, gamma, density, reward_decay):
        self.N_s = N_s
        self.N_a = N_a
        self.gamma = gamma
        self.P = np.zeros((N_a, N_s, N_s))
        self.R = np.zeros_like(self.P) #np.random.rand(self.N_a,self.N_s,self.N_s) #np.random.normal(size=(self.N_a,self.N_s,self.N_s)) 
        optimal_state = np.random.randint(0,N_s)
        
        for i in range(N_s):
            self.R[:,:,i] = 10*np.exp(-np.abs(i-optimal_state)*reward_decay)

        #self.R[:,optimal_state,:] = 1 
        self.density = density

def create_randomized_mdps(N_states: int, N_actions: int, gamma: float, r_seed: int = 1234, reward_decay: float = 10.0):
    """
    This method create a list of random MDP starting from the same 
    deterministic structure and adding variability in the transition
    probabilities and rewards.
   
    Parameters:
    -----------
    N_states: int
        Number of states in the MDP (even integer)
    N_actions: int
        Number of actions in the MDP
    gamma: float
        Discount factor
    
    Returns:
    --------
    mdp_list: list
        List of random MDPs
    """
    # set the seed to do the same experiments with different communication costs
    np.random.seed(r_seed)

    # create a list of MDPs
    mdp_list = [MDP(N_states, N_actions, gamma, 1/N_states, reward_decay) for _ in range(int(N_states/2))]

    # from the first MDP, create the other MDPs by adding links between states
    for a in range(N_actions):
        for s in range(N_states):
            mdp_list[0].P[a][s] = np.zeros(N_states)
            random_idx = np.random.randint(N_states)
            mdp_list[0].P[a][s][random_idx] = 1

            for i in range(1, int(N_states/2)):
                new_link_set = [(random_idx + j) % N_states for j in range(-i, i+1)]
                mdp_list[i].P[a][s] = np.zeros(N_states)
                for idx in new_link_set:
                    mdp_list[i].P[a][s][idx] = 1/len(new_link_set)

                mdp_list[i].P[a,s,random_idx] += 0.1
                mdp_list[i].P[a,s,:] /= np.sum(mdp_list[i].P[a,s,:])

        for i, m in enumerate(mdp_list):
            m.R = mdp_list[0].R 
            m.density = (2*i+1)/N_states
    return mdp_list
