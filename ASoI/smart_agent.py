import numpy as np
from ASoI.mdp import MDP
from ASoI.RL_algorithm import RL_Algorithm

class PolicyIterationWithSampling(RL_Algorithm):
    """
    Class that implements the policy iteration algorithm where sampling
    of the state comes with a cost. 

    Parameters:
    -----------
    mdp: MDP
        MDP object
    horizon: int
        Horizon of the problem
    t_max: int
        Maximum number of consecutive actions before sensing

    Attributes:
    -----------
    mdp: MDP
        MDP object
    H: int
        Horizon of the problem
    V: np.array
        Value function
    pi: np.array
        Policy
    t_max: int
        Maximum number of consecutive actions before sensing

    Methods:
    --------
    run(beta):
        Run the policy iteration algorithm
    policy_evaluation(beta):
        Evaluate the policy
    policy_improvement(beta):
        Improve the policy
    simulate(s,t,belief,beta):
        Simulate the reward for a given state and belief
    eval_perf(n_episodes):
        Evaluate the performance of the policy

    """
    def __init__(self, mdp, horizon, t_max):
        self.mdp = mdp
        self.H = horizon
        self.V = np.random.rand(self.mdp.N_s,t_max,2) #np.zeros((self.mdp.N_s, t_max, 2))
        self.pi = np.random.randint(0,self.mdp.N_a, (self.mdp.N_s, t_max, 2))
        self.pi[:,:,1] = np.random.randint(0,2, (self.mdp.N_s, t_max))
        self.pi[:,-1,1] = 1
        self.t_max = t_max
        self.gamma_square_root = self.mdp.gamma**(1/2)

    def run(self, beta):
        policy_stable = False
        max_iter = 100
        i = 0
        while not policy_stable:
            self.policy_evaluation(beta)
            policy_stable = self.policy_improvement(beta)
            print('Reward and cost: ', self.eval_perf(100))
            print(np.mean(self.V[:,0,0]))
            i += 1
            if i == max_iter:
                break

    def policy_evaluation(self, beta):
        delta = 1
        theta = 1e-5
        while delta > theta:
            delta = 0
            for s in range(self.mdp.N_s):
                belief = np.zeros(self.mdp.N_s)
                belief[s] = 1
                for t in range(self.t_max):
                    # Update the value function for the MDP action
                    v = self.V[s,t,0]
                    action = self.pi[s,t,0]
                    belief_after_action = np.dot(belief,self.mdp.P[action])
                    immediate_reward = np.dot(belief_after_action, np.dot(belief,self.mdp.R[action]))
                    self.V[s,t,0] = immediate_reward + self.gamma_square_root * self.V[s,t,1]
                    delta = max(delta, abs(v - self.V[s,t,0]))
                    # Update the value function for the sensing action
                    v = self.V[s,t,1]
                    belief = belief_after_action
                    c = self.pi[s,t,1]
                    if t == self.t_max-1:
                        self.V[s,t,1] = -beta + self.gamma_square_root*(np.dot(self.V[:,0,0],belief))
                        delta = max(delta, abs(v - self.V[s,t,1]))
                    else:
                        if c == 1:
                            self.V[s,t,1] = -beta + self.gamma_square_root*(np.dot(self.V[:,0,0],belief))
                            delta = max(delta, abs(v - self.V[s,t,1]))
                        else:
                            self.V[s,t,1] = self.gamma_square_root*self.V[s,t+1,0]
                            delta = max(delta, abs(v - self.V[s,t,1]))

    def policy_improvement(self, beta):
        policy_stable = True
        for s in range(self.mdp.N_s):
            belief = np.zeros(self.mdp.N_s)
            belief[s] = 1
            for t in range(self.t_max):
                # Update the policy for the MDP action
                old_action = self.pi[s,t,0]
                values = np.zeros(self.mdp.N_a)
                for action in range(self.mdp.N_a):
                    belief_after_action = np.dot(belief, self.mdp.P[action])
                    immediate_reward = np.dot(np.dot(belief, self.mdp.R[action]), belief_after_action)
                    c = self.pi[s,t,1]
                    if c == 1:
                        value = immediate_reward - self.gamma_square_root*beta + self.mdp.gamma*(belief_after_action.dot(self.V[:,0,0]))# self.gamma_square_root * self.V[s,t,1]
                    else:
                        reward_simulated = self.simulate(s,t,belief_after_action, beta)
                        value = immediate_reward + self.mdp.gamma*reward_simulated
                    values[action] = value
                #values += np.random.normal(0, 1e-8, self.mdp.N_a)

                self.pi[s,t,0] = np.argmax(values)
                if old_action != self.pi[s,t,0]:
                    policy_stable = False

                # Update the policy for the sensing action
                belief = np.dot(belief, self.mdp.P[self.pi[s,t,0]])
                old_c = self.pi[s,t,1]

                if t == self.t_max-1:
                    self.pi[s,t,1] = 1
                else:
                    values = np.zeros(2)
                    values[0] = self.gamma_square_root*self.V[s,t+1,0]
                    values[1] = -beta + self.gamma_square_root*(np.dot(self.V[:,0,0],belief))
                    self.pi[s,t,1] = np.argmax(values)

                if old_c != self.pi[s,t,1]:
                    policy_stable = False
        return policy_stable

    def simulate(self, s, t, belief, beta):
        reward = 0
        action_tau = self.pi[s,t+1,0]
        belief_after_action = np.dot(belief, self.mdp.P[action_tau])
        if self.pi[s,t+1,1] == 0:
            new_r =  np.dot(belief_after_action,np.dot(belief,self.mdp.R[action_tau])) + self.mdp.gamma*self.simulate(s,t+1,belief_after_action,beta)
            reward += new_r
        else: 
            new_r_2 = self.mdp.gamma*(belief_after_action.dot(self.V[:,0,0])) - self.gamma_square_root*beta
            new_r = np.dot(belief_after_action,np.dot(belief,self.mdp.R[action_tau])) + new_r_2
            reward += new_r
        return reward

    def eval_perf(self, n_episodes):
        tot_r = 0
        tot_c = 0
        for _ in range(n_episodes):
            s = np.random.randint(self.mdp.N_s)
            s_last = s
            t = 0
            for h in range(self.H):
                action = self.pi[s_last,t,0]
                s_next = np.random.choice(self.mdp.N_s, p=self.mdp.P[action,s])
                tot_r += self.mdp.R[action,s,s_next]

                c_t = self.pi[s_last,t,1]
                if h != self.H-1: tot_c += c_t
                if c_t == 0:
                    t += 1
                else:
                    t = 0
                    s_last = s_next
                s = s_next
        return tot_r/(n_episodes*self.H), tot_c/(n_episodes*self.H)
    
    def get_PeakAoI(self, target_state: int, n_episodes: int = 1000):
        tot_r = 0
        tot_c = 0
        PAOI = []
        paoi = 0
        for _ in range(n_episodes):
            s = np.random.randint(self.mdp.N_s)
            s_last = s
            t = 0
            if s_last == target_state:
                paoi = 1
            else:
                paoi = 0

            for h in range(self.H):
                action = self.pi[s_last,t,0]
                s_next = np.random.choice(self.mdp.N_s, p=self.mdp.P[action,s])
                tot_r += self.mdp.R[action,s,s_next]

                c_t = self.pi[s_last,t,1]
                if h != self.H-1: tot_c += c_t
                if c_t == 0:
                    t += 1
                else:
                    if paoi == 1: PAOI.append(t+1)
                    t = 0
                    s_last = s_next
                    
                    if s_last == target_state:
                        paoi = 1
                    else:
                        paoi = 0

                s = s_next
        return PAOI