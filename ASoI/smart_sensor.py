import numpy as np
from ASoI.mdp import MDP
from ASoI.RL_algorithm import RL_Algorithm

class RemotePolicyIteration(RL_Algorithm):
    """
    Class that implements the policy iteration algorithm where the state is 
    communicated by a remote sensor with a cost. 

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
    """
    def __init__(self, mdp: MDP, horizon: int, t_max: int):
        self.mdp = mdp
        self.H = horizon
        self.V_sensor = np.zeros((self.mdp.N_s, self.H, self.mdp.N_s))
        self.pi_sensor = np.random.randint(0,2,(self.mdp.N_s, self.H, self.mdp.N_s))

        for s in range(self.mdp.N_s):
            self.pi_sensor[:,:,s] = 0

        self.V_actuator = np.zeros((self.mdp.N_s, self.H))
        self.pi_actuator = np.random.randint(0,self.mdp.N_a,(self.mdp.N_s, self.H))

        self.t_max = t_max

        self.gsr = np.sqrt(self.mdp.gamma) # gamma square root

    def run(self, beta):
        policy_stable = False
        max_iter = 100
        i = 0
        while not policy_stable:
            self.policy_evaluation(beta)
            policy_stable = self.policy_improvement(beta)
            print(self.eval_perf(100))
            print(np.mean(self.V_actuator[:,0]))
            i += 1
            if i == max_iter:
                break

    def policy_evaluation(self, beta):
        delta = 1
        while delta > 1e-3:
            delta = 0
            for s in range(self.mdp.N_s):
                belief = np.zeros(self.mdp.N_s)
                belief[s] = 1

                for t in range(self.H):
                    old_value = self.V_actuator[s,t]
                    action = self.pi_actuator[s,t]
                    belief_after_action = np.dot(belief, self.mdp.P[action])
                    immediate_reward = np.dot(belief, self.mdp.R[action])

                    R = np.dot(belief_after_action, immediate_reward)
                    self.V_actuator[s,t] = R + self.gsr*(np.dot(self.V_sensor[s,t,:],belief_after_action))
                    delta = max(delta, np.abs(old_value - self.V_actuator[s,t]))

                    for s_curr in range(self.mdp.N_s):
                        c = self.pi_sensor[s,t,s_curr]
                        if c == 0:
                            old_value_sensor = self.V_sensor[s,t,s_curr]

                            if t == self.H-1:
                                self.V_sensor[s,t,s_curr] = -beta + self.gsr*self.V_actuator[s_curr,0]
                                delta = max(delta, np.abs(old_value_sensor - self.V_sensor[s,t,s_curr]))
                            else:
                                #UPDATE
                                self.V_sensor[s,t,s_curr] = self.gsr*self.V_actuator[s,t+1]
                                delta = max(delta, np.abs(old_value_sensor - self.V_sensor[s,t,s_curr]))

                        else:
                            old_value_sensor = self.V_sensor[s,t,s_curr]

                            #UPDATE
                            self.V_sensor[s,t,s_curr] = -beta + self.gsr*self.V_actuator[s_curr,0]
                            delta = max(delta, np.abs(old_value_sensor - self.V_sensor[s,t,s_curr]))

                    belief = belief_after_action

    def policy_improvement(self, beta):
        policy_stable = True
        for s in range(self.mdp.N_s):
            belief = np.zeros(self.mdp.N_s)
            belief[s] = 1
            for t in range(self.t_max):
                # Update the policy for the MDP action
                old_action = self.pi_actuator[s,t]
                values = np.zeros(self.mdp.N_a)
                for action in range(self.mdp.N_a):
                    belief_after_action = np.dot(belief, self.mdp.P[action])
                    immediate_reward = np.dot(np.dot(belief, self.mdp.R[action]), belief_after_action)
                    value = 0
                    for s_curr in range(self.mdp.N_s):
                        c = self.pi_sensor[s,t,s_curr]
                        if c == 1:
                            value += belief_after_action[s_curr]*(immediate_reward - self.gsr*beta + self.mdp.gamma*self.V_actuator[s_curr,0])
                        else:
                            reward_simulated = self.V_sensor[s,t,s_curr] # self.simulate(s,t,s_curr,beta, t)
                            value += belief_after_action[s_curr]*(immediate_reward + self.mdp.gamma*reward_simulated)
                    values[action] = value
                    
                self.pi_actuator[s,t] = np.argmax(values)
                if old_action != self.pi_actuator[s,t]:
                    policy_stable = False

                # Update the policy of the sensor
                for s_curr in range(self.mdp.N_s):
                    old_c = self.pi_sensor[s,t,s_curr]

                    if t == self.t_max-1:
                        self.pi_sensor[s,t,s_curr] = 1
                    else:
                        values = np.zeros(2)
                        values[0] = self.gsr*self.V_actuator[s,t+1]
                        values[1] = -beta + self.gsr*self.V_actuator[s_curr,0]
                        self.pi_sensor[s,t,s_curr] = np.argmax(values)
                        if np.abs(values[0] - values[1]) < 1e-3:
                            self.pi_sensor[s,t,s_curr] = 1

                    if old_c != self.pi_sensor[s,t,s_curr]:
                        policy_stable = False

                belief = np.dot(belief, self.mdp.P[self.pi_actuator[s,t]])
        return policy_stable

    def tbaa(self,b,s,t):
        for s_curr in range(self.mdp.N_s):
            if self.pi_sensor[s,t,s_curr] == 1:
                b[s_curr] = 0
        return b/np.sum(b)

    def simulate(self, s, t, s_curr, beta, t_0):
        reward = 0
        if t == self.t_max-1:
            return -beta + self.gsr*self.V_actuator[s_curr,0]
        else:
            action = self.pi_actuator[s,t+1]
            
    
    def eval_perf(self, n_episodes):
        tot_r = 0
        tot_c = 0
        for _ in range(n_episodes):
            s = np.random.randint(self.mdp.N_s)
            s_last = s
            t = 0
            for h in range(self.H):
                action = self.pi_actuator[s_last,t]
                s_next = np.random.choice(self.mdp.N_s, p=self.mdp.P[action,s])
                tot_r += self.mdp.R[action,s,s_next]

                c_t = self.pi_sensor[s_last,t,s_next]
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
                action = self.pi_actuator[s_last,t]
                s_next = np.random.choice(self.mdp.N_s, p=self.mdp.P[action,s])
                tot_r += self.mdp.R[action,s,s_next]

                c_t = self.pi_sensor[s_last,t,s_next]
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