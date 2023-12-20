# Abstract class for RL algorithms

class RL_Algorithm():
    def __init__(self, MDP, horizon, t_max, **kwargs):
        self.mdp = MDP
        self.H = horizon
        self.t_max = t_max

    def run(self, beta):
        pass

    def eval_perf(self, n_episodes):
        pass

    def get_PeakAoI(self, s, n_episodes):
        pass