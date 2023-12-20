# This script compute the PeakAOI for a given state, density, and beta
from utils import load_json, load_and_save_results, load_agent, get_peakAOI_dist
import argparse

parser = argparse.ArgumentParser(description='Compute PeakAOI for a given state, density, and beta')
parser.add_argument('--d_idx', type=int, default=1, help='Density to compute PeakAOI for')
parser.add_argument('--density', type=str, default='0.1', help='Density to compute PeakAOI for')
parser.add_argument('--beta', type=str, default='0.0', help='Beta to compute PeakAOI for')
parser.add_argument('--reward', type=str, default='sparse', help='Type of reward to use')

args = parser.parse_args()
params = load_json('params/push_'+args.reward+'_reward.json')
agent_push = load_agent(args, params)

params = load_json('params/pull_'+args.reward+'_reward.json')
agent_pull = load_agent(args, params)

get_peakAOI_dist(agent_push, agent_pull, args.density, args.beta, reward=args.reward)