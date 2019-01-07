# conda create --name drlcontinuous python=3.6
# source activate drlcontinuous
# pip install unityagents
# --> install Reacher for single and multiagent
from unityagents import UnityEnvironment
import numpy as np
import argparse
from ddpg_interaction import reset, step

parser = argparse.ArgumentParser()
# input argument for single or multiple arms agents
parser.add_argument('--multi-agent', action='store_true',
                    help='To run 20 arms instead of 1')
args = parser.parse_args()

# create environment
if args.multi_agent:
    env = UnityEnvironment(file_name='Reacher_Linux_20/Reacher.x86_64')
else:
    env = UnityEnvironment(file_name='Reacher_Linux_1/Reacher.x86_64')
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# examine
# reset the env
env_info = env.reset(train_mode=True)[brain_name]
# number of agents
num_agents =  len(env_info.agents)
print('Number of agents:', num_agents)
# size of each action
action_size =  brain.vector_action_space_size
print('Size of each action:', action_size)
# examine the state space
states =  env_info.vector_observations
state_size =  states.shape[1]
print('There are {} agents. Eachobserves a state with length: {}'.
      format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
# run a random controller through arms
reset(env, train_mode=False)
scores = np.zeros(num_agents)
while True:
    actions = np.random.randn(num_agents, action_size)
    actions = np.clip(actions, -1, 1)
    next_states, rewards, dones = step(env, actions)
    scores += rewards
    states =  next_states
    if np.any(dones):
        break
print('Total score (averaged over agents) this episode: {}'.
      format(np.mean(scores)))
env.close()
