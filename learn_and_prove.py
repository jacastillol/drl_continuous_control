# conda create --name drlcontinuous python=3.6
# source activate drlcontinuous
# pip install unityagents
# --> install Reacher for single and multiagent
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ddpg_interaction import info, reset, step, ddpg
from ddpg_agent import Agent

parser = argparse.ArgumentParser()
# input argument for single or multiple arms agents
parser.add_argument('--multi-agent', action='store_true',
                    help='To run 20 arms instead of 1')
# input argument to display learning curves
parser.add_argument('--display', action='store_true',
                    help='Display evolution of the scores')
args = parser.parse_args()

# current configuration
config = {
    'n_episodes':       300,   # max. number of episode to train the agent
    'max_t':            700,   # max. number of steps per episode
    'print_every':       10,   # save the last XXX returns of the agent
#    'eps_start':        1.0,  # GLIE parameters
#    'eps_end':        0.005,  #
#    'eps_decay':      0.960,  #
#    'BUFFER_SIZE': int(1e5),  # replay buffer size
#    'BATCH_SIZE':        16,  # minibatch size
#    'GAMMA':           0.99,  # discount factor
#    'TAU':             1e-3,  # for soft update or target parameters
#    'LR':              5e-4,  # learning rate
#    'UPDATE_EVERY':       1,  # how often to update the network
#    'FC1_UNITS':         16,  # number of neurons in fisrt layer
#    'FC2_UNITS':         16,  # number of neurons in second layer
}
# print configuration
print(' Config Parameters')
for k,v in config.items():
    print('{:<15}: {:>15}'.format(k,v))

# create environment
if args.multi_agent:
    env = UnityEnvironment(file_name='Reacher_Linux_20/Reacher.x86_64')
else:
    env = UnityEnvironment(file_name='Reacher_Linux_1/Reacher.x86_64')
# get info of the environment
num_agents, state_size, action_size = info(env)
# create an agent
agent = Agent(num_agents=num_agents, state_size=state_size, action_size=action_size)
#
scores, scores_avg = ddpg(env, agent,
                          n_episodes=config['n_episodes'],
                          max_t=config['max_t'],
                          print_every=config['print_every'])
# plot learning curves output
# display mode
if args.display:
    # to continue ...
    print('Press [Q] on the plot window to continue ...')
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(scores_avg)), scores_avg)
    plt.ylabel('Scores')
    plt.xlabel('Episode #')
    plt.show()
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
