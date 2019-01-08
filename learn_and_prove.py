# conda create --name drlcontinuous python=3.6
# source activate drlcontinuous
# pip install unityagents torch torchsummary
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
parser.add_argument('--file',
                    help='filename of the trained weights')
parser.add_argument('--train', action='store_true',
                    help='run a pre-trainded neural network agent')
parser.add_argument('--random', action='store_true',
                    help='run a tabula rasa agent')
args = parser.parse_args()

# setting filename
if args.file==None:
    filename='checkpoint.pth'
else:
    filename=args.file

# current configuration
config = {
    'n_episodes':      1000,   # max. number of episode to train the agent
    'max_t':           1000,   # max. number of steps per episode
    'print_every':       10,   # save the last XXX returns of the agent
    'SEED':               0,   # replay buffer size
    'LR_ACTOR':        1e-4,   # learning rate of the actor 
    'LR_CRITIC':       3e-4,   # learning rate of the critic
    'WEIGHT_DECAY':  0.0001,   # L2 weight decay
    'BUFFER_SIZE': int(1e5),   # replay buffer size
    'BATCH_SIZE':       128,   # minibatch size
    'UPDATE_EVERY':       2,   # how often to update the network
    'GAMMA':           0.99,   # discount factor
    'TAU':             1e-3,   # for soft update or target parameters
    'FC_ACTOR':          32,   # number of neurons in actor layer
    'FC1_CRITIC':        32,  # number of neurons in critic layer
    'FC2_CRITIC':        16,  # number of neurons in critic layer
}
# print configuration
print(' Config Parameters')
for k,v in config.items():
    print('{:<15}: {:>15}'.format(k,v))

# create environment
if args.multi_agent:
    env = UnityEnvironment(file_name='Reacher_Linux_20/Reacher.x86_64', seed=config['SEED'])
else:
    env = UnityEnvironment(file_name='Reacher_Linux_1/Reacher.x86_64', seed=config['SEED'])
# get info of the environment
num_agents, state_size, action_size = info(env)
# create an agent
agent = Agent(num_agents=num_agents, state_size=state_size, action_size=action_size,
              random_seed=config['SEED'],
              gamma=config['GAMMA'],
              tau=config['TAU'],
              lr_actor=config['LR_ACTOR'],
              lr_critic=config['LR_CRITIC'],
              weight_decay=config['WEIGHT_DECAY'],
              fc_a=config['FC_ACTOR'],
              fc1_c=config['FC1_CRITIC'],
              fc2_c=config['FC2_CRITIC'],
              buffer_size=config['BUFFER_SIZE'],
              batch_size=config['BATCH_SIZE'],
              update_every=config['UPDATE_EVERY'])
#
if args.train:
    scores, scores_avg = ddpg(env, agent,
                              n_episodes=config['n_episodes'],
                              max_t=config['max_t'],
                              print_every=config['print_every'],
                              filename=filename)
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
elif args.random:
    # choose random policy
    print('Run a Tabula Rasa or random agent')
else:
    # load the weights from file
    agent.actor_local.load_state_dict(torch.load(filename+'.actor.pth'))
    agent.actor_target.load_state_dict(torch.load(filename+'.actor.pth'))
    agent.critic_local.load_state_dict(torch.load(filename+'.critic.pth'))
    agent.critic_target.load_state_dict(torch.load(filename+'.critic.pth'))
    print('Loaded {}:'.format(filename))
# run a random controller through arms
for i in range(10):
    scores = np.zeros(num_agents)
    states = reset(env, train_mode=False)
    # check performance of the agent
    for j in range(1000):
        actions = agent.act(states, add_noise=False)
        next_states, rewards, dones = step(env, actions)
        states =  next_states
        scores += rewards
        if np.any(dones):
            break
    print('Total score (averaged over agents) this episode: {}'.
          format(np.mean(scores)))
env.close()
