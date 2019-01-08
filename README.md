# Deep Reinforcement Learning for Continuous Control
A robotic arm learns to reach with Deep Reinforcement Learning using __Unity ML-Agents__ plugin 

## Project Details
__Unity__ has an environment of a double-jointed arm with the goal of following a target, this environment is call the [__Reacher__](https://youtu.be/2N9EoF6pQyE). It's considered that the target at each time is reached if the end-effector is under a critial radio represented by a big sphere. In the animation this sphere changes its color to a light green when the center of the end effector is inside that big sphere.
* __Reward signal__:
  A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
* __Observation space__:
  The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 
* __Action space__:
  Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

To speed up training one controller will be used to control 20 arms at the same time. Finally, the environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

## Getting Started
### Dependencies and Installation
1. Clone this repo `git clone https://github.com/jacastillol/drl_continuous_control.git drl_continuous_control`
1. Create conda environment and install dependencies:
  ```bash
  conda crate --name drlcontinuous python=3.6
  source activate drlcontinuous
  pip install unityagents torch torchsummary
  ```
1. Download and intall Unity environment
  * Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
  * Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
  * Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
  
  Then, place the file in the `drl_continuous/` folder of this GitHub repository, and unzip the file.

## Instructions
### Configuration file `params.ini`
To run the main program you have to create a configuration file called `params.ini` where you can set all important parameters from the networks architectures, learning Adam algorithms, replay memory, and others. The repository has an example file `params_example.ini` that you can copy and rename. Here is an example of a configuration file, 

```
[DEFAULT]
# max. number of episode to train the agent
n_episodes:       200
# max. number of steps per episode
max_t:           1000
# save the last XXX returns of the agent
print_every:        1
# replay buffer size
SEED:               0
# replay buffer size
BUFFER_SIZE:      1e5
# minibatch size
BATCH_SIZE:        64
# how often to update the network
UPDATE_EVERY:       4
# discount factor
GAMMA:           0.99
# std noise over actions for exploration
SIGMA:           0.20
# for soft update or target parameters
TAU:             1e-3
# learning rate of the actor
LR_ACTOR:        1e-4
# learning rate of the critic
LR_CRITIC:       3e-4
# L2 weight decay
WEIGHT_DECAY:  0.0001
# number of neurons in actor first layer
FC_ACTOR:          32
# number of neurons in critic first layer
FC1_CRITIC:        32
# number of neurons in critic second layer
FC2_CRITIC:        32
```
### How to run the code
1. Create a config file. One way could be `cp params_example.ini params.ini` and then modify the parameters as you want
1. Remember to activate the environment with `source activate drlnavigation`
1. To train one-arm new agent or twenty-arm new agent:
  
    ```python learn_and_prove.py --train```
  
    ```python learn_and_prove.py --multi-agent --train```
  
    This will produce three files under the namespace `checkpoint`: `checkpoint.actor.pth` and `checkpoint.critic.pth` holding the weights of the final Actor and Critic networks. The third file `checkpoint.npz` contains information about the configuration run and learning curves. To change the default namespace use the option `--file NAMESPACE`.
1. To watch again the performance of the agent trained in the last step run again:

    ```python learn_and_prove.py [--multi-agent] [--file NAMESPACE]```

