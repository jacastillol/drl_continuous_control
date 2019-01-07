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
