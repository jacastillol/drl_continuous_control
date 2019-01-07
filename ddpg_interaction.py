import time
import numpy as np
from collections import deque

def info(env):
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # examine
    # reset the env
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents
    num_agents =  len(env_info.agents)
    # size of each action
    action_size =  brain.vector_action_space_size
    # examine the state space
    states =  env_info.vector_observations
    state_size =  states.shape[1]
    print('Number of agents:', num_agents)
    print('Size of each action:', action_size)
    print('There are {} agents. Eachobserves a state with length: {}'.
          format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    return num_agents, state_size, action_size

def reset(env,train_mode=True):
    """ Performs an Environment step with a particular action.

    Params
    ======
        env: instance of UnityEnvironment class
    """
    # get the default brain
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=train_mode)[brain_name]
    # get state
    states = env_info.vector_observations
    return states

def step(env, actions):
    """ Performs an Environment step with a particular action.

    Params
    ======
        env: instance of UnityEnvironment class
        action: a valid action on the env
    """
    # get the default brain
    brain_name = env.brain_names[0]
    # perform the step
    env_info = env.step(actions)[brain_name]
    # get result from taken action
    next_states = env_info.vector_observations
    rewards = env_info.rewards
    dones = env_info.local_done
    return next_states, rewards, dones

def ddpg(env, agent, n_episodes=300, max_t=700, print_every=10):
    """ Deep Deterministic Policy Gradient algorithm

    Params:
    =======

    """
    first_time=True
    scores_deque = deque(maxlen=100)
    scores_avg = deque(maxlen=n_episodes)
    scores = []
    max_score = -np.Inf
    # init timer
    tic = time.clock()
    # for each episode
    for i_episode in range(1, n_episodes+1):
        state = reset(env, train_mode=True)
        action = agent.act(state)
        # agent.reset()
        score = np.zeros(agent.num_agents)
        for t in range(max_t):
            #action = agent.act(state)
            next_state, reward, done = step(env, action)
            #agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        score = np.mean(score)
        scores_deque.append(score)
        scores.append(score)
        # geting averages
        # append to average
        curr_avg_score = np.mean(scores_deque)
        scores_avg.append(curr_avg_score)
        # update best average reward
        if curr_avg_score > max_score:
            max_score = curr_avg_score
        # monitor progress
        message = "\rEpisode {}/{} || Best average reward {}"
        if i_episode % print_every == 0:
            print(message.format(i_episode, n_episodes, max_score))
        else:
            print(message.format(i_episode, n_episodes, max_score),end="")
        # target criteria
        if np.mean(scores_deque)>=30.0 and first_time:
            first_time=False
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tin {:.2f} secs'.
                  format(i_episode, np.mean(scores_deque), time.clock()-tic))
        # stopping criteria
        if np.mean(scores_deque)>=35.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tin {:.2f} secs'.
                  format(i_episode, np.mean(scores_deque), time.clock()-tic))
            break

    return scores, scores_avg
