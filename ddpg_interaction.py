
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

