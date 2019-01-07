
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

