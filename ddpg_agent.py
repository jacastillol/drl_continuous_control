import numpy as np

class Agent:
    def __init__(self, num_agents, state_size, action_size):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size

    def reset(self):
        pass

    def act(self, state):
        action = np.random.randn(self.num_agents, self.action_size)
        action = np.clip(action, -1, 1)
        return action

    def step(self, state, action, reward, next_state, done):
        pass
