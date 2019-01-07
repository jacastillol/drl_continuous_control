import numpy as np
from collections import namedtuple, deque
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, num_agents, state_size, action_size,
                 gamma=0.99, tau=1e-3,
                 lr_actor=1e-4, lr_critic=3e-4, weight_decay=1e-4,
                 fc_a=32, fc_c=32,
                 buffer_size=int(1e5), batch_size=64):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        #
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        # Actor Network
        self.actor_local = Actor(state_size, action_size, fc_units=fc_a).to(device)
        self.actor_target = Actor(state_size, action_size, fc_units=fc_a).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=lr_actor)
        # Critic Network
        self.critic_local = Critic(state_size, action_size, fc_units=fc_c).to(device)
        self.critic_target = Critic(state_size, action_size, fc_units=fc_c).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=lr_critic, weight_decay=weight_decay)
         # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

    def reset(self):
        pass

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        states = torch.from_numpy(np.vstack([e for e in state if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e for e in action if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e for e in reward if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e for e in next_state if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e for e in done if e is not None]).astype(np.uint8)).float().to(device)
        self.learn(states, actions, rewards, next_states, dones)

    def learn(self, states, actions, rewards, next_states, dones):

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
