"""
Autoregressive PPO for task assignment.
Single policy that assigns tasks to robots one-by-one.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


class AutoregressiveActorCritic(nn.Module):
    """
    Actor-Critic network for autoregressive task assignment.

    Actor outputs: [robot_0_prob, robot_1_prob, ..., robot_N_prob, HOLD_prob]
    Critic outputs: value estimate of state
    """

    def __init__(self, state_dim, num_robots):
        super(AutoregressiveActorCritic, self).__init__()

        self.num_robots = num_robots
        self.action_dim = num_robots + 1  # robots + HOLD

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(256, self.action_dim)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self):
        raise NotImplementedError("Use act() or evaluate() instead")

    def act(self, state):
        """
        Select action from current policy.

        Args:
            state: numpy array or torch tensor

        Returns:
            action, action_logprob
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)

        # Encode state
        encoded = self.encoder(state)

        # Get action probabilities
        action_logits = self.actor(encoded)
        action_probs = torch.softmax(action_logits, dim=-1)

        # Sample action
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob

    def evaluate(self, states, actions):
        """
        Evaluate actions under current policy.

        Args:
            states: batch of states
            actions: batch of actions

        Returns:
            action_logprobs, state_values, dist_entropy
        """
        # Encode states
        encoded = self.encoder(states)

        # Actor evaluation
        action_logits = self.actor(encoded)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        # Critic evaluation
        state_values = self.critic(encoded).squeeze()

        return action_logprobs, state_values, dist_entropy


class AutoregressivePPO:
    """
    PPO for autoregressive task assignment.
    Processes tasks one-by-one, deciding robot assignment or HOLD.
    """

    def __init__(
        self,
        state_dim,
        num_robots,
        lr=0.0003,
        betas=(0.9, 0.999),
        gamma=0.99,
        K_epochs=4,
        eps_clip=0.2
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # Create policy networks
        self.policy = AutoregressiveActorCritic(state_dim, num_robots)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        # Old policy for PPO
        self.policy_old = AutoregressiveActorCritic(state_dim, num_robots)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        """
        Select action using old policy and store in memory.

        Args:
            state: Current state (numpy array)
            memory: Memory object to store experience

        Returns:
            action (int)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, action_logprob = self.policy_old.act(state_tensor)

        # Store in memory
        memory.states.append(state_tensor)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.item()

    def update(self, memory):
        """
        Update policy using collected experience.

        Args:
            memory: Memory object with states, actions, rewards, etc.
        """
        # Convert lists to tensors
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach()

        # Compute Monte Carlo returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions with current policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Importance sampling ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Compute advantages
            advantages = rewards - state_values.detach()

            # PPO clipped surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Total loss
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # Gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
