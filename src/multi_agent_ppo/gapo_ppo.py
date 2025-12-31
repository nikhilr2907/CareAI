"""
GAPO PPO Trainer with de-biasing.

Implements PPO training for GAPO policy network with:
- Graph-structured states
- De-biasing loss
- Attention visualization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import numpy as np

from .gapo_policy import GAPOPolicyNetwork


class Memory:
    """Memory buffer for PPO."""

    def __init__(self):
        self.state_dicts = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.robot_masks = []

    def clear_memory(self):
        del self.state_dicts[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.robot_masks[:]


class GAPOPPO:
    """
    PPO trainer for GAPO policy network.
    """

    def __init__(
        self,
        node_feat_dim=5,
        edge_feat_dim=3,
        robot_feat_dim=12,
        task_feat_dim=12,
        hidden_dim=64,
        lr=0.0003,
        gamma=0.99,
        K_epochs=4,
        eps_clip=0.2,
        use_gnn=True,
        use_debiasing=True,
        lambda_debias=0.1,
        device='cpu'
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.use_debiasing = use_debiasing
        self.device = torch.device(device)

        # GAPO policy network
        self.policy = GAPOPolicyNetwork(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            robot_feat_dim=robot_feat_dim,
            task_feat_dim=task_feat_dim,
            hidden_dim=hidden_dim,
            use_gnn=use_gnn,
            use_debiasing=use_debiasing,
            lambda_state_delta=lambda_debias
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Old policy for PPO ratio
        self.policy_old = GAPOPolicyNetwork(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            robot_feat_dim=robot_feat_dim,
            task_feat_dim=task_feat_dim,
            hidden_dim=hidden_dim,
            use_gnn=use_gnn,
            use_debiasing=False  # Don't need debiasing in old policy
        ).to(self.device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(
        self,
        state_dict: Dict[str, np.ndarray],
        memory: Memory,
        robot_mask: np.ndarray = None
    ) -> int:
        """
        Select action using current policy.

        Args:
            state_dict: State dictionary from environment
            memory: Memory buffer to store experience
            robot_mask: Boolean mask for available robots

        Returns:
            action: Selected action
        """
        # Convert numpy arrays to tensors
        state_dict_tensor = self._state_dict_to_tensor(state_dict)

        # Convert mask
        if robot_mask is not None:
            robot_mask_tensor = torch.tensor(robot_mask, dtype=torch.bool).to(self.device)
        else:
            robot_mask_tensor = None

        # Select action
        with torch.no_grad():
            action, log_prob = self.policy_old.select_action(
                state_dict_tensor,
                robot_mask_tensor
            )

        # Store in memory
        memory.state_dicts.append(state_dict)
        memory.actions.append(action)
        memory.logprobs.append(log_prob.item())
        memory.robot_masks.append(robot_mask)

        # Record action for de-biasing
        self.policy.record_action(action)

        return action

    def update(self, memory: Memory):
        """
        Update policy using PPO with de-biasing.

        Args:
            memory: Memory buffer with experiences
        """
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert to tensors
        old_actions = torch.tensor(memory.actions, dtype=torch.long).to(self.device)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(self.device)

        # Optimize policy for K epochs
        for epoch in range(self.K_epochs):
            # Convert state dicts to tensors
            state_dict_tensors = [
                self._state_dict_to_tensor(sd) for sd in memory.state_dicts
            ]

            robot_mask_tensors = []
            for mask in memory.robot_masks:
                if mask is not None:
                    robot_mask_tensors.append(
                        torch.tensor(mask, dtype=torch.bool).to(self.device)
                    )
                else:
                    robot_mask_tensors.append(None)

            # Evaluate actions
            logprobs, state_values, dist_entropy = self.policy.evaluate_actions(
                state_dict_tensors,
                old_actions,
                robot_mask_tensors
            )

            # PPO ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Advantages
            advantages = rewards - state_values.detach()

            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = 0.5 * self.MseLoss(state_values, rewards)

            # Entropy bonus (exploration)
            entropy_loss = -0.01 * dist_entropy.mean()

            # De-biasing loss
            if self.use_debiasing:
                debias_loss, debias_breakdown = self.policy.compute_debias_loss()
            else:
                debias_loss = torch.tensor(0.0)
                debias_breakdown = {}

            # Total loss
            loss = actor_loss + critic_loss + entropy_loss + debias_loss

            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            # Log losses (first epoch only)
            if epoch == 0:
                self.last_loss_info = {
                    'total_loss': loss.item(),
                    'actor_loss': actor_loss.item(),
                    'critic_loss': critic_loss.item(),
                    'entropy_loss': entropy_loss.item(),
                    'debias_loss': debias_loss.item() if self.use_debiasing else 0.0,
                    **debias_breakdown
                }

        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Reset episode tracking
        self.policy.reset_episode_tracking()

    def _state_dict_to_tensor(self, state_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert numpy state dict to tensor dict."""
        tensor_dict = {}

        for key, value in state_dict.items():
            if isinstance(value, np.ndarray):
                tensor_dict[key] = torch.tensor(value, dtype=torch.float32).to(self.device)
            else:
                tensor_dict[key] = value

        return tensor_dict

    def save(self, filepath: str):
        """Save policy network."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """Load policy network."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def get_last_loss_info(self) -> Dict:
        """Get last training loss breakdown."""
        return getattr(self, 'last_loss_info', {})


def test_gapo_ppo():
    """Test GAPO PPO trainer."""
    print("Testing GAPO PPO Trainer...")

    from src.environment.gapo_env import GAPOTaskAssignmentEnv

    # Create environment
    env = GAPOTaskAssignmentEnv(num_robots=5, num_nodes=10)

    # Create PPO
    ppo = GAPOPPO(
        node_feat_dim=5,
        edge_feat_dim=3,
        robot_feat_dim=12,
        task_feat_dim=12,
        hidden_dim=64,
        lr=0.0003,
        use_gnn=True,
        use_debiasing=True
    )

    # Create memory
    memory = Memory()

    # Run one episode
    state_dict = env.reset()

    for step in range(10):
        mask = env.get_action_mask()
        action = ppo.select_action(state_dict, memory, mask)

        next_state_dict, reward, done, info = env.step(action)

        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        state_dict = next_state_dict

        if done:
            break

    # Update policy
    print("\nUpdating policy...")
    ppo.update(memory)

    loss_info = ppo.get_last_loss_info()
    print("\nLoss breakdown:")
    for key, value in loss_info.items():
        print(f"  {key}: {value:.4f}")

    print("\n✓ GAPO PPO working!")


if __name__ == '__main__':
    test_gapo_ppo()
