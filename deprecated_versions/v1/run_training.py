"""
Autoregressive PPO training for hospital robot task assignment.
"""
import torch
import numpy as np
from pathlib import Path
from src.environment.autoregressive_env import AutoregressiveTaskAssignmentEnv
from src.multi_agent_ppo.autoregressive_ppo import AutoregressivePPO
from src.multi_agent_ppo.memory import Memory
from src.environment.autoregressive_state_builder import get_state_dim


def main():
    ############## Hyperparameters ##############
    num_robots = 5
    num_nodes = 10

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/autoregressive_ppo")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = AutoregressiveTaskAssignmentEnv(
        num_robots=num_robots,
        num_nodes=num_nodes
    )

    # Get state dimension
    state_dim = get_state_dim()  # 171 features
    action_dim = num_robots + 1  # robots + HOLD

    # PPO hyperparameters
    lr = 0.0003
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2

    # Training hyperparameters
    max_episodes = 500
    log_interval = 10
    save_interval = 50

    #############################################

    # Create autoregressive PPO agent
    ppo = AutoregressivePPO(
        state_dim=state_dim,
        num_robots=num_robots,
        lr=lr,
        betas=betas,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip
    )

    # Memory buffer
    memory = Memory()

    # Training loop
    print("=" * 80)
    print("Starting Autoregressive PPO Training")
    print("=" * 80)
    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim} (robots: {num_robots}, HOLD: 1)")
    print(f"Episodes: {max_episodes}")
    print("-" * 80)

    for i_episode in range(1, max_episodes + 1):
        # Reset environment
        state = env.reset()

        episode_reward = 0.0
        num_assignments = 0
        num_holds = 0

        # Autoregressive assignment loop
        done = False
        while not done:
            # Select action
            action = ppo.select_action(state, memory)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Track statistics
            if action == env.HOLD_ACTION:
                num_holds += 1
            else:
                num_assignments += 1

            # Store reward and terminal flag
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            episode_reward += reward
            state = next_state

        # Update PPO policy
        ppo.update(memory)
        memory.clear_memory()

        # Logging
        if i_episode % log_interval == 0:
            print(f"Episode {i_episode:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Assigned: {num_assignments:2d} | "
                  f"Held: {num_holds:2d} | "
                  f"Total Tasks: {num_assignments + num_holds:2d}")

        # Detailed logging
        if i_episode % (log_interval * 5) == 0:
            print("-" * 80)
            print(f"Episode {i_episode} - Detailed Stats:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Tasks Assigned: {num_assignments}")
            print(f"  Tasks Deferred (HOLD): {num_holds}")
            print(f"  Assignment Rate: {num_assignments / max(num_assignments + num_holds, 1) * 100:.1f}%")
            print("-" * 80)

        # Save model
        if i_episode % save_interval == 0:
            save_path = checkpoint_dir / f"ppo_autoregressive_ep{i_episode}.pth"
            ppo.save(str(save_path))
            print(f"Model saved to {save_path}")

    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)

    # Save final model
    final_model_path = checkpoint_dir / "ppo_autoregressive_final.pth"
    ppo.save(str(final_model_path))
    print(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    main()
