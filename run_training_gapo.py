"""
GAPO Training Script with De-biasing.

Trains hospital robot task allocation using:
- Graph Attention-Based Policy Optimization (GAPO)
- Graph Neural Networks for hospital + robot encoding
- Cross-attention for task-robot-node
- De-biasing mechanisms for autoregressive decisions
- Continuous time simulation with telemetry
"""
import torch
import numpy as np
from pathlib import Path
import time

from src.environment.gapo_env import GAPOTaskAssignmentEnv
from src.multi_agent_ppo.gapo_ppo import GAPOPPO, Memory


def main():
    ############## Hyperparameters ##############
    num_robots = 5
    num_nodes = 10
    max_episode_time = 28800.0  # 8 hours
    timestep_seconds = 1.0

    # GAPO parameters
    hidden_dim = 64
    num_attention_heads = 4
    use_gnn = True  # Set to False if torch_geometric not installed
    use_debiasing = True

    # PPO parameters
    lr = 0.0003
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2
    lambda_debias = 0.1

    # Training parameters
    max_episodes = 1000
    max_steps_per_episode = 50
    update_timestep = 2000
    save_interval = 50
    log_interval = 10

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/gapo")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GAPO TRAINING - Graph Attention + De-biasing")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"GNN Enabled: {use_gnn}")
    print(f"De-biasing Enabled: {use_debiasing}")
    print(f"Hidden Dimension: {hidden_dim}")
    print(f"Attention Heads: {num_attention_heads}")
    print(f"Learning Rate: {lr}")
    print(f"De-bias Lambda: {lambda_debias}")
    print("=" * 80)

    # Create environment
    env = GAPOTaskAssignmentEnv(
        num_robots=num_robots,
        num_nodes=num_nodes,
        max_episode_time=max_episode_time,
        timestep_seconds=timestep_seconds
    )

    # Create GAPO PPO
    ppo = GAPOPPO(
        node_feat_dim=5,
        edge_feat_dim=3,
        robot_feat_dim=12,
        task_feat_dim=12,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        use_gnn=use_gnn,
        use_debiasing=use_debiasing,
        lambda_debias=lambda_debias,
        device=device
    )

    # Training metrics
    time_step = 0
    memory = Memory()
    running_reward = 0
    episode_rewards = []

    print("\nStarting Training...")
    print("-" * 80)

    start_time = time.time()

    for i_episode in range(1, max_episodes + 1):
        state_dict = env.reset()
        episode_reward = 0.0
        num_assignments = 0
        num_holds = 0
        episode_start_time = time.time()

        for step in range(max_steps_per_episode):
            time_step += 1

            # Get action mask
            action_mask = env.get_action_mask()

            # Select action
            action = ppo.select_action(state_dict, memory, action_mask)

            # Track action type
            if action == env.HOLD_ACTION:
                num_holds += 1
            else:
                num_assignments += 1

            # Take step
            state_dict, reward, done, info = env.step(action)
            episode_reward += reward

            # Store reward and done
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # Update policy
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0

            if done:
                break

        # Episode metrics
        episode_time = time.time() - episode_start_time
        episode_rewards.append(episode_reward)
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])

            loss_info = ppo.get_last_loss_info()

            print(f"\nEpisode {i_episode}")
            print(f"  Episode Reward: {episode_reward:.2f}")
            print(f"  Avg Reward ({log_interval} eps): {avg_reward:.2f}")
            print(f"  Running Reward: {running_reward:.2f}")
            print(f"  Tasks Assigned: {num_assignments}")
            print(f"  Tasks Deferred (HOLD): {num_holds}")
            print(f"  Simulation Time: {info.get('current_time', 0) / 3600:.2f} hours")
            print(f"  Stockouts: {info.get('stockouts', 0)}")
            print(f"  Episode Time: {episode_time:.2f}s")

            if loss_info:
                print(f"\n  Loss Breakdown:")
                print(f"    Total: {loss_info.get('total_loss', 0):.4f}")
                print(f"    Actor: {loss_info.get('actor_loss', 0):.4f}")
                print(f"    Critic: {loss_info.get('critic_loss', 0):.4f}")
                print(f"    Entropy: {loss_info.get('entropy_loss', 0):.4f}")

                if use_debiasing:
                    print(f"    De-bias Total: {loss_info.get('debias_loss', 0):.4f}")
                    print(f"      - State Delta: {loss_info.get('state_delta', 0):.4f}")
                    print(f"      - Consistency: {loss_info.get('consistency', 0):.4f}")

            print("-" * 80)

        # Save model
        if i_episode % save_interval == 0:
            save_path = checkpoint_dir / f"gapo_ep{i_episode}.pth"
            ppo.save(str(save_path))
            print(f"✓ Model saved to {save_path}\n")

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"Total Training Time: {total_time / 3600:.2f} hours")
    print(f"Avg Episode Reward: {np.mean(episode_rewards):.2f}")
    print(f"Final Running Reward: {running_reward:.2f}")
    print("=" * 80)

    # Save final model
    final_model_path = checkpoint_dir / "gapo_final.pth"
    ppo.save(str(final_model_path))
    print(f"\n✓ Final model saved to {final_model_path}")

    # Save training metrics
    metrics_path = checkpoint_dir / "training_metrics.npz"
    np.savez(
        metrics_path,
        episode_rewards=np.array(episode_rewards),
        running_rewards=running_reward
    )
    print(f"✓ Training metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()
