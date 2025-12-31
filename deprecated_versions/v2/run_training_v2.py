"""
Autoregressive PPO training V2 with telemetry and continuous time simulation.

New features:
- Robot telemetry-based position tracking
- Inventory-driven task generation
- Stockout penalties
- Action masking for capacity/battery
- Continuous time simulation
"""
import torch
import numpy as np
from pathlib import Path
from src.environment.autoregressive_env_v2 import AutoregressiveTaskAssignmentEnvV2
from src.multi_agent_ppo.autoregressive_ppo import AutoregressivePPO
from src.multi_agent_ppo.memory import Memory
from src.environment.autoregressive_state_builder_v2 import get_state_dim


def main():
    ############## Hyperparameters ##############
    num_robots = 5
    num_nodes = 10
    max_episode_time = 28800.0  # 8 hours (seconds)
    timestep_seconds = 1.0  # 1 second simulation steps

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/autoregressive_ppo_v2")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = AutoregressiveTaskAssignmentEnvV2(
        num_robots=num_robots,
        num_nodes=num_nodes,
        max_episode_time=max_episode_time,
        timestep_seconds=timestep_seconds
    )

    # Get state dimension (updated for telemetry)
    state_dim = get_state_dim()  # 193 features
    action_dim = num_robots + 1  # Robots + HOLD

    # PPO hyperparameters
    lr = 0.0003
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2

    print("=" * 80)
    print("AUTOREGRESSIVE PPO TRAINING V2 - WITH TELEMETRY")
    print("=" * 80)
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Number of robots: {num_robots}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Max episode time: {max_episode_time / 3600:.1f} hours")
    print(f"Simulation timestep: {timestep_seconds} seconds")
    print("=" * 80)

    # Initialize PPO
    ppo = AutoregressivePPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip
    )

    # Training parameters
    max_episodes = 1000
    max_steps_per_episode = 50  # Max assignment decisions per episode
    update_timestep = 2000  # Update policy every N timesteps
    save_interval = 50  # Save model every N episodes
    log_interval = 10  # Log every N episodes

    # Training loop
    time_step = 0
    memory = Memory()

    print("\nStarting Training...")
    print("-" * 80)

    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        num_assignments = 0
        num_holds = 0

        for step in range(max_steps_per_episode):
            time_step += 1

            # Get action mask (prevent invalid actions)
            action_mask = env.get_action_mask()

            # Select action using policy
            action = ppo.select_action(state, memory, action_mask=action_mask)

            # Track action type
            if action == env.HOLD_ACTION:
                num_holds += 1
            else:
                num_assignments += 1

            # Take step in environment
            state, reward, done, info = env.step(action)
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

        # Logging
        if i_episode % log_interval == 0:
            print(f"Episode {i_episode}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Tasks Assigned: {num_assignments}")
            print(f"  Tasks Deferred (HOLD): {num_holds}")
            print(f"  Simulation Time: {info.get('current_time', 0) / 3600:.2f} hours")
            print(f"  Stockouts: {info.get('stockouts', 0)}")
            print(f"  Assignment Rate: {num_assignments / max(num_assignments + num_holds, 1) * 100:.1f}%")
            print("-" * 80)

        # Save model
        if i_episode % save_interval == 0:
            save_path = checkpoint_dir / f"ppo_autoregressive_v2_ep{i_episode}.pth"
            ppo.save(str(save_path))
            print(f"Model saved to {save_path}")

    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)

    # Save final model
    final_model_path = checkpoint_dir / "ppo_autoregressive_v2_final.pth"
    ppo.save(str(final_model_path))
    print(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    main()
