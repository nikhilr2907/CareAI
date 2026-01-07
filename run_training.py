"""
GAPO Training Script with De-biasing.

Trains hospital robot task allocation using:
- Graph Attention-Based Policy Optimization (GAPO)
- Graph Neural Networks for hospital + robot encoding
- Cross-attention for task-robot-node
- De-biasing mechanisms for autoregressive decisions
- Continuous time simulation with telemetry
- Supports both single-config and multi-config curriculum learning

Usage:
    # Single config training
    python run_training.py --config configs/hospital_3nodes_consumables.json

    # Multi-config curriculum learning
    python run_training.py --configs-dir configs --curriculum adaptive

    # Custom multi-config
    python run_training.py --config-list configs/small.json configs/medium.json configs/large.json
"""
import torch
import numpy as np
from pathlib import Path
import time
import argparse

from src.environment.gapo_env import GAPOTaskAssignmentEnv
from src.environment.graph.graph_state import GraphState
from src.environment.graph.edge import HospitalEdge
from src.environment.graph.config_loader import load_config_from_file, list_available_configs, get_num_robots_from_config
from src.multi_agent_ppo.gapo_ppo import GAPOPPO, Memory


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train GAPO policy for hospital robot task assignment')

    # Config selection (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=False)
    config_group.add_argument('--config', type=str,
                              help='Path to single hospital config JSON file')
    config_group.add_argument('--configs-dir', type=str, default='configs',
                              help='Directory containing multiple config files for curriculum learning')
    config_group.add_argument('--config-list', type=str, nargs='+',
                              help='List of specific config files for curriculum learning')

    # Curriculum settings
    parser.add_argument('--curriculum', type=str, default='none', choices=['adaptive', 'fixed', 'random', 'none'],
                        help='Curriculum learning schedule (default: none)')
    parser.add_argument('--config-interval', type=int, default=500,
                        help='Config switch interval for fixed schedule (default: 500)')
    parser.add_argument('--perf-threshold', type=float, default=50.0,
                        help='Performance threshold for adaptive schedule (default: 50.0)')

    # Training hyperparameters
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of training iterations (default: 10000)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate (default: 0.0003)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension for GNN (default: 64)')
    parser.add_argument('--no-gnn', action='store_true',
                        help='Disable GNN encoders (use MLP fallback)')
    parser.add_argument('--no-debiasing', action='store_true',
                        help='Disable de-biasing loss')

    # System
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    # Environment
    parser.add_argument('--num-robots', type=int, default=None,
                        help='Number of robots (default: read from config or 5)')
    parser.add_argument('--max-episode-time', type=float, default=28800.0,
                        help='Max episode time in seconds (default: 28800 = 8 hours)')

    return parser.parse_args()


def load_curriculum_from_configs(config_paths: list) -> list:
    """
    Load curriculum from list of config file paths.

    Args:
        config_paths: List of paths to config JSON files

    Returns:
        List of (config_path, description) tuples
    """
    curriculum = []

    for config_path in config_paths:
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"Warning: Config file not found: {config_path}, skipping...")
            continue

        # Create description from filename
        description = config_path.stem.replace('_', ' ').title()

        curriculum.append((str(config_path), description))

    if not curriculum:
        raise ValueError("No valid config files found for curriculum!")

    return curriculum


def create_env_from_config_file(config_path: str, num_robots: int = None,
                                max_episode_time: float = 28800.0,
                                timestep_seconds: float = 1.0):
    """
    Create environment from config file.

    Args:
        config_path: Path to config JSON file
        num_robots: Number of robots (if None, reads from config or uses 5)
        max_episode_time: Max episode time in seconds
        timestep_seconds: Timestep duration

    Returns:
        env: GAPOTaskAssignmentEnv instance
        num_nodes: Number of nodes in the config
    """
    # Load config
    nodes, edge_pairs = load_config_from_file(config_path)
    num_nodes = len(nodes)

    # Determine number of robots
    if num_robots is None:
        num_robots = get_num_robots_from_config(config_path)

    # Create graph state from loaded nodes
    graph_state = GraphState()
    graph_state.nodes = nodes

    # Create edges from edge pairs
    graph_state.edges = []
    for from_idx, to_idx in edge_pairs:
        from_node = nodes[from_idx]
        to_node = nodes[to_idx]

        # Calculate distance
        distance = np.sqrt((from_node.center_x - to_node.center_x)**2 +
                          (from_node.center_y - to_node.center_y)**2)

        edge = HospitalEdge(
            from_node=from_node.node_id,
            to_node=to_node.node_id,
            distance_m=distance,
            corridor_width=1.9,
            entry_point=(from_node.center_x, from_node.center_y),
            exit_point=(to_node.center_x, to_node.center_y),
            max_v_ms=1.0,
            clutter_level=np.random.random() * 0.3,
            active_robot_ids=[],
            has_patient_bed=np.random.random() < 0.1
        )
        graph_state.edges.append(edge)

    # Create environment with custom graph state
    env = GAPOTaskAssignmentEnv(
        num_robots=num_robots,
        num_nodes=num_nodes,
        max_episode_time=max_episode_time,
        timestep_seconds=timestep_seconds,
        hospital_config=None  # We're providing graph_state directly
    )

    # Override graph_state with our custom one
    env.graph_state = graph_state

    return env, num_nodes


def main():
    # Parse command-line arguments
    args = parse_args()

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    ############## Load Configs ##############
    use_curriculum = args.curriculum != 'none'

    if args.config:
        # Single config mode
        if not Path(args.config).exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        curriculum_configs = [(args.config, Path(args.config).stem)]
        use_curriculum = False
        print(f"Single-config mode: {args.config}")

    elif args.config_list:
        # Explicit list of configs for curriculum
        curriculum_configs = load_curriculum_from_configs(args.config_list)
        print(f"Multi-config mode: {len(curriculum_configs)} configs specified")

    elif args.configs_dir:
        # Load all configs from directory
        available_configs = list_available_configs(args.configs_dir)
        if not available_configs:
            raise ValueError(f"No config files found in directory: {args.configs_dir}")
        curriculum_configs = load_curriculum_from_configs(available_configs)
        print(f"Curriculum mode: {len(curriculum_configs)} configs from {args.configs_dir}")

    else:
        raise ValueError("Must specify --config, --config-list, or --configs-dir")

    # If only one config, disable curriculum
    if len(curriculum_configs) == 1:
        use_curriculum = False

    current_config_idx = 0

    ############## Hyperparameters ##############
    num_robots = args.num_robots  # Can be None (will be read from config)
    max_episode_time = args.max_episode_time
    timestep_seconds = 1.0

    # Curriculum settings
    curriculum_schedule = args.curriculum
    config_switch_interval = args.config_interval
    performance_threshold = args.perf_threshold

    # GAPO parameters
    hidden_dim = args.hidden_dim
    num_attention_heads = 4
    use_gnn = not args.no_gnn
    use_debiasing = not args.no_debiasing

    # PPO parameters
    lr = args.lr
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2
    lambda_debias = 0.1
    lambda_gae = 0.95

    # Training parameters
    max_training_iterations = args.iterations
    rollout_steps = 2000
    timesteps_per_decision = 60.0
    save_interval = 100
    log_interval = 10

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/gapo")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GAPO TRAINING - Real Hospital Configs with Category Tracking")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"GNN Enabled: {use_gnn}")
    print(f"De-biasing Enabled: {use_debiasing}")
    print(f"Hidden Dimension: {hidden_dim}")
    print(f"Attention Heads: {num_attention_heads}")
    print(f"Learning Rate: {lr}")
    print(f"De-bias Lambda: {lambda_debias}")

    if use_curriculum:
        print(f"\nCurriculum Learning: Enabled ({curriculum_schedule} schedule)")
        print(f"  Stages: {len(curriculum_configs)} configurations")
        if curriculum_schedule == 'fixed':
            print(f"  Switch interval: {config_switch_interval} iterations")
        elif curriculum_schedule == 'adaptive':
            print(f"  Performance threshold: {performance_threshold} avg reward")

        print("\n  Config List:")
        for idx, (path, desc) in enumerate(curriculum_configs):
            print(f"    {idx + 1}. {desc} ({path})")
    else:
        print("\nSingle Config Training:")
        print(f"  Config: {curriculum_configs[0][1]} ({curriculum_configs[0][0]})")

    print("=" * 80)

    # Helper function to create environment with config
    def create_env_with_config(config_idx):
        """Create environment with specified config."""
        config_path, config_desc = curriculum_configs[config_idx]

        print(f"\n{'='*80}")
        if use_curriculum:
            print(f"Loading Config {config_idx + 1}/{len(curriculum_configs)}: {config_desc}")
        else:
            print(f"Loading Config: {config_desc}")
        print(f"  Path: {config_path}")
        print(f"{'='*80}")

        env, num_nodes = create_env_from_config_file(
            config_path,
            num_robots=num_robots,
            max_episode_time=max_episode_time,
            timestep_seconds=timestep_seconds
        )

        print(f"  Nodes: {num_nodes}")
        print(f"  Robots: {env.num_robots}")
        print(f"  Edges: {len(env.graph_state.edges)}")

        # Print category info if available
        total_categories = set()
        for node in env.graph_state.nodes:
            total_categories.update(node.get_all_categories())

        if total_categories:
            print(f"  Categories: {len(total_categories)} - {', '.join(sorted(total_categories))}")

        print(f"{'='*80}")

        return env, config_desc

    # Create initial environment
    env, current_config_desc = create_env_with_config(current_config_idx)

    # Create GAPO PPO
    ppo = GAPOPPO(
        node_continuous_dim=18,  # Enhanced with category info
        num_node_types=4,
        edge_feat_dim=15,  # Enhanced with congestion info
        robot_feat_dim=12,
        task_feat_dim=12,
        hidden_dim=hidden_dim,
        num_attention_heads=num_attention_heads,
        lr=lr,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        use_gnn=use_gnn,
        use_debiasing=use_debiasing,
        lambda_debias=lambda_debias,
        lambda_gae=lambda_gae,
        device=device
    )

    # Training metrics
    memory = Memory()
    running_reward = 0
    iteration_rewards = []
    config_rewards = []
    total_timesteps = 0
    iterations_on_current_config = 0

    print("\nStarting Training...")
    print("-" * 80)

    start_time = time.time()

    # Initialize environment
    state_dict = env.reset()

    for iteration in range(1, max_training_iterations + 1):
        iterations_on_current_config += 1
        iteration_reward = 0.0
        num_assignments = 0
        num_holds = 0
        iteration_start_time = time.time()

        # Collect rollout_steps timesteps of experience
        for step in range(rollout_steps):
            total_timesteps += 1

            # Autoregressive task assignment phase
            while len(env.pending_tasks) > 0:
                task = env.pending_tasks[0]

                # Build action mask
                action_mask = np.ones(env.num_robots + 1, dtype=bool)
                for i, robot in enumerate(env.robots):
                    if not robot.can_accept_items(task.num_items):
                        action_mask[i] = False

                # Select action
                action = ppo.select_action(state_dict, memory, action_mask)

                # Handle HOLD action
                if action == env.num_robots:
                    num_holds += 1
                    reward = -0.01

                    memory.rewards.append(reward)
                    memory.is_terminals.append(False)
                    iteration_reward += reward
                    break

                # Assign task to robot
                success = env.assign_task_to_robot(action, task)
                if success:
                    num_assignments += 1
                    reward = 1.0
                    state_dict = env._get_state_dict()
                else:
                    reward = -1.0

                iteration_reward += reward
                memory.rewards.append(reward)
                memory.is_terminals.append(False)

            # Simulation time step
            state_dict, step_reward, done, info = env.step(Δt=timesteps_per_decision)

            if step_reward != 0:
                iteration_reward += step_reward
                if len(memory.rewards) > 0:
                    memory.rewards[-1] += step_reward

            if done:
                state_dict = env.reset()

        # Update policy
        next_state_dict = state_dict
        ppo.update(memory, next_state_dict)
        memory.clear_memory()

        # Metrics
        iteration_time = time.time() - iteration_start_time
        iteration_rewards.append(iteration_reward)
        config_rewards.append(iteration_reward)
        running_reward = 0.05 * iteration_reward + (1 - 0.05) * running_reward

        # Curriculum switching logic
        should_switch_config = False

        if use_curriculum and current_config_idx < len(curriculum_configs) - 1:
            if curriculum_schedule == 'fixed':
                if iterations_on_current_config >= config_switch_interval:
                    should_switch_config = True

            elif curriculum_schedule == 'adaptive':
                if len(config_rewards) >= 20:
                    recent_avg = np.mean(config_rewards[-20:])
                    if recent_avg >= performance_threshold:
                        should_switch_config = True

            elif curriculum_schedule == 'random':
                if iterations_on_current_config >= 100 and np.random.random() < 0.1:
                    should_switch_config = True

        if should_switch_config:
            current_config_idx += 1
            print(f"\n{'='*80}")
            print(f"CURRICULUM ADVANCE: Moving to config {current_config_idx + 1}/{len(curriculum_configs)}")
            print(f"  Previous config avg reward: {np.mean(config_rewards[-20:]):.2f}")
            print(f"{'='*80}\n")

            env, current_config_desc = create_env_with_config(current_config_idx)
            state_dict = env.reset()

            config_rewards = []
            iterations_on_current_config = 0

        # Logging
        if iteration % log_interval == 0:
            avg_reward = np.mean(iteration_rewards[-log_interval:])
            loss_info = ppo.get_last_loss_info()

            print(f"\nIteration {iteration}")
            if use_curriculum:
                print(f"  Config: {current_config_idx + 1}/{len(curriculum_configs)} - {current_config_desc}")
                print(f"  Iterations on config: {iterations_on_current_config}")
            print(f"  Iteration Reward: {iteration_reward:.2f}")
            print(f"  Avg Reward ({log_interval} iters): {avg_reward:.2f}")
            print(f"  Running Reward: {running_reward:.2f}")
            print(f"  Tasks Assigned: {num_assignments}")
            print(f"  Tasks Deferred (HOLD): {num_holds}")
            print(f"  Total Timesteps: {total_timesteps}")
            print(f"  Simulation Time: {env.current_time / 3600:.2f} hours")
            print(f"  Pending Tasks: {len(env.pending_tasks)}")
            print(f"  Completed Tasks: {len(env.completed_tasks)}")
            print(f"  Iteration Time: {iteration_time:.2f}s")

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
        if iteration % save_interval == 0:
            save_path = checkpoint_dir / f"gapo_iter{iteration}.pth"
            ppo.save(str(save_path))
            print(f"✓ Model saved to {save_path}\n")

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"Total Training Time: {total_time / 3600:.2f} hours")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Avg Iteration Reward: {np.mean(iteration_rewards):.2f}")
    print(f"Final Running Reward: {running_reward:.2f}")
    print(f"Final Simulation Time: {env.current_time / 3600:.2f} hours")
    print(f"Total Completed Tasks: {len(env.completed_tasks)}")
    print("=" * 80)

    # Save final model
    final_model_path = checkpoint_dir / "gapo_final.pth"
    ppo.save(str(final_model_path))
    print(f"\n✓ Final model saved to {final_model_path}")

    # Save training metrics
    metrics_path = checkpoint_dir / "training_metrics.npz"
    np.savez(
        metrics_path,
        iteration_rewards=np.array(iteration_rewards),
        running_reward=running_reward,
        total_timesteps=total_timesteps
    )
    print(f"✓ Training metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()
