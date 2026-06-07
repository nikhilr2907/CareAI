import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from src.environment.gapo_env import GAPOTaskAssignmentEnv
from src.environment.graph.graph_state import GraphState
from src.environment.graph.edge import GraphEdge
from src.environment.graph.config_loader import load_config_from_file, get_num_robots_from_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train GAPO policy for ILC pilot robot task assignment')

    # Config selection (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=False)
    config_group.add_argument('--config', type=str,
                              help='Path to ILC config JSON file')
    config_group.add_argument('--configs-dir', type=str, default=None,
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
    parser.add_argument('--actor-lr', type=float, default=None,
                        help='Actor learning rate (default: same as --lr)')
    parser.add_argument('--critic-lr', type=float, default=None,
                        help='Critic learning rate (default: same as --lr)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension for GNN (default: 64)')
    parser.add_argument('--no-debiasing', action='store_true',
                        help='Disable de-biasing loss')
    parser.add_argument('--critic-coef', type=float, default=0.5,
                        help='Critic loss coefficient (default: 0.5)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='Entropy coefficient magnitude (default: 0.01)')
    parser.add_argument('--min-adv-std', type=float, default=1e-3,
                        help='Minimum advantage std for normalization (default: 1e-3)')
    parser.add_argument('--rollout-steps', type=int, default=1000,
                        help='Rollout steps per iteration (default: 1000)')

    # Real robot deployment
    parser.add_argument('--bridge-url', type=str, default=None,
                        help='WebSocket URL of the ROS bridge server (overrides ROS_BRIDGE_URL in .env)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume from')

    # System
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    # Environment
    parser.add_argument('--num-robots', type=int, default=None,
                        help='Number of robots (default: read from config or 5)')
    parser.add_argument('--max-episode-time', type=float, default=float('inf'),
                        help='Max episode time in seconds (default: inf = no episode resets)')
    parser.add_argument('--stochastic-tasks-per-hour', type=float, default=0.0,
                        help='Expected stochastic ad-hoc task generation rate per simulated hour (default: 0.0 for ILC)')
    parser.add_argument('--stochastic-task-cap-per-hour', type=int, default=2,
                        help='Hard cap on stochastic ad-hoc tasks created in any rolling simulated hour (default: 2)')
    parser.add_argument('--initial-stochastic-tasks', type=int, default=0,
                        help='Initial stochastic ad-hoc tasks at reset (bounded by hourly cap, default: 0)')

    # Logging and output
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory for checkpoints and logs (default: outputs)')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: auto-generated timestamp)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging interval in iterations (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Model save interval in iterations (default: 100)')
    parser.add_argument('--max-assignments-per-step', type=int, default=10,
                        help='Max task assignments per simulation step (default: 10)')
    parser.add_argument('--reward-clip', type=float, default=200.0,
                        help='Clip per-step rewards to [-reward_clip, reward_clip] (default: 200)')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                        help='Scale per-step rewards (default: 1.0)')
    parser.add_argument('--warmup-iters', type=int, default=50,
                        help='Warmup iterations using heuristic/mild penalties (default: 50)')
    parser.add_argument('--warmup-mix', type=float, default=0.8,
                        help='Probability of heuristic action during warmup (default: 0.8)')
    parser.add_argument('--warmup-consumption-scale', type=float, default=0.6,
                        help='Scale consumption rates during warmup (default: 0.6)')
    parser.add_argument('--warmup-entropy-mult', type=float, default=2.0,
                        help='Entropy coefficient multiplier during warmup (default: 2.0)')

    parser.add_argument('--ranking-max-pairs-per-group', type=int, default=64,
                        help='Max sampled assignment pairs per env-step group for ranking losses (default: 64)')
    parser.add_argument('--ranking-min-adv-gap', type=float, default=1e-4,
                        help='Skip ranking pairs with |adv_i-adv_j| below this threshold (default: 1e-4)')

    parser.add_argument('--eval-interval', type=int, default=200,
                        help='Evaluation interval in iterations (default: 200)')
    parser.add_argument('--eval-steps', type=int, default=1000,
                        help='Number of environment steps per evaluation (default: 1000)')
    parser.add_argument('--eval-seeds', type=int, nargs='*', default=[123, 456, 789],
                        help='Fixed seeds for evaluation runs (default: 123 456 789)')

    return parser.parse_args()


def load_curriculum_from_configs(config_paths: List[str]) -> List[Tuple[str, str]]:
    """Build curriculum list of (config_path, description) from file paths, skipping missing files."""
    curriculum = []

    for config_path in config_paths:
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"Warning: Config file not found: {config_path}, skipping...")
            continue

        description = config_path.stem.replace('_', ' ').title()
        curriculum.append((str(config_path), description))

    if not curriculum:
        raise ValueError("No valid config files found for curriculum!")

    return curriculum


def _build_graph_state(nodes, edge_pairs, meta) -> GraphState:
    """Parse nodes + edge pairs from config into a GraphState with fully populated GraphEdge objects."""
    graph_state = GraphState()
    graph_state.nodes = nodes
    edges_detailed = {(e.get("from"), e.get("to")): e for e in meta.get("edges_detailed", [])}
    graph_state.edges = []
    for from_idx, to_idx in edge_pairs:
        from_node = nodes[from_idx]
        to_node = nodes[to_idx]
        distance = np.sqrt((from_node.center_x - to_node.center_x) ** 2 +
                           (from_node.center_y - to_node.center_y) ** 2)
        detail = edges_detailed.get((from_idx, to_idx)) or edges_detailed.get((to_idx, from_idx))
        graph_state.edges.append(GraphEdge(
            from_node=from_node.node_id,
            to_node=to_node.node_id,
            distance_m=float(detail.get("distance_m", distance)) if detail else distance,
            corridor_width=1.9,
            entry_point=(from_node.center_x, from_node.center_y),
            exit_point=(to_node.center_x, to_node.center_y),
            max_v_ms=0.5,
            clutter_level=np.random.random() * 0.3,
            active_robot_ids=[],
            edge_id=detail.get("edge_id") if detail else None,
            mode=detail.get("mode") if detail else None,
            floor_delta=int(detail.get("floor_delta", 0)) if detail else 0,
            travel_time_model=detail.get("travel_time_model") if detail else None,
            constraints=detail.get("constraints", "") if detail else ""
        ))
    graph_state.sku_database = meta.get("sku_database")
    graph_state.demand_profiles = meta.get("demand_profiles")
    graph_state.category_order = meta.get("category_order")
    graph_state.location_tag_order = meta.get("location_tag_order", [])
    graph_state.school_schedule = meta.get("school_schedule")
    return graph_state


def create_env_from_config_file(config_path: str, num_robots: Optional[int] = None,
                                max_episode_time: float = 28800.0,
                                timestep_seconds: float = 1.0,
                                stochastic_tasks_per_hour: float = 2.0,
                                stochastic_task_cap_per_hour: int = 2,
                                initial_stochastic_tasks: int = 0,
                                fleet_event_logger=None,
                                log_dir=None):
    """Create a simulation environment from a config JSON file."""
    nodes, edge_pairs, meta = load_config_from_file(config_path)
    if num_robots is None:
        num_robots = get_num_robots_from_config(config_path)
    graph_state = _build_graph_state(nodes, edge_pairs, meta)
    env = GAPOTaskAssignmentEnv(
        num_robots=num_robots,
        num_nodes=len(nodes),
        max_episode_time=max_episode_time,
        timestep_seconds=timestep_seconds,
        stochastic_tasks_per_hour=stochastic_tasks_per_hour,
        max_stochastic_tasks_per_hour=stochastic_task_cap_per_hour,
        initial_stochastic_tasks=initial_stochastic_tasks,
        fleet_event_logger=fleet_event_logger,
        log_dir=log_dir,
    )
    env.graph_state = graph_state
    env._custom_graph_state = graph_state
    return env, len(nodes)


def create_real_env_from_config_file(config_path: str, robot_backend,
                                     num_robots: Optional[int] = None,
                                     max_episode_time: float = 28800.0,
                                     timestep_seconds: float = 1.0,
                                     stochastic_tasks_per_hour: float = 2.0,
                                     stochastic_task_cap_per_hour: int = 2,
                                     initial_stochastic_tasks: int = 0,
                                     fleet_event_logger=None,
                                     log_dir=None):
    """Create a real-deployment environment (GAPOTaskAssignmentEnvReal) from a config JSON file."""
    from ..environment.gapo_env_real import GAPOTaskAssignmentEnvReal

    nodes, edge_pairs, meta = load_config_from_file(config_path)
    if num_robots is None:
        num_robots = get_num_robots_from_config(config_path)
    graph_state = _build_graph_state(nodes, edge_pairs, meta)
    env = GAPOTaskAssignmentEnvReal(
        robot_backend=robot_backend,
        num_robots=num_robots,
        num_nodes=len(nodes),
        max_episode_time=max_episode_time,
        timestep_seconds=timestep_seconds,
        stochastic_tasks_per_hour=stochastic_tasks_per_hour,
        max_stochastic_tasks_per_hour=stochastic_task_cap_per_hour,
        initial_stochastic_tasks=initial_stochastic_tasks,
        fleet_event_logger=fleet_event_logger,
        log_dir=log_dir,
    )
    env.graph_state = graph_state
    env._custom_graph_state = graph_state
    return env, len(nodes)


def setup_logging_and_output(args):
    """
    Setup logging, output directories, and system monitoring for GPU cluster training.
    """
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"gapo_{timestamp}"

    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)

    logger = logging.getLogger('GAPO_Training')
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                                         datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    log_file = output_dir / "logs" / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")

    logger.info("=" * 80)
    logger.info("GAPO Training Session")
    logger.info("=" * 80)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Output Directory: {output_dir}")

    if torch.cuda.is_available():
        logger.info(f"GPU Available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.2f} GB)")
    else:
        logger.info("GPU: Not available, using CPU")

    config_dict = vars(args)
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info(f"Configuration saved to: {config_file}")
    logger.info("=" * 80)

    return logger, output_dir, args.exp_name


def log_gpu_memory(logger):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            logger.info(f"GPU {i} Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def select_nearest_robot(task, robots, graph_state, action_mask):
    available = [i for i in range(len(robots)) if action_mask[i]]
    if not available:
        available = list(range(len(robots)))
    task_node = graph_state.nodes[task.from_location_index]
    best_robot = available[0]
    best_dist = float('inf')
    for robot_id in available:
        robot = robots[robot_id]
        rx, ry = robot.current_position
        dist = ((task_node.center_x - rx) ** 2 + (task_node.center_y - ry) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_robot = robot_id
    return best_robot


def apply_consumption_scale(env, scale: Optional[float]):
    if scale is None:
        return
    if hasattr(env, "graph_state") and env.graph_state is not None:
        env.graph_state.consumption_scale = float(scale)


def run_evaluation(eval_env, eval_steps, eval_seed, policy, max_assignments_per_step,
                   timesteps_per_decision, reward_clip, use_baseline=False):
    """Run a fixed-seed evaluation rollout (greedy policy or baseline)."""
    torch.manual_seed(eval_seed)
    np.random.seed(eval_seed)
    state = eval_env.reset()
    eval_reward = 0.0
    eval_assignments = 0
    late_at_assignment = 0
    assigned_total = 0
    completed_on_time = 0
    completed_late = 0
    for _ in range(eval_steps):
        if eval_env.pending_tasks:
            eval_env.pending_tasks = policy.score_and_rank_tasks(
                eval_env.pending_tasks,
                state,
                eval_env.current_time,
                temperature=0.0,
            )

        for robot in eval_env.robots:
            all_tasks = robot.task_queue[1:] + robot.overflow_queue
            if all_tasks:
                task_features = np.stack([t.get_features(eval_env.current_time) for t in all_tasks])
                task_features_tensor = torch.tensor(task_features, dtype=torch.float32).to(policy.device)
                with torch.no_grad():
                    state_tensor = policy._state_dict_to_tensor(state)
                    graph_emb, fleet_emb = policy.policy_old.encode_context(state_tensor)
                    scores = policy.policy_old.score_tasks(task_features_tensor, graph_emb, fleet_emb).cpu().numpy()
                for task, score in zip(all_tasks, scores):
                    task.learned_score = float(score)
                robot.resort_queue()
                robot.resort_overflow()
            robot.enforce_capacity_limits()
            robot.promote_from_overflow()

        assignments_this_step = 0
        while eval_env.pending_tasks and assignments_this_step < max_assignments_per_step:
            task = eval_env.pending_tasks[0]
            robot_mask = np.ones(eval_env.num_robots, dtype=bool)
            if use_baseline:
                action = select_nearest_robot(task, eval_env.robots, eval_env.graph_state, robot_mask)
            else:
                action, _, _ = policy.select_action_greedy(eval_env._get_state_dict(), robot_mask)
            success = eval_env.assign_task_to_robot(action, task)
            if not success:
                # Primary robot rejected — check for other available robots and try heuristic.
                other_available = [
                    rid for rid in range(eval_env.num_robots)
                    if rid != action
                    and rid not in eval_env._offline_robots
                    and not eval_env._should_robot_charge(rid)
                ]
                if other_available:
                    fallback = select_nearest_robot(task, eval_env.robots, eval_env.graph_state, robot_mask)
                    if fallback != action and not eval_env._should_robot_charge(fallback):
                        success = eval_env.assign_task_to_robot(fallback, task)
                if not success:
                    break  # No robot available; stop assigning this step
            if success:
                eval_assignments += 1
                assignments_this_step += 1
                assigned_total += 1
                if task.get_time_to_deadline(eval_env.current_time) < 0:
                    late_at_assignment += 1

        prev_completed = len(eval_env.completed_tasks)
        state, step_reward, done, _ = eval_env.step(dt=timesteps_per_decision)
        if reward_clip is not None and reward_clip > 0:
            step_reward = float(np.clip(step_reward, -reward_clip, reward_clip))
        eval_reward += step_reward
        if len(eval_env.completed_tasks) > prev_completed:
            for task in eval_env.completed_tasks[prev_completed:]:
                if task.deadline is None:
                    continue
                if eval_env.current_time <= task.deadline:
                    completed_on_time += 1
                else:
                    completed_late += 1
        if done:
            state = eval_env.reset()

    return {
        "reward": eval_reward,
        "assignments": eval_assignments,
        "late_at_assignment": late_at_assignment,
        "assigned_total": assigned_total,
        "completed_on_time": completed_on_time,
        "completed_late": completed_late,
        "pending": len(eval_env.pending_tasks),
        "completed": len(eval_env.completed_tasks),
        "sim_hours": eval_env.current_time / 3600.0
    }
