"""
GAPO Real-Robot Deployment Script.

Runs a trained ILC pilot policy against a live robot via ROS bridge — inference only,
no gradient updates. Intended for production use after a policy has been
trained with run_training_real.py or run_training.py.

Differences from run_training_real.py:
  - --checkpoint is required (policy must be pre-trained)
  - No PPO updates, no Memory buffer, no rollout boundaries
  - Actions selected greedily (deterministic=True)
  - Infinite step loop; stops on Ctrl-C or --max-steps
  - No warmup logic
  - No checkpoint saving

Usage:
    python run_real_deployment.py --config configs/ilc_pilot_v1_care_robotics.json \\
                                  --checkpoint outputs/.../gapo_real_final.pth
    python run_real_deployment.py --config configs/ilc_pilot_v1_care_robotics.json --checkpoint ... \\
                                  --bridge-url ws://192.168.1.50:9090
    python run_real_deployment.py --config configs/ilc_pilot_v1_care_robotics.json --checkpoint ... \\
                                  --max-steps 86400
"""
import argparse
import asyncio
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.deployment.robot_backend import ROSBridgeRobotBackend
from src.multi_agent_ppo.gapo_ppo import GAPOPPO
from src.utils.training_utils import (
    create_real_env_from_config_file,
    select_nearest_robot,
)
from src.utils.task_logger import TaskLogger
from src.analytics.realtime_collector import RealtimeAnalyticsCollector


def parse_deploy_args():
    parser = argparse.ArgumentParser(
        description='Deploy trained GAPO policy on real ILC pilot robot'
    )

    # Required
    parser.add_argument('--config', type=str, required=True,
                        help='Path to ILC config JSON file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained policy checkpoint (.pth)')

    # Connection
    parser.add_argument('--bridge-url', type=str, default=None,
                        help='WebSocket URL of ROS bridge (overrides ROS_BRIDGE_URL in .env)')

    # Environment
    parser.add_argument('--num-robots', type=int, default=None,
                        help='Number of robots (default: read from config)')
    parser.add_argument('--max-episode-time', type=float, default=28800.0,
                        help='Max episode time in seconds (default: 28800 = 8 hours)')
    parser.add_argument('--stochastic-tasks-per-hour', type=float, default=0.0,
                        help='Stochastic task generation rate per hour (default: 0.0 for ILC)')
    parser.add_argument('--stochastic-task-cap-per-hour', type=int, default=2,
                        help='Hard cap on stochastic tasks per rolling hour (default: 2)')
    parser.add_argument('--initial-stochastic-tasks', type=int, default=0,
                        help='Initial stochastic tasks at reset (default: 0)')
    parser.add_argument('--max-assignments-per-step', type=int, default=10,
                        help='Max task assignments per step (default: 10)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Stop after this many steps (default: run indefinitely)')

    # Model
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension — must match checkpoint (default: 64)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'])

    # Output
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory for logs and analytics (default: outputs)')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: auto-generated timestamp)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Log every N steps (default: 100)')

    return parser.parse_args()


async def main():
    args = parse_deploy_args()

    # --- Output / logging setup ---
    exp_name = args.exp_name or f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "deploy.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    task_logger = TaskLogger(output_dir / "logs")
    analytics = RealtimeAnalyticsCollector(
        output_dir=output_dir / "analytics",
        mode='real',
        snapshot_frequency=50,
        task_logger=task_logger,
    )

    # --- Validate inputs ---
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # --- Start robot backend ---
    num_robots = args.num_robots
    robot_backend = ROSBridgeRobotBackend(
        bridge_url=args.bridge_url,
        num_robots=num_robots if num_robots else 1,
    )

    logger.info("=" * 80)
    logger.info("GAPO REAL-ROBOT DEPLOYMENT - ILC Pilot")
    logger.info("=" * 80)
    logger.info(f"Config:      {args.config}")
    logger.info(f"Checkpoint:  {args.checkpoint}")
    logger.info(f"Bridge URL:  {robot_backend._bridge_url}")
    logger.info(f"Device:      {device}")
    logger.info(f"Max steps:   {args.max_steps or 'unlimited'}")
    logger.info("=" * 80)

    await robot_backend.start()
    logger.info("ROS bridge backend started")

    # --- Create environment ---
    env, num_nodes = create_real_env_from_config_file(
        args.config,
        robot_backend=robot_backend,
        num_robots=num_robots,
        max_episode_time=args.max_episode_time,
        timestep_seconds=1.0,
        stochastic_tasks_per_hour=args.stochastic_tasks_per_hour,
        stochastic_task_cap_per_hour=args.stochastic_task_cap_per_hour,
        initial_stochastic_tasks=args.initial_stochastic_tasks,
    )
    logger.info(f"Nodes: {num_nodes}  Robots: {env.num_robots}  Edges: {len(env.graph_state.edges)}")

    # --- Build and load policy ---
    if getattr(env.graph_state, "category_order", None):
        base_node_dim = env.graph_state.get_node_features_with_category_stats()[0].shape[1]
        sku_feat_dim = env.graph_state.get_node_sku_features()[0].shape[2]
    else:
        base_node_dim = 8
        sku_feat_dim = None
    edge_feat_dim = env.graph_state.get_edge_features_complete()[0].shape[1]
    sku_embed_dim = 16
    node_continuous_dim = base_node_dim + (sku_embed_dim if sku_feat_dim is not None else 0)
    num_location_tags = len(env.graph_state.location_tag_order) if getattr(env.graph_state, 'location_tag_order', None) else 3
    schedule = getattr(env.graph_state, 'school_schedule', None) or {}
    num_school_periods = len(schedule.get('periods', [])) or 8

    ppo = GAPOPPO(
        node_continuous_dim=node_continuous_dim,
        num_node_types=4,
        num_location_tags=num_location_tags,
        num_school_periods=num_school_periods,
        num_day_types=2,
        edge_feat_dim=edge_feat_dim,
        node_type_embedding_dim=8,
        location_tag_embedding_dim=16,
        school_period_embedding_dim=4,
        day_type_embedding_dim=4,
        robot_feat_dim=19,
        task_feat_dim=15,
        queue_feat_dim=16,
        sku_feat_dim=sku_feat_dim,
        sku_embed_dim=sku_embed_dim,
        hidden_dim=args.hidden_dim,
        num_attention_heads=4,
        lr=1e-4,          # unused in deployment, required by constructor
        actor_lr=1e-4,
        critic_lr=5e-5,
        gamma=0.99,
        K_epochs=4,
        eps_clip=0.2,
        use_debiasing=False,
        lambda_debias=0.1,
        lambda_gae=0.95,
        critic_coef=0.5,
        entropy_coef=0.01,
        min_adv_std=1e-3,
        ranking_max_pairs_per_group=64,
        ranking_min_adv_gap=1e-4,
        task_creation_actor=getattr(env, 'task_creation_actor', None),
        device=device,
        logger=logger,
    )

    ppo.load(args.checkpoint)
    ppo.policy.eval()
    ppo.policy_old.eval()
    logger.info(f"Policy loaded from: {args.checkpoint}")

    # --- Deployment state ---
    total_steps = 0
    total_assignments = 0
    total_completions = 0
    completion_time_history = deque(maxlen=1000)
    start_time = time.time()

    state_dict = env.reset()
    logger.info("Environment initialised — starting deployment loop")
    logger.info("-" * 80)

    try:
        while args.max_steps is None or total_steps < args.max_steps:
            step_wall_start = time.monotonic()

            # Re-rank pending tasks (task scorer, no gradient)
            if env.pending_tasks:
                ppo.score_and_rank_tasks(
                    env.pending_tasks, state_dict, env.current_time,
                    temperature=0.0,  # deterministic ranking
                )

            # Re-score robot queues (no gradient)
            for robot in env.robots:
                all_tasks = robot.task_queue[1:] + robot.overflow_queue
                if all_tasks:
                    task_features = np.stack([t.get_features(env.current_time) for t in all_tasks])
                    task_features_tensor = torch.tensor(task_features, dtype=torch.float32).to(ppo.device)
                    state_tensor = ppo._state_dict_to_tensor(state_dict)
                    with torch.no_grad():
                        graph_emb, fleet_emb = ppo.policy.encode_context(state_tensor)
                        scores = ppo.policy.score_tasks(
                            task_features_tensor, graph_emb, fleet_emb
                        ).cpu().numpy()
                    for task, score in zip(all_tasks, scores):
                        task.learned_score = float(score)
                    robot.resort_queue()
                    robot.resort_overflow()
                robot.enforce_capacity_limits()
                robot.promote_from_overflow()

            # Task assignment (greedy / deterministic)
            assignments_this_step = 0
            while len(env.pending_tasks) > 0 and assignments_this_step < args.max_assignments_per_step:
                task = env.pending_tasks[0]
                robot_mask = np.ones(env.num_robots, dtype=bool)

                try:
                    action = ppo.select_action_greedy(state_dict, robot_mask)
                except Exception as e:
                    logger.error(f"Action selection error at step {total_steps}: {e}")
                    raise

                success = env.assign_task_to_robot(action, task)
                if not success:
                    break

                total_assignments += 1
                assignments_this_step += 1
                task_logger.log_assignment(
                    task, action, 0.0,
                    total_steps, env.current_time, total_assignments, total_steps,
                )
                state_dict = env._get_state_dict()

            # Env step
            prev_completed = len(env.completed_tasks)
            state_dict, _, done, info = env.step(dt=1.0)
            total_steps += 1

            newly_completed = env.completed_tasks[prev_completed:]
            for task in newly_completed:
                total_completions += 1
                if task.arrival_time is not None:
                    ct = env.current_time - task.arrival_time
                    completion_time_history.append(ct)
                task_logger.log_completion(
                    task.task_id, 0.0,
                    (env.current_time - task.arrival_time) if task.arrival_time is not None else None,
                    env.current_time,
                    distance_traveled=0.0, energy_consumed=0.0,
                )

            if done:
                # Continuous operation — no reset, just clear per-episode task metadata
                task_logger.task_metadata.clear()

            # Periodic logging
            if total_steps % args.log_interval == 0:
                wall_hours = (time.time() - start_time) / 3600
                sim_hours = env.current_time / 3600
                avg_ct = float(np.mean(completion_time_history)) / 60 if completion_time_history else 0.0
                logger.info(
                    f"[Step {total_steps}] "
                    f"Assigned={total_assignments}  Completed={total_completions}  "
                    f"Pending={len(env.pending_tasks)}  "
                    f"AvgCompletionTime={avg_ct:.1f}min  "
                    f"SimTime={sim_hours:.2f}h  WallTime={wall_hours:.2f}h"
                )

            # Pace to 1 Hz — yields for WebSocket listen tasks
            elapsed = time.monotonic() - step_wall_start
            await asyncio.sleep(max(0.0, 1.0 - elapsed))

    except KeyboardInterrupt:
        logger.info("Deployment stopped by user (Ctrl-C)")

    # --- Teardown ---
    wall_hours = (time.time() - start_time) / 3600
    logger.info("-" * 80)
    logger.info(f"Deployment complete")
    logger.info(f"  Total steps:       {total_steps}")
    logger.info(f"  Total assignments: {total_assignments}")
    logger.info(f"  Total completions: {total_completions}")
    logger.info(f"  Wall time:         {wall_hours:.2f}h")
    if completion_time_history:
        logger.info(f"  Avg completion:    {float(np.mean(completion_time_history)) / 60:.1f}min")

    await robot_backend.stop()


if __name__ == '__main__':
    asyncio.run(main())
