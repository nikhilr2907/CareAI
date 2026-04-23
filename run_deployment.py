"""
Deployment runner: inference-only simulation with optional pygame visualization.

Runs the environment with optional trained policy for task allocation.
Matches training script structure for consistency.

Usage:
    # Heuristic (no trained policy)
    python run_deployment.py --config configs/hospital_3nodes_consumables.json --headless

    # With trained policy
    python run_deployment.py --config configs/hospital_3nodes_consumables.json \\
        --use-trained --checkpoint outputs/gapo_latest/checkpoints/gapo_final.pth

    # Custom environment
    python run_deployment.py --num-robots 10 --max-runtime 7200 --headless
"""
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime
import os

import numpy as np
import torch
import pygame

from src.multi_agent_ppo.gapo_ppo import GAPOPPO
from src.utils.training_utils import (
    create_env_from_config_file,
    setup_logging_and_output,
)
from src.utils.deployment_utils import (
    apply_floorplan_layout,
    compute_bounds,
    draw_graph,
    draw_nodes,
    draw_tasks,
    draw_robots,
    rank_pending_tasks,
    rescore_robot_queues,
    assign_tasks,
)
from src.utils.task_logger import TaskLogger
from src.utils.fleet_event_logger import FleetEventLogger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments matching training script structure."""
    parser = argparse.ArgumentParser(
        description="ILC pilot robot task allocation deployment (inference-only)"
    )

    # Config selection (same as training)
    parser.add_argument(
        "--config", type=str,
        default="configs/ilc_pilot_v1_care_robotics.json",
        help="Path to ILC config JSON file"
    )

    # Policy/Model
    parser.add_argument(
        "--use-trained", action="store_true",
        help="Load and use trained policy (default: heuristic assignment)"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="outputs/gapo_latest/checkpoints/gapo_final.pth",
        help="Path to trained policy checkpoint"
    )

    # Environment (match training)
    parser.add_argument(
        "--num-robots", type=int, default=None,
        help="Number of robots (default: read from config or 5)"
    )
    parser.add_argument(
        "--max-runtime", type=float, default=3600.0,
        help="Max runtime in seconds (default: 3600)"
    )
    parser.add_argument(
        "--max-episode-time", type=float, default=28800.0,
        help="Max episode time for environment (default: 28800 = 8 hours)"
    )
    parser.add_argument(
        "--stochastic-tasks-per-hour", type=float, default=0.0,
        help="Ad-hoc task generation rate (default: 0.0 for ILC)"
    )
    parser.add_argument(
        "--stochastic-task-cap-per-hour", type=int, default=2,
        help="Max ad-hoc tasks per rolling hour (default: 2)"
    )
    parser.add_argument(
        "--initial-stochastic-tasks", type=int, default=0,
        help="Initial stochastic tasks at reset (default: 0)"
    )
    parser.add_argument(
        "--max-assignments-per-step", type=int, default=50,
        help="Max task assignments per step (default: 50)"
    )

    # Training parameters (for consistency, not used in deployment)
    parser.add_argument(
        "--reward-scale", type=float, default=1.0,
        help="Reward scale (for info logging, not used in deployment)"
    )
    parser.add_argument(
        "--reward-clip", type=float, default=200.0,
        help="Reward clip (for info logging, not used in deployment)"
    )

    # Hyperparameters (for validation/logging only)
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor (for info, should match training checkpoint)"
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.01,
        help="Entropy coefficient (for info, should match training)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64,
        help="Hidden dimension (for info, should match training)"
    )

    # System
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )

    # Logging & output (match training)
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory for logs and analytics (default: outputs)"
    )
    parser.add_argument(
        "--exp-name", type=str, default=None,
        help="Experiment name (default: auto-generated from config + timestamp)"
    )
    parser.add_argument(
        "--log-interval", type=int, default=60,
        help="Logging/print interval in simulation seconds (default: 60)"
    )

    # Simulation timing
    parser.add_argument(
        "--cycle-time", type=float, default=1.0,
        help="Simulation cycle time in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--time-scale", type=float, default=1.0,
        help="Sim seconds per real second (default: 1.0)"
    )

    # Digital twin
    parser.add_argument(
        "--twin", action="store_true",
        help="Enable digital twin WebSocket publisher (browser 3D view)"
    )
    parser.add_argument(
        "--twin-port", type=int, default=8766,
        help="WebSocket port for digital twin (HTTP served on port+1, default 8767)"
    )

    # Visualization
    parser.add_argument(
        "--headless", action="store_true",
        help="Disable pygame visualization (run headless)"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Pygame FPS (default: 30)"
    )
    parser.add_argument(
        "--width", type=int, default=1200,
        help="Visualization window width (default: 1200)"
    )
    parser.add_argument(
        "--height", type=int, default=800,
        help="Visualization window height (default: 800)"
    )

    return parser.parse_args()


def main():
    """Main deployment runner."""
    args = parse_args()

    # ============ Setup Logging & Output ============
    # Match training script logging setup
    if args.exp_name is None:
        config_stem = Path(args.config).stem if args.config else "procedural"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"deploy_{config_stem}_{timestamp}"

    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging (same as training)
    log_file = output_dir / "deployment.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    task_logger = TaskLogger(output_dir / "logs")

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")

    # ============ Device Setup ============
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # ============ Environment Setup ============
    logger.info("\n" + "=" * 80)
    logger.info("GAPO DEPLOYMENT - ILC Pilot Robot Task Assignment (Inference)")
    logger.info("=" * 80)

    # Config resolution: explicit --config > ILC_CONFIG_NAME env var
    config_path = None
    config_name_override = os.environ.get("ILC_CONFIG_NAME")
    if config_name_override:
        config_path = Path("configs") / config_name_override
        logger.info(f"Using config from env var ILC_CONFIG_NAME: {config_path}")
    elif args.config:
        config_path = Path(args.config)

    if not config_path or not config_path.exists():
        raise FileNotFoundError(
            f"ILC config not found at '{config_path}'. "
            f"Pass --config configs/ilc_pilot_v1_care_robotics.json or set ILC_CONFIG_NAME."
        )

    logger.info(f"Loading ILC config: {config_path}")
    env, num_nodes = create_env_from_config_file(
        str(config_path),
        num_robots=args.num_robots,
        max_episode_time=args.max_episode_time,
        timestep_seconds=args.cycle_time,
        stochastic_tasks_per_hour=args.stochastic_tasks_per_hour,
        stochastic_task_cap_per_hour=args.stochastic_task_cap_per_hour,
        initial_stochastic_tasks=args.initial_stochastic_tasks,
        fleet_event_logger=FleetEventLogger(output_dir / "logs")
    )
    env.reset()
    apply_floorplan_layout(env.graph_state)

    logger.info(f"  Nodes: {num_nodes}")
    logger.info(f"  Robots: {env.num_robots}")
    logger.info(f"  Edges: {len(env.graph_state.edges)}")

    total_categories = set()
    for node in env.graph_state.nodes:
        if hasattr(node, 'get_all_categories'):
            total_categories.update(node.get_all_categories())
    if total_categories:
        logger.info(f"  Categories: {len(total_categories)}")

    # ============ Policy Setup ============
    ppo = None
    if args.use_trained:
        logger.info("\nLoading trained policy...")
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Infer feature dimensions from environment (match training script)
        if getattr(env.graph_state, "category_order", None):
            base_node_dim = env.graph_state.get_node_features_with_category_stats()[0].shape[1]
            sku_feat_dim = env.graph_state.get_node_sku_features()[0].shape[2]
        else:
            base_node_dim = 8
            sku_feat_dim = None

        edge_feat_dim = env.graph_state.get_edge_features_complete()[0].shape[1]
        sku_embed_dim = 16
        node_continuous_dim = base_node_dim + (sku_embed_dim if sku_feat_dim is not None else 0)

        logger.info(f"  Node continuous dim: {node_continuous_dim}")
        logger.info(f"  Edge feature dim: {edge_feat_dim}")
        logger.info(f"  SKU feature dim: {sku_feat_dim}")

        num_location_tags = len(env.graph_state.location_tag_order) if getattr(env.graph_state, 'location_tag_order', None) else 3
        schedule = getattr(env.graph_state, 'school_schedule', None) or {}
        num_school_periods = len(schedule.get('periods', [])) or 8

        ppo = GAPOPPO(
            node_continuous_dim=node_continuous_dim,
            num_node_types=4,
            num_location_tags=num_location_tags,
            num_school_periods=num_school_periods,
            num_day_types=2,
            node_type_embedding_dim=8,
            location_tag_embedding_dim=16,
            school_period_embedding_dim=4,
            day_type_embedding_dim=4,
            edge_feat_dim=edge_feat_dim,
            robot_feat_dim=19,
            task_feat_dim=10,
            queue_feat_dim=16,
            sku_feat_dim=sku_feat_dim,
            sku_embed_dim=sku_embed_dim,
            task_creation_actor=None,
            edge_cost_manager=getattr(env, 'edge_cost_manager', None),
            device=device
        )
        ppo.load(str(checkpoint_path), edge_cost_manager=getattr(env, 'edge_cost_manager', None))
        logger.info(f"  Policy loaded from: {checkpoint_path}")
    else:
        logger.info("No trained policy: using heuristic task assignment")

    # ============ Log Hyperparameters ============
    logger.info("\n" + "-" * 80)
    logger.info("HYPERPARAMETERS (Info Only)")
    logger.info("-" * 80)
    logger.info(f"Gamma: {args.gamma}")
    logger.info(f"Entropy Coef: {args.entropy_coef}")
    logger.info(f"Hidden Dim: {args.hidden_dim}")
    logger.info(f"Reward Scale: {args.reward_scale}")
    logger.info(f"Reward Clip: {args.reward_clip}")
    logger.info(f"Max Assignments Per Step: {args.max_assignments_per_step}")
    logger.info(f"Stochastic Tasks: rate={args.stochastic_tasks_per_hour}/h "
                f"cap={args.stochastic_task_cap_per_hour}/h "
                f"initial={args.initial_stochastic_tasks}")
    logger.info("\n" + "=" * 80)

    # ============ Digital Twin Publisher ============
    twin_publisher = None
    if args.twin:
        from src.deployment.twin_publisher import TwinPublisher
        twin_publisher = TwinPublisher(graph_state=env.graph_state, port=args.twin_port)
        twin_publisher.start()
        logger.info(f"Open http://localhost:{args.twin_port + 1} to view the digital twin")

    # ============ Visualization Setup ============
    if not args.headless:
        pygame.init()
        screen = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption("ILC Pilot Robot Simulation - GAPO Deployment")
        font = pygame.font.SysFont("Consolas", 14)
        bounds = compute_bounds(env.graph_state)
        clock = pygame.time.Clock()
        logger.info(f"Visualization enabled: {args.width}x{args.height} @ {args.fps} FPS")
    else:
        screen = None
        font = None
        bounds = None
        clock = None
        logger.info("Running headless (no visualization)")

    # ============ Simulation State ============
    current_time = env.current_time
    cycle_count = 0
    total_timesteps = 0

    # Metrics (match training script tracking)
    total_tasks_assigned = 0
    total_tasks_completed = 0
    total_tasks_created = 0
    total_on_time = 0
    total_late = 0
    total_lateness = 0.0

    # For tracking per-cycle metrics
    last_log_time = current_time

    logger.info(f"\nStarting deployment loop (max_runtime={args.max_runtime}s)...")
    logger.info("=" * 80)

    try:
        while current_time < args.max_runtime:
            cycle_start_time = time.time()

            if not args.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

            # Calculate simulation timestep
            sim_dt = args.cycle_time * max(args.time_scale, 0.0)
            total_timesteps += 1

            # Task assignment phase
            pending_before = len(env.pending_tasks)

            # Rank pending tasks using policy or heuristic
            if env.pending_tasks:
                rank_pending_tasks(env, ppo)

            # Re-score robot queues
            rescore_robot_queues(env, ppo)

            # Assign top pending tasks to available robots
            assigned_this_cycle = assign_tasks(env, ppo, args.max_assignments_per_step, task_logger=task_logger, total_assignments=total_tasks_assigned)
            total_tasks_assigned += assigned_this_cycle

            # Simulation step
            prev_completed = len(env.completed_tasks)
            _, step_reward, done, info = env.step(dt=sim_dt)
            current_time = env.current_time

            # Track tasks created this cycle (from queue accounting)
            pending_after = len(env.pending_tasks)
            created_this_cycle = max(0, pending_after - pending_before + assigned_this_cycle)
            total_tasks_created += created_this_cycle

            # Track task completions and timeliness
            completed_tasks = env.completed_tasks[prev_completed:]
            if completed_tasks:
                total_tasks_completed += len(completed_tasks)
                for task in completed_tasks:
                    if hasattr(task, 'deadline') and task.deadline:
                        lateness = current_time - task.deadline
                        if lateness <= 0:
                            total_on_time += 1
                        else:
                            total_late += 1
                            total_lateness += lateness
                    task_logger.log_completion(
                        task.task_id, 0.0,
                        current_time - task.arrival_time if task.arrival_time is not None else None,
                        current_time,
                        distance_traveled=getattr(task, "distance_traveled", 0.0),
                        energy_consumed=getattr(task, "energy_consumed_wh", 0.0),
                        completed_task=task,
                    )

            # Log metrics at specified time interval
            if current_time - last_log_time >= args.log_interval:
                last_log_time = current_time

                # Gather metrics
                pending = len(env.pending_tasks)
                assigned = 0
                pickup_items = 0
                dropoff_items = 0
                pickup_items_overflow = 0
                dropoff_items_overflow = 0
                assigned_deadlines = []
                total_robot_load = 0

                for robot in env.robots:
                    assigned += len(robot.task_queue) + len(robot.overflow_queue)
                    total_robot_load += getattr(robot, 'current_load', 0)

                    for task in robot.task_queue + robot.overflow_queue:
                        if hasattr(task, 'get_time_to_deadline'):
                            assigned_deadlines.append(task.get_time_to_deadline(current_time))

                    for task in robot.task_queue:
                        leg_type = getattr(task, "leg_type", "full")
                        if leg_type == "pickup":
                            pickup_items += task.num_items
                        elif leg_type == "dropoff":
                            dropoff_items += task.num_items

                    for task in robot.overflow_queue:
                        leg_type = getattr(task, "leg_type", "full")
                        if leg_type == "pickup":
                            pickup_items_overflow += task.num_items
                        elif leg_type == "dropoff":
                            dropoff_items_overflow += task.num_items

                # Compute pending task deadline stats
                if pending:
                    deadlines = [
                        t.get_time_to_deadline(current_time)
                        for t in env.pending_tasks
                        if hasattr(t, 'get_time_to_deadline')
                    ]
                    avg_deadline = float(np.mean(deadlines)) if deadlines else 0.0
                    min_deadline = float(np.min(deadlines)) if deadlines else 0.0
                else:
                    avg_deadline = 0.0
                    min_deadline = 0.0

                # Compute assigned task deadline stats
                if assigned_deadlines:
                    avg_assigned_deadline = float(np.mean(assigned_deadlines))
                    min_assigned_deadline = float(np.min(assigned_deadlines))
                else:
                    avg_assigned_deadline = 0.0
                    min_assigned_deadline = 0.0

                avg_lateness = (total_lateness / max(total_late, 1)) if total_late > 0 else 0.0

                # Log to both file and console
                log_msg = (
                    f"[t={current_time/60:.1f}m] created={total_tasks_created} "
                    f"assigned={total_tasks_assigned} completed={total_tasks_completed} | "
                    f"pending={pending} in_progress={assigned} "
                    f"pickup_items={pickup_items} dropoff_items={dropoff_items} "
                    f"pickup_ov={pickup_items_overflow} dropoff_ov={dropoff_items_overflow} | "
                    f"on_time={total_on_time} late={total_late} avg_late={avg_lateness/60:.1f}m | "
                    f"deadline_pending_avg={avg_deadline/60:.1f}m min={min_deadline/60:.1f}m "
                    f"deadline_assigned_avg={avg_assigned_deadline/60:.1f}m min={min_assigned_deadline/60:.1f}m"
                )
                logger.info(log_msg)

                # Log per-robot status
                for robot, simulator in zip(env.robots, env.robot_simulators):
                    current_task = robot.current_task
                    leg_type = getattr(current_task, "leg_type", None) if current_task else None
                    parent_id = getattr(current_task, "parent_task_id", None) if current_task else None
                    pickup_done = (
                        robot.is_pickup_complete(parent_id)
                        if (parent_id is not None and hasattr(robot, 'is_pickup_complete'))
                        else None
                    )
                    battery_level = getattr(simulator, 'battery_level', 1.0)
                    is_charging = getattr(simulator, 'is_charging', False)

                    robot_msg = (
                        f"R{robot.robot_id} | task={leg_type} parent={parent_id} pickup_done={pickup_done} | "
                        f"q={robot.num_queued_tasks} ov={len(robot.overflow_queue)} "
                        f"load={getattr(robot, 'current_load', 0)}/{getattr(robot, 'max_capacity', '?')} | "
                        f"battery={battery_level:.2f} charging={is_charging} | "
                        f"node={robot.current_node_index} target={simulator.current_target_node} "
                        f"edge={getattr(simulator, 'current_edge_index', None)} "
                        f"pathq={len(getattr(simulator, 'path_queue', []))}"
                    )
                    logger.debug(robot_msg)

            if not args.headless:
                screen.fill((243, 246, 249))
                hud_panel_width = 430
                viz_width = max(640, args.width - hud_panel_width)
                draw_graph(screen, env.graph_state, bounds, viz_width, args.height)
                draw_nodes(screen, env.graph_state, bounds, viz_width, args.height, font)
                draw_tasks(screen, env, bounds, viz_width, args.height)
                draw_robots(screen, env, bounds, viz_width, args.height, font)
                pending = len(env.pending_tasks)
                assigned = 0
                pickup_items = 0
                dropoff_items = 0
                pickup_items_overflow = 0
                dropoff_items_overflow = 0
                assigned_deadlines = []
                for robot in env.robots:
                    assigned += len(robot.task_queue) + len(robot.overflow_queue)
                    for task in robot.task_queue + robot.overflow_queue:
                        assigned_deadlines.append(task.get_time_to_deadline(current_time))
                    for task in robot.task_queue:
                        if getattr(task, "leg_type", "full") == "pickup":
                            pickup_items += task.num_items
                        elif getattr(task, "leg_type", "full") == "dropoff":
                            dropoff_items += task.num_items
                    for task in robot.overflow_queue:
                        if getattr(task, "leg_type", "full") == "pickup":
                            pickup_items_overflow += task.num_items
                        elif getattr(task, "leg_type", "full") == "dropoff":
                            dropoff_items_overflow += task.num_items

                if pending:
                    deadlines = [t.get_time_to_deadline(current_time) for t in env.pending_tasks]
                    avg_deadline = float(np.mean(deadlines))
                    min_deadline = float(np.min(deadlines))
                else:
                    avg_deadline = 0.0
                    min_deadline = 0.0

                if assigned_deadlines:
                    avg_assigned_deadline = float(np.mean(assigned_deadlines))
                    min_assigned_deadline = float(np.min(assigned_deadlines))
                else:
                    avg_assigned_deadline = 0.0
                    min_assigned_deadline = 0.0

                avg_lateness = (total_lateness / max(total_late, 1)) if total_late > 0 else 0.0

                hud_lines = [
                    f"time={current_time/60:.1f}m",
                    f"created={total_tasks_created} assigned={total_tasks_assigned}",
                    f"completed={total_tasks_completed} pending={pending} in_progress={assigned}",
                    f"pickup_items={pickup_items} dropoff_items={dropoff_items}",
                    f"pickup_ov={pickup_items_overflow} dropoff_ov={dropoff_items_overflow}",
                    f"on_time={total_on_time} late={total_late} avg_late={avg_lateness/60:.1f}m",
                    f"deadline_avg={avg_deadline/60:.1f}m min={min_deadline/60:.1f}m",
                    f"assigned_deadline_avg={avg_assigned_deadline/60:.1f}m min={min_assigned_deadline/60:.1f}m",
                ]
                for robot in env.robots:
                    hud_lines.append(
                        f"R{robot.robot_id} load={robot.current_load}/{robot.max_capacity} "
                        f"q={robot.num_queued_tasks} ov={len(robot.overflow_queue)}"
                    )

                # Inventory summary for recovery nodes
                for node in env.graph_state.nodes:
                    if node.node_type != "recovery":
                        continue
                    line = f"{node.node_id}: {node.stock_level:.0f}/{node.max_stock:.0f} tts={node.time_to_stockout:.1f}h"
                    if node.category_inventory:
                        cat_name = next(iter(node.category_inventory.keys()))
                        cat_stock = node.get_category_stock_level(cat_name)
                        cat_max = node.get_category_max_stock(cat_name)
                        line += f" | {cat_name}:{cat_stock:.0f}/{cat_max:.0f}"
                    hud_lines.append(line)
                    if len(hud_lines) >= 12:
                        break

                panel_x = viz_width
                panel_rect = pygame.Rect(panel_x, 0, args.width - panel_x, args.height)
                pygame.draw.rect(screen, (233, 238, 244), panel_rect)
                pygame.draw.line(screen, (172, 182, 194), (panel_x, 0), (panel_x, args.height), 2)

                y = 10
                max_lines = max(1, (args.height - 20) // 16)
                for line in hud_lines[:max_lines]:
                    screen.blit(font.render(line, True, (36, 48, 60)), (panel_x + 10, y))
                    y += 16

                pygame.display.flip()
                clock.tick(args.fps)

            if twin_publisher:
                twin_publisher.broadcast(env)

            # Check for episode termination
            if done:
                logger.info(f"Episode complete at t={current_time/60:.1f}m")
                break

            # Frame timing control
            elapsed = time.time() - cycle_start_time
            sleep_time = max(0, args.cycle_time - elapsed)
            if sleep_time > 0 and args.headless:
                time.sleep(sleep_time)

            cycle_count += 1

    except KeyboardInterrupt:
        logger.info("\nDeployment interrupted by user")
    except Exception as e:
        logger.error(f"Error during deployment: {e}", exc_info=True)
        raise
    finally:
        task_logger.plot_metrics(output_dir)
        # Cleanup
        logger.info("\n" + "=" * 80)
        logger.info("DEPLOYMENT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total simulation time: {current_time/60:.1f} minutes ({current_time:.0f}s)")
        logger.info(f"Total cycles: {cycle_count}")
        logger.info(f"Total timesteps: {total_timesteps}")
        logger.info(f"\nTask Statistics:")
        logger.info(f"  Created: {total_tasks_created}")
        logger.info(f"  Assigned: {total_tasks_assigned}")
        logger.info(f"  Completed: {total_tasks_completed}")
        logger.info(f"  On-time: {total_on_time}")
        logger.info(f"  Late: {total_late}")
        if total_late > 0:
            logger.info(f"  Avg lateness (late tasks): {total_lateness / total_late / 60:.1f} minutes")

        if not args.headless:
            pygame.quit()

        if twin_publisher:
            twin_publisher.stop()

        logger.info(f"\nLogs saved to: {output_dir}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
