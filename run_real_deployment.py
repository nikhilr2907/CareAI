"""
GAPO Real-Robot Deployment Script.

Runs an ILC pilot policy (or heuristic baseline) against a live robot via ROS bridge —
inference only, no gradient updates. Intended for production use after a policy has been
trained with run_training_real.py or run_training.py.

Differences from run_training_real.py:
  - No PPO updates, no Memory buffer, no rollout boundaries
  - Actions selected greedily (deterministic=True) when a policy is loaded
  - Infinite step loop; stops on Ctrl-C or --max-steps
  - No warmup logic
  - No checkpoint saving

Usage:
    # Heuristic (nearest-robot, no trained policy) — useful as a safety baseline
    python run_real_deployment.py --config configs/ilc_pilot_v1_care_robotics.json

    # With trained policy
    python run_real_deployment.py --config configs/ilc_pilot_v1_care_robotics.json \\
                                  --use-trained --checkpoint outputs/.../gapo_real_final.pth

    # Custom bridge URL, capped step count, seeded RNG for reproducible task arrivals
    python run_real_deployment.py --config configs/ilc_pilot_v1_care_robotics.json \\
                                  --use-trained --checkpoint ... \\
                                  --bridge-url ws://192.168.1.50:9090 --max-steps 86400 --seed 42
"""
import argparse
import asyncio
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame
import torch

from src.deployment.robot_backend import ROSBridgeRobotBackend
from src.multi_agent_ppo.gapo_ppo import GAPOPPO
from src.utils.training_utils import (
    create_real_env_from_config_file,
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
from src.utils.decision_logger import DecisionLogger
from src.utils.ranking_snapshot_logger import RankingSnapshotLogger
from src.utils.robot_sample_logger import RobotSampleLogger
from src.analytics.realtime_collector import RealtimeAnalyticsCollector
from src.utils.fleet_event_logger import FleetEventLogger


def parse_deploy_args():
    parser = argparse.ArgumentParser(
        description='Deploy trained GAPO policy on real ILC pilot robot'
    )

    # Required
    parser.add_argument('--config', type=str, required=True,
                        help='Path to ILC config JSON file')

    # Policy (optional — falls back to heuristic nearest-robot when omitted)
    parser.add_argument('--use-trained', action='store_true',
                        help='Load and use a trained policy (default: heuristic assignment)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained policy checkpoint (.pth). Required only with --use-trained')

    # Connection
    parser.add_argument('--bridge-url', type=str, default=None,
                        help='WebSocket URL of ROS bridge (overrides ROS_BRIDGE_URL in .env)')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed numpy + torch RNGs (affects stochastic task generation; '
                             'world dynamics are not reproducible on real hardware)')

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

    # Digital twin
    parser.add_argument('--twin', action='store_true',
                        help='Enable digital twin WebSocket publisher (browser 3D view)')
    parser.add_argument('--twin-port', type=int, default=8766,
                        help='WebSocket port for digital twin (HTTP on port+1, default 8767)')

    # Visualization (pygame live view)
    parser.add_argument('--headless', action='store_true',
                        help='Disable pygame visualization (run headless)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Pygame FPS (default: 30)')
    parser.add_argument('--width', type=int, default=1200,
                        help='Visualization window width (default: 1200)')
    parser.add_argument('--height', type=int, default=800,
                        help='Visualization window height (default: 800)')

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

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")

    task_logger = TaskLogger(output_dir / "logs")
    decision_logger = DecisionLogger(output_dir / "logs")
    ranking_logger = RankingSnapshotLogger(output_dir / "logs")
    robot_sample_logger = RobotSampleLogger(output_dir / "logs")
    analytics = RealtimeAnalyticsCollector(
        output_dir=output_dir / "analytics",
        mode='real',
        snapshot_frequency=50,
        task_logger=task_logger,
        robot_sample_logger=robot_sample_logger,
    )

    # --- Validate inputs ---
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if args.use_trained:
        if not args.checkpoint:
            raise ValueError("--use-trained requires --checkpoint <path>")
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
    if args.use_trained:
        logger.info(f"Checkpoint:  {args.checkpoint}")
    else:
        logger.info("Checkpoint:  (heuristic mode — no trained policy)")
    logger.info(f"Bridge URL:  {robot_backend._bridge_url}")
    logger.info(f"Device:      {device}")
    logger.info(f"Max steps:   {args.max_steps or 'unlimited'}")
    logger.info("=" * 80)

    await robot_backend.start()
    logger.info("ROS bridge backend started")

    # Wait for first telemetry from each robot before env.reset() runs.
    # Yields so the WebSocket listener task can connect + receive its first
    # robot_state message. Logs progress every second; reports if the
    # listener task itself died (e.g. wrong URL, connection refused).
    _loop = asyncio.get_running_loop()
    _wait_start = _loop.time()
    _deadline = _wait_start + 10.0
    _last_log = _wait_start
    _expected_robots = num_robots if num_robots else 1
    while _loop.time() < _deadline:
        if all(robot_backend.get_telemetry(r) is not None for r in range(_expected_robots)):
            logger.info(f"Telemetry confirmed for all robots after {_loop.time() - _wait_start:.2f}s")
            break
        if _loop.time() - _last_log > 1.0:
            logger.info(
                f"Waiting for telemetry... bridge_url={robot_backend._bridge_url}, "
                f"elapsed={_loop.time() - _wait_start:.1f}s"
            )
            for rid, task in robot_backend._listen_tasks.items():
                if task.done():
                    try:
                        exc = task.exception()
                        logger.error(f"Listener for robot_id={rid} terminated with: {exc!r}")
                    except asyncio.CancelledError:
                        logger.error(f"Listener for robot_id={rid} was cancelled")
            _last_log = _loop.time()
        await asyncio.sleep(0.1)
    else:
        raise RuntimeError(
            f"No telemetry received within 10s. "
            f"bridge_url={robot_backend._bridge_url}. "
            f"Verify the bridge is running and reachable."
        )


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
        fleet_event_logger=FleetEventLogger(output_dir / "logs"),
        log_dir=output_dir / "logs",
    )
    logger.info(f"Nodes: {num_nodes}  Robots: {env.num_robots}  Edges: {len(env.graph_state.edges)}")
    analytics.sku_logger = env.sku_logger
    analytics.initialize(env.graph_state.nodes)

    # --- Build and load policy (or run heuristic) ---
    ppo = None
    if args.use_trained:
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
            task_feat_dim=10,
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
            edge_cost_manager=getattr(env, 'edge_cost_manager', None),
            device=device,
            logger=logger,
        )

        ppo.load(args.checkpoint, edge_cost_manager=getattr(env, 'edge_cost_manager', None))
        ppo.policy.eval()
        ppo.policy_old.eval()
        logger.info(f"Policy loaded from: {args.checkpoint}")
    else:
        logger.info("No trained policy: using heuristic task assignment (nearest robot)")

    # --- Deployment state ---
    total_steps = 0
    total_assignments = 0
    total_completions = 0
    total_tasks_created = 0
    total_on_time = 0
    total_late = 0
    total_lateness = 0.0
    completion_time_history = deque(maxlen=1000)
    start_time = time.time()

    # Diagnostic: log the first telemetry packet seen per robot so you can confirm
    # at a glance what the bridge is actually publishing (battery / position / etc).
    # If battery prints 0.000 here AND stays flat in the periodic log, the ROS bridge
    # is not populating battery_level → _should_robot_charge() always returns True →
    # tasks get assigned but the robot is force-routed to a hub instead of executing.
    first_telemetry_logged: set = set()

    state_dict = env.reset()
    apply_floorplan_layout(env.graph_state)

    twin_publisher = None
    if args.twin:
        from src.deployment.twin_publisher import TwinPublisher
        twin_publisher = TwinPublisher(graph_state=env.graph_state, port=args.twin_port)
        twin_publisher.start()
        logger.info(f"Open http://localhost:{args.twin_port + 1} to view the digital twin")

    # --- Visualization setup ---
    if not args.headless:
        pygame.init()
        screen = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption("ILC Pilot Robot Deployment - GAPO (Real Bridge)")
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

    logger.info("Environment initialised — starting deployment loop")
    logger.info("-" * 80)

    try:
        while args.max_steps is None or total_steps < args.max_steps:
            step_wall_start = time.monotonic()

            if not args.headless:
                quit_requested = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_requested = True
                        break
                if quit_requested:
                    logger.info("Pygame window closed — stopping deployment")
                    break

            # First-telemetry diagnostic — fires once per robot the moment the
            # bridge sends its first packet, so you can immediately see what fields
            # the bridge is populating (especially battery_level).
            for robot in env.robots:
                if robot.robot_id in first_telemetry_logged:
                    continue
                tele = robot.telemetry
                if tele is None:
                    continue
                first_telemetry_logged.add(robot.robot_id)
                logger.info(
                    f"FIRST_TELEMETRY R{robot.robot_id} step={total_steps} | "
                    f"battery={tele.battery_level:.3f} charging={int(bool(tele.is_charging))} "
                    f"available={int(bool(tele.is_available))} | "
                    f"x={tele.x:.2f} y={tele.y:.2f} heading={tele.heading:.2f} "
                    f"vel={tele.velocity_ms:.2f} | "
                    f"node={tele.current_node_index} edge={tele.current_edge_index} "
                    f"prog={tele.edge_progress:.2f} | "
                    f"load={tele.current_capacity}/{robot.max_capacity}"
                )
                if tele.battery_level == 0.0:
                    logger.warning(
                        f"  R{robot.robot_id} battery_level is 0.000 on first telemetry — "
                        f"likely the ROS bridge is not publishing a 'battery_level' field. "
                        f"_should_robot_charge() will return True every step and the robot "
                        f"will be routed to charge instead of executing assigned tasks."
                    )

            pending_before = len(env.pending_tasks)

            # Re-rank pending tasks, re-score robot queues, then assign — shared helpers
            # so behavior matches run_deployment.py and any future changes land in one place.
            rank_pending_tasks(env, ppo)
            rescore_robot_queues(env, ppo)
            assignments_this_step = assign_tasks(
                env, ppo, args.max_assignments_per_step,
                task_logger=task_logger,
                decision_logger=decision_logger,
                ranking_logger=ranking_logger,
                total_assignments=total_assignments,
                iteration=total_steps,
            )
            total_assignments += assignments_this_step
            state_dict = env._get_state_dict()

            # Env step
            prev_completed = len(env.completed_tasks)
            state_dict, _, _, _ = env.step(dt=1.0)
            total_steps += 1

            pending_after = len(env.pending_tasks)
            created_this_step = max(0, pending_after - pending_before + assignments_this_step)
            total_tasks_created += created_this_step

            newly_completed = env.completed_tasks[prev_completed:]
            for task in newly_completed:
                total_completions += 1
                if task.arrival_time is not None:
                    ct = env.current_time - task.arrival_time
                    completion_time_history.append(ct)
                if hasattr(task, 'deadline') and task.deadline:
                    lateness = env.current_time - task.deadline
                    if lateness <= 0:
                        total_on_time += 1
                    else:
                        total_late += 1
                        total_lateness += lateness
                task_logger.log_completion(
                    task.task_id, 0.0,
                    (env.current_time - task.arrival_time) if task.arrival_time is not None else None,
                    env.current_time,
                    distance_traveled=getattr(task, "distance_traveled", 0.0),
                    energy_consumed=getattr(task, "energy_consumed_wh", 0.0),
                    completed_task=task,
                )

            task_logger.log_running_summary(total_steps)
            task_logger.log_iteration_summary(total_steps)

            if total_steps % 10 == 0:
                analytics.record_robot_telemetry(env.robots, env.current_time)
            if total_steps % 50 == 0:
                analytics.increment_iteration()

            if total_steps % 50 == 0:
                snapshot_dir = output_dir / f"metrics_snapshot_step{total_steps}"
                task_logger.plot_metrics(snapshot_dir)
                logger.info(f"Task metrics snapshot saved to step {total_steps}")
                analytics.plot_snapshot(final=False)
                analytics.save_metrics_json()
                logger.info(f"Analytics snapshot saved to step {total_steps}")

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
                # Per-robot diagnostic block — exposes battery + charge gating so a
                # silent bridge (battery never published, defaults to 0.0) is obvious.
                # If battery is flat 0.0 across steps, _should_robot_charge() returns True
                # every tick → robot is force-routed to hub and tasks never execute.
                for robot in env.robots:
                    tele = robot.telemetry
                    if tele is None:
                        logger.info(f"  R{robot.robot_id} | NO TELEMETRY YET")
                        continue
                    try:
                        needs_charge = env._should_robot_charge(robot.robot_id)
                    except Exception as e:
                        needs_charge = f"ERR({type(e).__name__})"
                    current_task = robot.current_task
                    leg = getattr(current_task, "leg_type", None) if current_task else None
                    cur_task_id = current_task.task_id if current_task else None
                    logger.info(
                        f"  R{robot.robot_id} | "
                        f"battery={tele.battery_level:.3f} charging={int(bool(tele.is_charging))} "
                        f"needs_charge={needs_charge} | "
                        f"pos=({tele.x:.2f},{tele.y:.2f}) head={tele.heading:.2f} vel={tele.velocity_ms:.2f} | "
                        f"node={tele.current_node_index} edge={tele.current_edge_index} "
                        f"prog={tele.edge_progress:.2f} | "
                        f"q={len(robot.task_queue)} ov={len(robot.overflow_queue)} "
                        f"load={robot.current_load}/{robot.max_capacity} | "
                        f"task={cur_task_id} leg={leg} "
                        f"rem_path={len(tele.remaining_path or [])}"
                    )

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
                        if hasattr(task, 'get_time_to_deadline'):
                            assigned_deadlines.append(task.get_time_to_deadline(env.current_time))
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
                    deadlines = [
                        t.get_time_to_deadline(env.current_time)
                        for t in env.pending_tasks
                        if hasattr(t, 'get_time_to_deadline')
                    ]
                    avg_deadline = float(np.mean(deadlines)) if deadlines else 0.0
                    min_deadline = float(np.min(deadlines)) if deadlines else 0.0
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
                    f"step={total_steps}  sim_t={env.current_time/60:.1f}m",
                    f"created={total_tasks_created} assigned={total_assignments}",
                    f"completed={total_completions} pending={pending} in_progress={assigned}",
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

                for node in env.graph_state.nodes:
                    if node.node_type != "recovery":
                        continue
                    stockout = " STOCKOUT" if node.is_stockout else ""
                    line = (
                        f"{node.node_id}: "
                        f"{node.stock_level:.0f}/{node.max_stock:.0f}{stockout}"
                    )
                    if node.category_inventory:
                        cat_parts = [
                            f"{cat}:{node.get_category_stock_level(cat):.0f}/"
                            f"{node.get_category_max_stock(cat):.0f}"
                            for cat in node.category_inventory.keys()
                        ]
                        line += " | " + " ".join(cat_parts)
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
                twin_publisher.broadcast(env, robot_backend=robot_backend)

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

    final_metrics_dir = output_dir / "metrics_final"
    task_logger.plot_metrics(final_metrics_dir)
    logger.info(f"Final task metrics plots saved to {final_metrics_dir}")
    analytics.increment_iteration()
    analytics.plot_snapshot(final=True)
    analytics.save_metrics_json()
    logger.info(f"Final analytics plots saved to {analytics.output_dir / 'plots_final'}")
    logger.info(f"Analytics metrics saved to {analytics.output_dir}")

    if not args.headless:
        pygame.quit()

    if twin_publisher:
        twin_publisher.stop()

    await robot_backend.stop()


if __name__ == '__main__':
    asyncio.run(main())
