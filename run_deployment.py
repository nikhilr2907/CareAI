"""
Simulation runner with optional pygame visualization.
"""
import argparse
import time

import numpy as np
import torch
import pygame
from pathlib import Path
import os

from src.environment.gapo_env import GAPOTaskAssignmentEnv
from src.deployment.robot_bridge import MockRobotBridge
from src.multi_agent_ppo.gapo_ppo import GAPOPPO
from src.utils.training_utils import create_env_from_config_file
from src.utils.deployment_utils import (
    compute_bounds,
    draw_graph,
    draw_nodes,
    draw_tasks,
    draw_robots,
    rank_pending_tasks,
    rescore_robot_queues,
    assign_tasks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hospital robot simulation runner")
    parser.add_argument("--use-trained", action="store_true", help="Use trained policy")
    parser.add_argument("--checkpoint", type=str, default="outputs/gapo_latest/checkpoints/gapo_final.pth")
    parser.add_argument("--config", type=str, default="configs/revised_hospital_config_v3.json",
                        help="Path to hospital config JSON (default: revised_hospital_config_v3.json)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-robots", type=int, default=5)
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("--cycle-time", type=float, default=1.0)
    parser.add_argument("--time-scale", type=float, default=1.0, help="Sim seconds per real second")
    parser.add_argument("--max-runtime", type=float, default=3600.0)
    parser.add_argument("--max-assignments-per-step", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=60)
    parser.add_argument("--headless", action="store_true", help="Disable pygame visualization")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    return parser.parse_args()


def main():
    args = parse_args()

    # Config resolution: --config flag > HOSPITAL_CONFIG_NAME env var > fallback to procedural
    config_path = None
    config_name_override = os.environ.get("HOSPITAL_CONFIG_NAME")
    if config_name_override:
        config_path = Path("configs") / config_name_override
    elif args.config:
        config_path = Path(args.config)

    if config_path and config_path.exists():
        print(f"Loading hospital config: {config_path}")
        env, _ = create_env_from_config_file(
            str(config_path),
            num_robots=args.num_robots,
            max_episode_time=args.max_runtime,
            timestep_seconds=args.cycle_time
        )
        env.reset()
    else:
        print(f"Config not found at '{config_path}', falling back to procedural {args.num_nodes}-node layout")
        env = GAPOTaskAssignmentEnv(
            num_robots=args.num_robots,
            num_nodes=args.num_nodes,
            max_episode_time=args.max_runtime,
            timestep_seconds=args.cycle_time,
        )
        env.reset()

    robot_bridge = MockRobotBridge(env.robot_simulators)
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

        ppo = GAPOPPO(
            node_continuous_dim=node_continuous_dim,
            num_node_types=4,
            num_departments=10,
            num_shift_periods=4,
            num_day_types=2,
            node_type_embedding_dim=8,
            department_embedding_dim=16,
            shift_embedding_dim=4,
            day_type_embedding_dim=4,
            edge_feat_dim=edge_feat_dim,
            robot_feat_dim=19,
            task_feat_dim=15,
            queue_feat_dim=16,
            sku_feat_dim=sku_feat_dim,
            sku_embed_dim=sku_embed_dim,
            # Deployment path intentionally disables task-creation actor wiring for now.
            # This keeps inference aligned with the legacy allocation-only policy path.
            enable_task_creation_actor=False,
            device=args.device
        )
        ppo.load(args.checkpoint)

    if not args.headless:
        pygame.init()
        screen = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption("Hospital Robot Simulation")
        font = pygame.font.SysFont("Consolas", 14)
        bounds = compute_bounds(env.graph_state)
        clock = pygame.time.Clock()
    else:
        screen = None
        font = None
        bounds = None
        clock = None

    current_time = env.current_time
    cycle_count = 0
    total_tasks_assigned = 0
    total_tasks_completed = 0
    total_tasks_created = 0
    total_on_time = 0
    total_late = 0
    total_lateness = 0.0

    try:
        while current_time < args.max_runtime:
            cycle_start_time = time.time()

            if not args.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

            sim_dt = args.cycle_time * max(args.time_scale, 0.0)
            pending_before = len(env.pending_tasks)
            if env.pending_tasks:
                rank_pending_tasks(env, ppo)
            rescore_robot_queues(env, ppo)
            assigned_this_cycle = assign_tasks(env, ppo, args.max_assignments_per_step)
            total_tasks_assigned += assigned_this_cycle

            prev_completed = len(env.completed_tasks)
            _, _, done, _ = env.step(dt=sim_dt)
            current_time = env.current_time

            # Approximate tasks created this cycle from queue accounting:
            # pending_after = pending_before - assigned + created
            pending_after = len(env.pending_tasks)
            created_this_cycle = max(0, pending_after - pending_before + assigned_this_cycle)
            total_tasks_created += created_this_cycle

            completed_tasks = env.completed_tasks[prev_completed:]
            if completed_tasks:
                total_tasks_completed += len(completed_tasks)
                for task in completed_tasks:
                    if task.deadline:
                        lateness = current_time - task.deadline
                        if lateness <= 0:
                            total_on_time += 1
                        else:
                            total_late += 1
                            total_lateness += lateness

            if cycle_count % args.log_interval == 0:
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

                print(
                    f"time={current_time:.1f}s created={total_tasks_created} "
                    f"assigned={total_tasks_assigned} completed={total_tasks_completed} "
                    f"pending={pending} in_progress={assigned} "
                    f"pickup_items={pickup_items} dropoff_items={dropoff_items} "
                    f"pickup_ov={pickup_items_overflow} dropoff_ov={dropoff_items_overflow} "
                    f"on_time={total_on_time} late={total_late} avg_late={avg_lateness/60:.1f}m "
                    f"deadline_avg={avg_deadline/60:.1f}m min={min_deadline/60:.1f}m "
                    f"assigned_deadline_avg={avg_assigned_deadline/60:.1f}m min={min_assigned_deadline/60:.1f}m"
                )
                for robot, simulator in zip(env.robots, env.robot_simulators):
                    current_task = robot.current_task
                    leg_type = getattr(current_task, "leg_type", None) if current_task else None
                    parent_id = getattr(current_task, "parent_task_id", None) if current_task else None
                    pickup_done = robot.is_pickup_complete(parent_id) if parent_id is not None else None
                    print(
                        f"R{robot.robot_id} task={leg_type} parent={parent_id} pickup_done={pickup_done} "
                        f"q={robot.num_queued_tasks} ov={len(robot.overflow_queue)} "
                        f"cur_node={robot.current_node_index} "
                        f"target={simulator.current_target_node} edge={simulator.current_edge_index} "
                        f"pathq={len(simulator.path_queue)}"
                    )

            if not args.headless:
                screen.fill((18, 18, 22))
                draw_graph(screen, env.graph_state, bounds, args.width, args.height)
                draw_nodes(screen, env.graph_state, bounds, args.width, args.height, font)
                draw_tasks(screen, env, bounds, args.width, args.height)
                draw_robots(screen, env, bounds, args.width, args.height, font)
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

                y = 10
                for line in hud_lines:
                    screen.blit(font.render(line, True, (240, 240, 240)), (10, y))
                    y += 16

                pygame.display.flip()
                clock.tick(args.fps)

            cycle_count += 1

            if done:
                break

            elapsed = time.time() - cycle_start_time
            sleep_time = max(0, args.cycle_time - elapsed)
            if sleep_time > 0 and args.headless:
                time.sleep(sleep_time)

    finally:
        if not args.headless:
            pygame.quit()


if __name__ == "__main__":
    main()
