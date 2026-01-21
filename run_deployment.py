"""
Simulation runner with optional pygame visualization.
"""
import argparse
import time
from typing import Tuple

import numpy as np
import torch
import pygame
from pathlib import Path

from src.environment.gapo_env import GAPOTaskAssignmentEnv
from src.environment.graph.hospital_config import HospitalConfig
from src.deployment.robot_bridge import MockRobotBridge
from src.environment.tasks.task_generator import generate_inventory_tasks, update_inventory_levels
from src.multi_agent_ppo.gapo_ppo import GAPOPPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hospital robot simulation runner")
    parser.add_argument("--use-trained", action="store_true", help="Use trained policy")
    parser.add_argument("--checkpoint", type=str, default="outputs/gapo_latest/checkpoints/gapo_final.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-robots", type=int, default=5)
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("--cycle-time", type=float, default=1.0)
    parser.add_argument("--max-runtime", type=float, default=3600.0)
    parser.add_argument("--max-assignments-per-step", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=60)
    parser.add_argument("--headless", action="store_true", help="Disable pygame visualization")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    return parser.parse_args()


def compute_bounds(graph_state) -> Tuple[float, float, float, float]:
    min_x = min(n.bounds[0] for n in graph_state.nodes)
    min_y = min(n.bounds[1] for n in graph_state.nodes)
    max_x = max(n.bounds[2] for n in graph_state.nodes)
    max_y = max(n.bounds[3] for n in graph_state.nodes)
    pad = 2.0
    return min_x - pad, min_y - pad, max_x + pad, max_y + pad


def world_to_screen(x, y, bounds, width, height, margin=40):
    min_x, min_y, max_x, max_y = bounds
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    sx = margin + (x - min_x) / span_x * (width - 2 * margin)
    sy = margin + (y - min_y) / span_y * (height - 2 * margin)
    return int(sx), int(height - sy)


def draw_graph(screen, graph_state, bounds, width, height):
    for edge in graph_state.edges:
        from_node = next(n for n in graph_state.nodes if n.node_id == edge.from_node)
        to_node = next(n for n in graph_state.nodes if n.node_id == edge.to_node)
        fx, fy = world_to_screen(from_node.center_x, from_node.center_y, bounds, width, height)
        tx, ty = world_to_screen(to_node.center_x, to_node.center_y, bounds, width, height)
        congestion = min(len(edge.active_robot_ids), 5)
        color = (80 + congestion * 30, 80, 80)
        pygame.draw.line(screen, color, (fx, fy), (tx, ty), 2)


def draw_nodes(screen, graph_state, bounds, width, height, font):
    type_colors = {
        "storage": (80, 130, 200),
        "corridor": (100, 100, 100),
        "recovery": (180, 80, 80),
        "hub": (80, 180, 120),
    }
    for node in graph_state.nodes:
        x, y = world_to_screen(node.center_x, node.center_y, bounds, width, height)
        w = max(6, int(node.width * 6))
        h = max(6, int(node.height * 6))
        color = type_colors.get(node.node_type, (160, 160, 160))
        rect = pygame.Rect(x - w // 2, y - h // 2, w, h)
        pygame.draw.rect(screen, color, rect)
        if node.is_stockout:
            pygame.draw.rect(screen, (255, 0, 0), rect, 2)
        label = font.render(node.node_id, True, (230, 230, 230))
        screen.blit(label, (x + 6, y + 6))


def draw_tasks(screen, env, bounds, width, height):
    for task in env.pending_tasks:
        from_node = env.graph_state.get_node_by_index(task.from_location_index)
        to_node = env.graph_state.get_node_by_index(task.to_location_index)
        fx, fy = world_to_screen(from_node.center_x, from_node.center_y, bounds, width, height)
        tx, ty = world_to_screen(to_node.center_x, to_node.center_y, bounds, width, height)
        color = (255, 200, 50) if task.manual_priority >= 4 else (200, 200, 200)
        pygame.draw.circle(screen, color, (fx, fy), 4)
        pygame.draw.circle(screen, (50, 200, 255), (tx, ty), 4)
        pygame.draw.line(screen, (80, 80, 120), (fx, fy), (tx, ty), 1)


def draw_robots(screen, env, bounds, width, height, font):
    for robot in env.robots:
        if not robot.telemetry:
            continue
        x, y = world_to_screen(robot.telemetry.x, robot.telemetry.y, bounds, width, height)
        load_ratio = min(robot.current_load / max(robot.max_capacity, 1), 1.0)
        color = (50 + int(200 * load_ratio), 80, 200)
        pygame.draw.circle(screen, color, (x, y), 8)
        label = font.render(f"R{robot.robot_id}", True, (255, 255, 255))
        screen.blit(label, (x + 10, y - 8))


def rank_pending_tasks(env, ppo):
    if not env.pending_tasks:
        return
    if ppo is None:
        env.pending_tasks.sort(key=lambda t: (-t.manual_priority, t.arrival_time))
        for i, t in enumerate(env.pending_tasks):
            t.queue_position = i
            t.learned_score = 0.0
        return
    task_features = np.stack([t.get_features(env.current_time) for t in env.pending_tasks])
    task_features_tensor = torch.tensor(task_features, dtype=torch.float32).to(ppo.device)
    with torch.no_grad():
        scores = ppo.policy.score_tasks(task_features_tensor).cpu().numpy()
    scored = list(zip(env.pending_tasks, scores))
    scored.sort(key=lambda x: (-x[1], -x[0].manual_priority))
    env.pending_tasks = [t for t, _ in scored]
    for i, (t, score) in enumerate(scored):
        t.queue_position = i
        t.learned_score = float(score)


def rescore_robot_queues(env, ppo):
    if ppo is None:
        return
    for robot in env.robots:
        all_tasks = robot.task_queue[1:] + robot.overflow_queue
        if not all_tasks:
            continue
        task_features = np.stack([t.get_features(env.current_time) for t in all_tasks])
        task_features_tensor = torch.tensor(task_features, dtype=torch.float32).to(ppo.device)
        with torch.no_grad():
            scores = ppo.policy.score_tasks(task_features_tensor).cpu().numpy()
        for task, score in zip(all_tasks, scores):
            task.learned_score = float(score)
        robot.resort_queue()
        robot.resort_overflow()
        robot.enforce_capacity_limits()
        robot.promote_from_overflow()


def assign_tasks(env, ppo, max_assignments):
    assignments = 0
    while env.pending_tasks and assignments < max_assignments:
        task = env.pending_tasks[0]
        robot_mask = np.ones(env.num_robots, dtype=bool)
        if ppo is not None:
            state_dict = env._get_state_dict()
            action = ppo.select_action_greedy(state_dict, robot_mask)
        else:
            action = select_nearest_robot(task, env.robots, env.graph_state, robot_mask)
        env.assign_task_to_robot(action, task)
        assignments += 1
    return assignments


def select_nearest_robot(task, robots, graph_state, action_mask):
    available_robots = [i for i in range(len(robots)) if action_mask[i] == 1]
    if not available_robots:
        available_robots = list(range(len(robots)))

    task_node = graph_state.nodes[task.from_location_index]
    min_distance = float('inf')
    best_robot = None

    for robot_id in available_robots:
        robot = robots[robot_id]
        robot_x, robot_y = robot.current_position
        distance = np.sqrt(
            (task_node.center_x - robot_x)**2 +
            (task_node.center_y - robot_y)**2
        )
        if distance < min_distance:
            min_distance = distance
            best_robot = robot_id

    return best_robot if best_robot is not None else 0


def main():
    args = parse_args()
    hospital_config = None
    configs_dir = Path("configs")
    if configs_dir.exists():
        config_files = sorted(configs_dir.glob("*.json"))
        if len(config_files) == 1:
            hospital_config = HospitalConfig.from_file(str(config_files[0]))
        elif len(config_files) > 1:
            hospital_config = HospitalConfig.from_file(str(config_files[0]))

    num_nodes = args.num_nodes
    if hospital_config is not None:
        num_nodes = len(hospital_config.nodes)

    env = GAPOTaskAssignmentEnv(
        num_robots=args.num_robots,
        num_nodes=num_nodes,
        max_episode_time=args.max_runtime,
        timestep_seconds=args.cycle_time,
        hospital_config=hospital_config
    )
    env.reset()

    robot_bridge = MockRobotBridge(env.robot_simulators)
    ppo = None
    if args.use_trained:
        ppo = GAPOPPO(
            node_continuous_dim=15,
            num_node_types=4,
            edge_feat_dim=12,
            robot_feat_dim=12,
            task_feat_dim=12,
            queue_feat_dim=11,
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

    current_time = 0.0
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

            env.current_time = current_time
            if current_time - env.last_inventory_check >= env.inventory_check_interval:
                new_tasks, env.next_task_id = generate_inventory_tasks(
                    env.graph_state, current_time, env.next_task_id
                )
                if new_tasks:
                    env.pending_tasks.extend(new_tasks)
                    total_tasks_created += len(new_tasks)
                    rank_pending_tasks(env, ppo)
                env.last_inventory_check = current_time
            rescore_robot_queues(env, ppo)
            total_tasks_assigned += assign_tasks(env, ppo, args.max_assignments_per_step)

            env._update_edge_congestion()
            env._update_robot_positions(args.cycle_time)

            completed_tasks = env._check_task_completions()
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

            update_inventory_levels(env.graph_state, args.cycle_time / 3600.0)

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

            current_time += args.cycle_time
            cycle_count += 1

            elapsed = time.time() - cycle_start_time
            sleep_time = max(0, args.cycle_time - elapsed)
            if sleep_time > 0 and args.headless:
                time.sleep(sleep_time)

    finally:
        if not args.headless:
            pygame.quit()


if __name__ == "__main__":
    main()
