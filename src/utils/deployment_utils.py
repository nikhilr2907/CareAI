"""
Utility helpers for GAPO deployment/simulation entrypoints.
"""
from typing import Tuple

import numpy as np
import torch
import pygame


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


def draw_robots(screen, env, bounds, width, height, font):
    for robot in env.robots:
        if not robot.telemetry:
            continue
        x, y = world_to_screen(robot.telemetry.x, robot.telemetry.y, bounds, width, height)

        # Draw the robot's actual planned movement path on the graph (if any).
        remaining_path = list(robot.telemetry.remaining_path or [])
        if remaining_path:
            path_points = [(x, y)]
            for node_idx in remaining_path:
                if 0 <= node_idx < len(env.graph_state.nodes):
                    node = env.graph_state.nodes[node_idx]
                    path_points.append(
                        world_to_screen(node.center_x, node.center_y, bounds, width, height)
                    )
            if len(path_points) >= 2:
                pygame.draw.lines(screen, (70, 170, 255), False, path_points, 2)

        load_ratio = min(robot.current_load / max(robot.max_capacity, 1), 1.0)
        color = (50 + int(200 * load_ratio), 80, 200)
        pygame.draw.circle(screen, color, (x, y), 10)

        # Heading/velocity indicator to make motion direction visible.
        heading = float(robot.telemetry.heading)
        vx = int(14 * np.cos(heading))
        vy = int(-14 * np.sin(heading))
        pygame.draw.line(screen, (235, 235, 235), (x, y), (x + vx, y + vy), 2)

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
    state_dict = env._get_state_dict()
    env.pending_tasks = ppo.score_and_rank_tasks(
        env.pending_tasks, state_dict, env.current_time, temperature=0.0
    )


def rescore_robot_queues(env, ppo):
    if ppo is None:
        return
    state_dict = env._get_state_dict()
    state_tensor = ppo._state_dict_to_tensor(state_dict)
    with torch.no_grad():
        graph_emb, fleet_emb = ppo.policy.encode_context(state_tensor)
    for robot in env.robots:
        all_tasks = robot.task_queue[1:] + robot.overflow_queue
        if not all_tasks:
            continue
        task_features = np.stack([t.get_features(env.current_time) for t in all_tasks])
        task_features_tensor = torch.tensor(task_features, dtype=torch.float32).to(ppo.device)
        with torch.no_grad():
            scores = ppo.policy.score_tasks(task_features_tensor, graph_emb, fleet_emb).cpu().numpy()
        for task, score in zip(all_tasks, scores):
            task.learned_score = float(score)
        robot.resort_queue()
        robot.resort_overflow()
        robot.enforce_capacity_limits()
        robot.promote_from_overflow()


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
