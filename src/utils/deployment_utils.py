"""Utility helpers for GAPO deployment/simulation entrypoints."""
from typing import Dict, List, Tuple

import numpy as np
import torch
import pygame


def _get_node_center(node) -> Tuple[float, float]:
    return (
        float(getattr(node, "viz_center_x", node.center_x)),
        float(getattr(node, "viz_center_y", node.center_y)),
    )

def _get_node_size(node) -> Tuple[float, float]:
    return (
        float(getattr(node, "viz_width", node.width)),
        float(getattr(node, "viz_height", node.height)),
    )

def _node_label(node) -> str:
    display_name = getattr(node, "display_name", None)
    if display_name:
        return str(display_name)
    return str(node.node_id)

def _short_label(text: str, max_len: int = 22) -> str:
    text = str(text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"

def apply_floorplan_layout(graph_state):
    """
    Create a clean floorplan-style visualization layout without changing simulation
    dynamics. Stores coordinates in node.viz_* attributes used by renderer only.
    """
    nodes = list(graph_state.nodes)
    if not nodes:
        return

    node_idx = {node.node_id: i for i, node in enumerate(nodes)}
    adjacency: Dict[int, List[int]] = {i: [] for i in range(len(nodes))}
    for edge in graph_state.edges:
        a = node_idx.get(edge.from_node)
        b = node_idx.get(edge.to_node)
        if a is None or b is None:
            continue
        adjacency[a].append(b)
        adjacency[b].append(a)

    spine_indices = [
        i for i, node in enumerate(nodes) if node.node_type in ("corridor", "hub")
    ]
    if not spine_indices:
        spine_indices = sorted(
            range(len(nodes)), key=lambda i: len(adjacency[i]), reverse=True
        )[: max(1, min(3, len(nodes)))]

    spine_indices = sorted(
        spine_indices,
        key=lambda i: (
            float(nodes[i].center_x),
            float(nodes[i].center_y),
            nodes[i].node_id,
        ),
    )

    base_x = 8.0
    x_gap = 14.0
    main_y = 20.0
    side_gap = 10.0
    stack_gap = 5.0

    anchor_pos: Dict[int, Tuple[float, float]] = {}
    for k, i in enumerate(spine_indices):
        anchor_pos[i] = (base_x + k * x_gap, main_y)

    non_spine = [i for i in range(len(nodes)) if i not in anchor_pos]
    slots: Dict[Tuple[int, int], int] = {}

    for i in non_spine:
        if anchor_pos:
            nearest_anchor_idx = min(
                anchor_pos.keys(),
                key=lambda j: (
                    (nodes[i].center_x - nodes[j].center_x) ** 2
                    + (nodes[i].center_y - nodes[j].center_y) ** 2
                ),
            )
        else:
            nearest_anchor_idx = spine_indices[0]

        ax, ay = anchor_pos[nearest_anchor_idx]
        node_type = nodes[i].node_type
        if node_type == "recovery":
            side = 1
        elif node_type == "storage":
            side = -1
        else:
            side = 1 if (i % 2 == 0) else -1

        key = (nearest_anchor_idx, side)
        slot = slots.get(key, 0)
        slots[key] = slot + 1

        col_offset = (slot % 3) - 1
        row_offset = slot // 3
        x = ax + col_offset * 4.0
        y = ay + side * (side_gap + row_offset * stack_gap)
        anchor_pos[i] = (x, y)

    size_by_type = {
        "storage": (7.0, 4.5),
        "recovery": (8.0, 5.0),
        "corridor": (5.5, 2.5),
        "hub": (6.0, 3.6),
    }

    for i, node in enumerate(nodes):
        x, y = anchor_pos[i]
        w, h = size_by_type.get(node.node_type, (6.0, 4.0))
        node.viz_center_x = float(x)
        node.viz_center_y = float(y)
        node.viz_width = float(w)
        node.viz_height = float(h)

def compute_bounds(graph_state) -> Tuple[float, float, float, float]:
    min_x = min(_get_node_center(n)[0] - _get_node_size(n)[0] * 0.5 for n in graph_state.nodes)
    min_y = min(_get_node_center(n)[1] - _get_node_size(n)[1] * 0.5 for n in graph_state.nodes)
    max_x = max(_get_node_center(n)[0] + _get_node_size(n)[0] * 0.5 for n in graph_state.nodes)
    max_y = max(_get_node_center(n)[1] + _get_node_size(n)[1] * 0.5 for n in graph_state.nodes)
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
    node_map = {n.node_id: n for n in graph_state.nodes}

    # Corridor background
    for edge in graph_state.edges:
        from_node = node_map.get(edge.from_node)
        to_node = node_map.get(edge.to_node)
        if from_node is None or to_node is None:
            continue
        from_x, from_y = _get_node_center(from_node)
        to_x, to_y = _get_node_center(to_node)
        fx, fy = world_to_screen(from_x, from_y, bounds, width, height)
        tx, ty = world_to_screen(to_x, to_y, bounds, width, height)
        pygame.draw.line(screen, (214, 220, 228), (fx, fy), (tx, ty), 12)

    # Active travel overlay
    for edge in graph_state.edges:
        from_node = node_map.get(edge.from_node)
        to_node = node_map.get(edge.to_node)
        if from_node is None or to_node is None:
            continue
        from_x, from_y = _get_node_center(from_node)
        to_x, to_y = _get_node_center(to_node)
        fx, fy = world_to_screen(from_x, from_y, bounds, width, height)
        tx, ty = world_to_screen(to_x, to_y, bounds, width, height)
        congestion = min(len(edge.active_robot_ids), 5)
        color = (110, 145 + congestion * 12, 178)
        pygame.draw.line(screen, color, (fx, fy), (tx, ty), 2)


def draw_nodes(screen, graph_state, bounds, width, height, font):
    type_colors = {
        "storage": (136, 184, 228),
        "corridor": (206, 212, 218),
        "recovery": (245, 203, 145),
        "hub": (151, 218, 186),
    }
    node_draw_data = []
    for node in graph_state.nodes:
        cx, cy = _get_node_center(node)
        nw, nh = _get_node_size(node)
        x, y = world_to_screen(cx, cy, bounds, width, height)
        w = max(22, int(nw * 8))
        h = max(14, int(nh * 8))
        color = type_colors.get(node.node_type, (160, 160, 160))
        rect = pygame.Rect(x - w // 2, y - h // 2, w, h)
        node_draw_data.append((node, x, y, w, h, color, rect))

    for node, _, _, _, _, color, rect in node_draw_data:
        pygame.draw.rect(screen, color, rect, border_radius=6)
        pygame.draw.rect(screen, (112, 126, 140), rect, width=1, border_radius=6)
        if node.is_stockout:
            pygame.draw.rect(screen, (255, 86, 86), rect, 2, border_radius=6)

    # Place labels in a second pass so they avoid both node geometry and each other.
    blocked_rects = [rect.inflate(10, 10) for _, _, _, _, _, _, rect in node_draw_data]
    occupied_label_rects = []
    for node, x, y, w, h, _, _ in sorted(node_draw_data, key=lambda t: (t[2], t[1])):
        name_text = _short_label(_node_label(node), max_len=18)
        label = font.render(name_text, True, (37, 50, 63))
        lw, lh = label.get_size()
        candidates = [
            (x + w // 2 + 8, y - lh // 2),
            (x - w // 2 - lw - 8, y - lh // 2),
            (x - lw // 2, y - h // 2 - lh - 8),
            (x - lw // 2, y + h // 2 + 8),
            (x + w // 2 + 8, y - h // 2 - lh - 6),
            (x + w // 2 + 8, y + h // 2 + 6),
            (x - w // 2 - lw - 8, y - h // 2 - lh - 6),
            (x - w // 2 - lw - 8, y + h // 2 + 6),
        ]

        placed = False
        for lx, ly in candidates:
            label_rect = pygame.Rect(lx - 2, ly - 1, lw + 4, lh + 2)
            if label_rect.left < 0 or label_rect.right > width:
                continue
            if label_rect.top < 0 or label_rect.bottom > height:
                continue
            if any(label_rect.colliderect(r) for r in occupied_label_rects):
                continue
            if any(label_rect.colliderect(r) for r in blocked_rects):
                continue
            screen.blit(label, (lx, ly))
            occupied_label_rects.append(label_rect)
            placed = True
            break
        if not placed:
            # Skip labels when space is dense; avoids unreadable overlap.
            continue


def draw_tasks(screen, env, bounds, width, height):
    for task in env.pending_tasks:
        from_node = env.graph_state.get_node_by_index(task.from_location_index)
        to_node = env.graph_state.get_node_by_index(task.to_location_index)
        from_x, from_y = _get_node_center(from_node)
        to_x, to_y = _get_node_center(to_node)
        fx, fy = world_to_screen(from_x, from_y, bounds, width, height)
        tx, ty = world_to_screen(to_x, to_y, bounds, width, height)
        color = (255, 200, 50) if task.manual_priority >= 4 else (200, 200, 200)
        pygame.draw.circle(screen, color, (fx, fy), 4)
        pygame.draw.circle(screen, (50, 200, 255), (tx, ty), 4)


def draw_robots(screen, env, bounds, width, height, font):
    node_map = {node.node_id: node for node in env.graph_state.nodes}

    for robot in env.robots:
        if not robot.telemetry:
            continue

        if robot.telemetry.current_node_index is not None and 0 <= robot.telemetry.current_node_index < len(env.graph_state.nodes):
            node = env.graph_state.nodes[robot.telemetry.current_node_index]
            wx, wy = _get_node_center(node)
        elif robot.telemetry.current_edge_index is not None and 0 <= robot.telemetry.current_edge_index < len(env.graph_state.edges):
            edge = env.graph_state.edges[robot.telemetry.current_edge_index]
            from_node = node_map.get(edge.from_node)
            to_node = node_map.get(edge.to_node)
            if from_node is not None and to_node is not None:
                fx, fy = _get_node_center(from_node)
                tx, ty = _get_node_center(to_node)
                p = float(np.clip(robot.telemetry.edge_progress, 0.0, 1.0))
                wx = fx + (tx - fx) * p
                wy = fy + (ty - fy) * p
            else:
                wx, wy = float(robot.telemetry.x), float(robot.telemetry.y)
        else:
            wx, wy = float(robot.telemetry.x), float(robot.telemetry.y)

        x, y = world_to_screen(wx, wy, bounds, width, height)

        # Draw the robot's actual planned movement path on the graph (if any).
        remaining_path = list(robot.telemetry.remaining_path or [])
        if remaining_path:
            path_points = [(x, y)]
            for node_idx in remaining_path:
                if 0 <= node_idx < len(env.graph_state.nodes):
                    node = env.graph_state.nodes[node_idx]
                    path_points.append(
                        world_to_screen(*_get_node_center(node), bounds, width, height)
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
        env.pending_tasks.sort(key=lambda t: t.arrival_time)
        for i, t in enumerate(env.pending_tasks):
            t.queue_position = i
            t.learned_score = 0.0
        return
    state_dict = env._get_state_dict()
    env.pending_tasks = ppo.score_and_rank_tasks(
        env.pending_tasks, state_dict, env.current_time, temperature=0.0
    )


def rescore_robot_queues(env, ppo):
    graph_emb = None
    fleet_emb = None
    if ppo is not None:
        state_dict = env._get_state_dict()
        state_tensor = ppo._state_dict_to_tensor(state_dict)
        with torch.no_grad():
            graph_emb, fleet_emb = ppo.policy.encode_context(state_tensor)

    for robot in env.robots:
        all_tasks = robot.task_queue[1:] + robot.overflow_queue
        if ppo is not None and all_tasks:
            task_features = np.stack([t.get_features(env.current_time) for t in all_tasks])
            task_features_tensor = torch.tensor(task_features, dtype=torch.float32).to(ppo.device)
            with torch.no_grad():
                scores = ppo.policy.score_tasks(task_features_tensor, graph_emb, fleet_emb).cpu().numpy()
            for task, score in zip(all_tasks, scores):
                task.learned_score = float(score)
            robot.resort_queue()
            robot.resort_overflow()
        elif ppo is None:
            # Keep queue ordering deterministic in heuristic simulation mode.
            robot.resort_queue()
            robot.resort_overflow()

        # Capacity/overflow behavior must hold regardless of PPO usage.
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

        # Apply capacity overflow rules immediately after insertion so a full robot
        # does not accumulate pickup legs in the main queue.
        if 0 <= action < len(env.robots):
            robot = env.robots[action]
            robot.enforce_capacity_limits()
            robot.promote_from_overflow()
    return assignments
