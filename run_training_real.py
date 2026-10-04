"""
GAPO Real-Robot Training Script.

Trains the hospital robot task allocation policy against a live robot via
ROS bridge. Identical PPO/memory/reward-routing logic to run_training.py
with the following differences:

  - GAPOTaskAssignmentEnvReal replaces GAPOTaskAssignmentEnv
  - ROSBridgeRobotBackend started as async background task
  - Each step paced to 1 Hz wall clock (await asyncio.sleep)
  - No curriculum / config switching
  - No apply_consumption_scale (bridge provides real inventory)
  - No episode resets — continuous operation; rollout boundaries by iteration
  - No run_evaluation (no seeded reset on real hardware)

Usage:
    python run_training_real.py --config configs/hospital_3nodes_consumables.json
    python run_training_real.py --config configs/... --bridge-url ws://192.168.1.50:8765
    python run_training_real.py --config configs/... --checkpoint outputs/.../gapo_final.pth
"""
import asyncio
import torch
import numpy as np
from pathlib import Path
import time
from collections import deque

from src.multi_agent_ppo.gapo_ppo import GAPOPPO, Memory
from src.deployment.robot_backend import ROSBridgeRobotBackend
from src.utils.training_utils import (
    parse_args,
    create_real_env_from_config_file,
    setup_logging_and_output,
    log_gpu_memory,
    select_nearest_robot,
)
from src.utils.task_logger import TaskLogger
from src.utils.policy_inspector import PolicyInspector
from src.analytics.realtime_collector import RealtimeAnalyticsCollector


async def main():
    args = parse_args()

    logger, output_dir, exp_name = setup_logging_and_output(args)
    task_logger = TaskLogger(output_dir / "logs")
    analytics = RealtimeAnalyticsCollector(
        output_dir=output_dir / "analytics",
        mode='real',
        snapshot_frequency=50,
        task_logger=task_logger,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # --- Config (single only, no curriculum) ---
    if not args.config:
        raise ValueError("--config is required for real training")
    config_path = args.config
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # --- Hyperparameters (same as run_training.py) ---
    num_robots = args.num_robots
    max_episode_time = args.max_episode_time
    timestep_seconds = 1.0
    stochastic_tasks_per_hour = args.stochastic_tasks_per_hour
    stochastic_task_cap_per_hour = args.stochastic_task_cap_per_hour
    initial_stochastic_tasks = args.initial_stochastic_tasks

    hidden_dim = args.hidden_dim
    num_attention_heads = 4
    use_debiasing = False

    lr = args.lr
    actor_lr = args.actor_lr if args.actor_lr is not None else lr * 1.2
    critic_lr = args.critic_lr if args.critic_lr is not None else lr * 0.5
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2
    lambda_debias = 0.1
    lambda_gae = 0.95
    critic_coef = args.critic_coef
    entropy_coef = args.entropy_coef
    min_adv_std = args.min_adv_std

    max_training_iterations = args.iterations
    rollout_steps = args.rollout_steps
    save_interval = args.save_interval
    log_interval = args.log_interval
    max_assignments_per_step = args.max_assignments_per_step
    reward_clip = args.reward_clip
    reward_scale = args.reward_scale
    warmup_iters = args.warmup_iters
    warmup_mix = args.warmup_mix
    warmup_entropy_mult = args.warmup_entropy_mult
    ranking_max_pairs_per_group = args.ranking_max_pairs_per_group
    ranking_min_adv_gap = args.ranking_min_adv_gap

    OPEN_ASSIGNMENT_TTL = 5

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    checkpoint_dir = output_dir / "checkpoints"

    # --- Start robot backend ---
    # bridge_url=None → backend reads ROS_BRIDGE_URL from .env, falls back to ws://localhost:9090
    robot_backend = ROSBridgeRobotBackend(
        bridge_url=args.bridge_url,
        num_robots=num_robots if num_robots else 1,
    )

    logger.info("=" * 80)
    logger.info("GAPO REAL-ROBOT TRAINING")
    logger.info("=" * 80)
    logger.info(f"Config:      {config_path}")
    logger.info(f"Bridge URL:  {robot_backend._bridge_url}")
    logger.info(f"Device:      {device}")
    logger.info(f"LR:          {lr}  Actor: {actor_lr}  Critic: {critic_lr}")
    logger.info(f"Iterations:  {max_training_iterations}  Rollout: {rollout_steps} steps")
    logger.info(f"Warmup:      {warmup_iters} iters  mix={warmup_mix}  ent_mult={warmup_entropy_mult}")
    logger.info("=" * 80)

    await robot_backend.start()
    logger.info("ROS bridge backend started")

    # --- Create environment ---
    env, num_nodes = create_real_env_from_config_file(
        config_path,
        robot_backend=robot_backend,
        num_robots=num_robots,
        max_episode_time=max_episode_time,
        timestep_seconds=timestep_seconds,
        stochastic_tasks_per_hour=stochastic_tasks_per_hour,
        stochastic_task_cap_per_hour=stochastic_task_cap_per_hour,
        initial_stochastic_tasks=initial_stochastic_tasks,
    )
    logger.info(f"Nodes: {num_nodes}  Robots: {env.num_robots}  Edges: {len(env.graph_state.edges)}")

    # --- Build PPO model ---
    if getattr(env.graph_state, "category_order", None):
        base_node_dim = env.graph_state.get_node_features_with_category_stats()[0].shape[1]
        sku_feat_dim = env.graph_state.get_node_sku_features()[0].shape[2]
    else:
        base_node_dim = 8
        sku_feat_dim = None
    edge_feat_dim = env.graph_state.get_edge_features_complete()[0].shape[1]
    sku_embed_dim = 16
    node_continuous_dim = base_node_dim + (sku_embed_dim if sku_feat_dim is not None else 0)
    num_departments = (
        len(env.graph_state.department_order)
        if hasattr(env.graph_state, 'department_order') and env.graph_state.department_order
        else 10
    )

    ppo = GAPOPPO(
        node_continuous_dim=node_continuous_dim,
        num_node_types=4,
        num_departments=num_departments,
        num_shift_periods=4,
        num_day_types=2,
        edge_feat_dim=edge_feat_dim,
        node_type_embedding_dim=8,
        department_embedding_dim=16,
        shift_embedding_dim=4,
        day_type_embedding_dim=4,
        robot_feat_dim=19,
        task_feat_dim=15,
        queue_feat_dim=16,
        sku_feat_dim=sku_feat_dim,
        sku_embed_dim=sku_embed_dim,
        hidden_dim=hidden_dim,
        num_attention_heads=num_attention_heads,
        lr=lr,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        use_debiasing=use_debiasing,
        lambda_debias=lambda_debias,
        lambda_gae=lambda_gae,
        critic_coef=critic_coef,
        entropy_coef=entropy_coef,
        min_adv_std=min_adv_std,
        ranking_max_pairs_per_group=ranking_max_pairs_per_group,
        ranking_min_adv_gap=ranking_min_adv_gap,
        task_creation_actor=getattr(env, 'task_creation_actor', None),
        device=device,
        logger=logger,
    )

    # Load checkpoint if provided (fine-tuning from sim-trained weights)
    if args.checkpoint and Path(args.checkpoint).exists():
        ppo.load(args.checkpoint)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    policy_inspector = PolicyInspector(ppo.policy, device=device)

    # --- Training state ---
    memory = Memory()
    task_to_memory_idx: dict = {}
    open_assignments: dict = {}
    closed_buffer: list = []
    running_reward = 0.0
    iteration_rewards = []
    total_timesteps = 0
    clip_fraction_history = deque(maxlen=50)
    grad_norm_history = deque(maxlen=50)
    entropy_history = deque(maxlen=50)
    completion_time_history = deque(maxlen=1000)

    log_gpu_memory(logger)
    state_dict = env.reset()
    start_time = time.time()
    prev_process_time = time.process_time()

    logger.info("Starting real-robot training...")
    logger.info("-" * 80)

    for iteration in range(1, max_training_iterations + 1):

        if iteration <= warmup_iters:
            ppo.entropy_coef = entropy_coef * warmup_entropy_mult
        else:
            ppo.entropy_coef = entropy_coef

        iteration_reward = 0.0
        num_assignments = 0
        iteration_start_time = time.time()
        buffer_size = 0
        iteration_completion_times = []

        # ---- Rollout ----
        for step in range(rollout_steps):
            total_timesteps += 1
            step_wall_start = time.monotonic()

            # Re-rank pending tasks
            if env.pending_tasks:
                scorer_temp = max(0.5, 1.0 - iteration / max(max_training_iterations, 1))
                env.pending_tasks = ppo.score_and_rank_tasks(
                    env.pending_tasks, state_dict, env.current_time,
                    temperature=scorer_temp,
                )

            # Re-score robot queues
            for robot in env.robots:
                all_tasks = robot.task_queue[1:] + robot.overflow_queue
                if all_tasks:
                    task_features = np.stack([t.get_features(env.current_time) for t in all_tasks])
                    task_features_tensor = torch.tensor(task_features, dtype=torch.float32).to(ppo.device)
                    state_tensor = ppo._state_dict_to_tensor(state_dict)
                    with torch.no_grad():
                        graph_emb, fleet_emb = ppo.policy_old.encode_context(state_tensor)
                        scores = ppo.policy_old.score_tasks(
                            task_features_tensor, graph_emb, fleet_emb
                        ).cpu().numpy()
                    for task, score in zip(all_tasks, scores):
                        task.learned_score = float(score)
                    robot.resort_queue()
                    robot.resort_overflow()
                robot.enforce_capacity_limits()
                robot.promote_from_overflow()

            # Task assignment
            assignments_this_step = 0
            step_memory_indices = []
            while len(env.pending_tasks) > 0 and assignments_this_step < max_assignments_per_step:
                task = env.pending_tasks[0]
                robot_mask = np.ones(env.num_robots, dtype=bool)
                mem_idx_before = len(memory.actions)

                try:
                    if iteration <= warmup_iters and np.random.random() < warmup_mix:
                        action = select_nearest_robot(task, env.robots, env.graph_state, robot_mask)
                        state_tensor = ppo._state_dict_to_tensor(state_dict)
                        mask_tensor = torch.tensor(robot_mask, dtype=torch.bool).to(ppo.device)
                        with torch.no_grad():
                            logprob, _, _, _, _, _ = ppo.policy_old.evaluate_actions(
                                [state_tensor],
                                torch.tensor([action], dtype=torch.long).to(ppo.device),
                                [mask_tensor],
                            )
                        memory.state_dicts.append(state_dict)
                        memory.actions.append(action)
                        memory.logprobs.append(logprob.item())
                        memory.robot_masks.append(robot_mask)
                        ppo.policy.record_action(action)
                    else:
                        action = ppo.select_action(state_dict, memory, robot_mask)
                except Exception as e:
                    logger.error(f"Action selection error iter {iteration} step {step}: {e}")
                    raise

                step_memory_indices.append(mem_idx_before)
                success = env.assign_task_to_robot(action, task)
                if not success:
                    memory.rewards.append(0.0)
                    memory.is_terminals.append(False)
                    break

                num_assignments += 1
                assignments_this_step += 1
                task_to_memory_idx[task.task_id] = mem_idx_before

                feasibility = (task.deadline - env.current_time) / max(task.estimated_duration, 1.0) - 1.0
                assignment_reward = 0.5 * max(-1.0, min(1.0, feasibility))
                open_assignments[task.task_id] = {
                    'state':             state_dict,
                    'action':            action,
                    'logprob':           memory.logprobs[mem_idx_before],
                    'mask':              robot_mask,
                    'reward':            0.0,
                    'assignment_reward': assignment_reward,
                    'born_iter':         iteration,
                }

                task_logger.log_assignment(
                    task, action, assignment_reward,
                    iteration, env.current_time, num_assignments, len(memory.actions),
                )
                state_dict = env._get_state_dict()
                memory.rewards.append(0.0)
                memory.is_terminals.append(False)

            if len(step_memory_indices) > 1:
                memory.step_assignment_groups.append(step_memory_indices)

            # Env step
            prev_completed = len(env.completed_tasks)
            state_dict, step_reward, done, info = env.step(dt=1.0)

            newly_completed = env.completed_tasks[prev_completed:]
            for task in newly_completed:
                if task.arrival_time is not None:
                    iteration_completion_times.append(env.current_time - task.arrival_time)

            task_completion_credits: dict = info.get('task_completion_credits', {})

            for parent_id, bonus in task_completion_credits.items():
                scaled_bonus = bonus * reward_scale
                if reward_clip is not None and reward_clip > 0:
                    scaled_bonus = float(np.clip(scaled_bonus, -reward_clip, reward_clip))

                completed_task = next(
                    (t for t in newly_completed if t.task_id == parent_id), None
                )
                actual_completion_time = None
                if completed_task is not None and completed_task.arrival_time is not None:
                    actual_completion_time = env.current_time - completed_task.arrival_time

                task_logger.log_completion(
                    parent_id, scaled_bonus, actual_completion_time, env.current_time,
                    distance_traveled=0.0, energy_consumed=0.0,
                )

                if env.task_creation_actor is not None:
                    env.task_creation_actor.record_completion(parent_id, scaled_bonus)
                iteration_reward += scaled_bonus

                orig_idx = task_to_memory_idx.get(parent_id)
                if orig_idx is not None and orig_idx < len(memory.rewards):
                    entry = open_assignments.pop(parent_id, None)
                    a_rew = entry['assignment_reward'] if entry else 0.0
                    memory.rewards[orig_idx] = a_rew + scaled_bonus
                else:
                    entry = open_assignments.pop(parent_id, None)
                    if entry is not None:
                        entry['reward'] = entry['assignment_reward'] + scaled_bonus
                        closed_buffer.append(entry)

            battery_pen = info.get('battery_penalty', 0.0)
            if battery_pen != 0.0 and step_memory_indices:
                share = battery_pen / len(step_memory_indices)
                for midx in step_memory_indices:
                    if midx < len(memory.rewards):
                        memory.rewards[midx] += share

            # Continuous operation: clear per-episode tracking at boundary, no env.reset()
            if done:
                task_to_memory_idx.clear()
                open_assignments.clear()
                task_logger.task_metadata.clear()
                if memory.is_terminals:
                    memory.is_terminals[-1] = True

            # Pace to 1 Hz — yields control so backend WebSocket tasks can process messages
            elapsed = time.monotonic() - step_wall_start
            await asyncio.sleep(max(0.0, 1.0 - elapsed))

        # ---- Inject cross-rollout completions ----
        for entry in closed_buffer:
            if iteration - entry['born_iter'] <= OPEN_ASSIGNMENT_TTL:
                memory.state_dicts.append(entry['state'])
                memory.actions.append(entry['action'])
                memory.logprobs.append(entry['logprob'])
                memory.rewards.append(entry['reward'])
                memory.is_terminals.append(True)
                memory.robot_masks.append(entry['mask'])
        closed_buffer.clear()

        # ---- Prune unresolved open assignments from memory ----
        if open_assignments and task_to_memory_idx:
            pending_mem_indices = {
                task_to_memory_idx[tid]
                for tid in open_assignments
                if tid in task_to_memory_idx
            }
            if pending_mem_indices:
                keep = [i for i in range(len(memory.actions)) if i not in pending_mem_indices]
                idx_remap = {old: new for new, old in enumerate(keep)}
                memory.state_dicts = [memory.state_dicts[i] for i in keep]
                memory.actions = [memory.actions[i] for i in keep]
                memory.logprobs = [memory.logprobs[i] for i in keep]
                memory.rewards = [memory.rewards[i] for i in keep]
                memory.is_terminals = [memory.is_terminals[i] for i in keep]
                memory.robot_masks = [memory.robot_masks[i] for i in keep]
                new_groups = []
                for group in memory.step_assignment_groups:
                    new_group = [idx_remap[i] for i in group if i in idx_remap]
                    if len(new_group) > 1:
                        new_groups.append(new_group)
                memory.step_assignment_groups = new_groups

        buffer_size = len(memory.actions)

        # ---- PPO update ----
        if buffer_size < 2:
            logger.info(f"  Skipping PPO update: buffer too small ({buffer_size})")
        else:
            try:
                ppo.update(memory, state_dict)
            except Exception as e:
                logger.error(f"PPO update error iter {iteration}: {e}")
                raise

        memory.clear_memory()
        task_to_memory_idx.clear()
        stale_ids = [
            pid for pid, e in open_assignments.items()
            if iteration - e['born_iter'] > OPEN_ASSIGNMENT_TTL
        ]
        for pid in stale_ids:
            open_assignments.pop(pid)

        # ---- Metrics ----
        iteration_time = time.time() - iteration_start_time
        process_time_now = time.process_time()
        cpu_time_delta = process_time_now - prev_process_time
        prev_process_time = process_time_now
        cpu_util = (cpu_time_delta / iteration_time * 100.0) if iteration_time > 0 else 0.0

        iteration_rewards.append(iteration_reward)
        running_reward = 0.05 * iteration_reward + 0.95 * running_reward
        if iteration_completion_times:
            completion_time_history.extend(iteration_completion_times)

        loss_info_current = ppo.get_last_loss_info()
        if loss_info_current:
            logger.info(
                f"[ITER {iteration}] Reward={iteration_reward:.2f} | "
                f"Actor={loss_info_current.get('actor_loss', 0):.4f} | "
                f"Clip={loss_info_current.get('clip_fraction', 0):.4f} | "
                f"GradNorm={loss_info_current.get('grad_norm', 0):.2f} | "
                f"Assigned={num_assignments} Pending={len(env.pending_tasks)}"
            )

        task_logger.log_running_summary(iteration)

        if iteration % log_interval == 0:
            avg_reward = np.mean(iteration_rewards[-log_interval:])
            loss_info = ppo.get_last_loss_info()

            logger.info(f"\nIteration {iteration}/{max_training_iterations}")
            if iteration <= warmup_iters:
                logger.info(f"  Warmup: active ({iteration}/{warmup_iters})")
            logger.info(f"  Buffer size: {buffer_size}")
            logger.info(f"  Iteration Reward: {iteration_reward:.2f}")
            logger.info(f"  Avg Reward ({log_interval} iters): {avg_reward:.2f}")
            logger.info(f"  Running Reward: {running_reward:.2f}")
            logger.info(f"  Tasks Assigned: {num_assignments}")
            logger.info(f"  Total Timesteps: {total_timesteps}")
            logger.info(f"  Wall Time: {env.current_time / 3600:.2f} hours")
            logger.info(f"  Pending: {len(env.pending_tasks)} | Completed: {len(env.completed_tasks)}")
            logger.info(f"  Iteration Time: {iteration_time:.2f}s  CPU Util: {cpu_util:.1f}%")

            if iteration_completion_times:
                logger.info(
                    f"  Avg Completion Time (iter): "
                    f"{float(np.mean(iteration_completion_times)) / 60:.2f} min"
                )
            if completion_time_history:
                logger.info(
                    f"  Avg Completion Time (rolling): "
                    f"{float(np.mean(completion_time_history)) / 60:.2f} min"
                )

            if loss_info:
                logger.info(
                    f"  Loss - Total: {loss_info.get('total_loss', 0):.4f} | "
                    f"Actor: {loss_info.get('actor_loss', 0):.4f} | "
                    f"Critic: {loss_info.get('critic_loss', 0):.4f} | "
                    f"Ranking: {loss_info.get('ranking_loss', 0):.4f}"
                )
                if 'clip_fraction' in loss_info:
                    clip_fraction_history.append(loss_info['clip_fraction'])
                if 'grad_norm' in loss_info:
                    grad_norm_history.append(loss_info['grad_norm'])
                if 'entropy_loss' in loss_info:
                    entropy_history.append(loss_info['entropy_loss'])

                try:
                    policy_inspector.log_learning_state(logger, loss_info, iteration)
                except Exception as e:
                    logger.warning(f"  Policy inspection failed: {e}")

                if clip_fraction_history:
                    clip_mean = float(np.mean(clip_fraction_history))
                    if clip_mean < 0.01:
                        logger.warning(f"  Warning: clip_fraction low ({clip_mean:.3f})")
                    elif clip_mean > 0.5:
                        logger.warning(f"  Warning: clip_fraction high ({clip_mean:.3f})")
                if grad_norm_history:
                    grad_mean = float(np.mean(grad_norm_history))
                    if grad_mean < 1e-3:
                        logger.warning(f"  Warning: grad_norm very low ({grad_mean:.6f})")
                    elif grad_mean > 10.0:
                        logger.warning(f"  Warning: grad_norm high ({grad_mean:.3f})")

            if hasattr(env, 'edge_cost_manager'):
                ec = env.edge_cost_manager.get_stats()
                model_status = 'learned' if ec['using_learned_model'] else 'heuristic'
                logger.info(
                    f"  EdgeCost [{model_status}]: buffer={ec['buffer_size']} "
                    f"records={ec['total_records']} nll_loss={ec['avg_recent_loss']:.4f}"
                )

        # ---- Checkpoint ----
        if iteration % save_interval == 0:
            ckpt_path = checkpoint_dir / f"gapo_real_iter{iteration}.pth"
            ppo.save(str(ckpt_path))
            ppo.save(str(checkpoint_dir / "gapo_real_latest.pth"))
            logger.info(f"  Checkpoint saved: {ckpt_path}")

    # ---- Final checkpoint + teardown ----
    ppo.save(str(checkpoint_dir / "gapo_real_final.pth"))
    logger.info(f"\nTraining complete. Total wall time: {(time.time() - start_time) / 3600:.2f}h")
    await robot_backend.stop()


if __name__ == '__main__':
    asyncio.run(main())
