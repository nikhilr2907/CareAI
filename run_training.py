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
from collections import deque

from src.environment.graph.config_loader import list_available_configs
from src.multi_agent_ppo.gapo_ppo import GAPOPPO, Memory
from src.utils.training_utils import (
    parse_args,
    load_curriculum_from_configs,
    create_env_from_config_file,
    setup_logging_and_output,
    log_gpu_memory,
    select_nearest_robot,
    apply_consumption_scale,
    run_evaluation,
)


def main():
    # Parse command-line arguments
    args = parse_args()

    # Setup logging and output directories
    logger, output_dir, exp_name = setup_logging_and_output(args)

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")

    ############## Load Configs ##############
    use_curriculum = args.curriculum != 'none'

    if not args.config and not args.config_list and not args.configs_dir:
        default_config = Path("configs") / "revised_hospital_config_v3.json"
        args.config = str(default_config)

    if args.config:
        # Single config mode
        if not Path(args.config).exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        curriculum_configs = [(args.config, Path(args.config).stem)]
        use_curriculum = False
        logger.info(f"Single-config mode: {args.config}")

    elif args.config_list:
        # Explicit list of configs for curriculum
        curriculum_configs = load_curriculum_from_configs(args.config_list)
        logger.info(f"Multi-config mode: {len(curriculum_configs)} configs specified")

    elif args.configs_dir:
        # Load all configs from directory
        available_configs = list_available_configs(args.configs_dir)
        if not available_configs:
            raise ValueError(f"No config files found in directory: {args.configs_dir}")
        curriculum_configs = load_curriculum_from_configs(available_configs)
        logger.info(f"Curriculum mode: {len(curriculum_configs)} configs from {args.configs_dir}")

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
    use_debiasing = not args.no_debiasing

    # PPO parameters
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

    # Training parameters
    max_training_iterations = args.iterations
    rollout_steps = args.rollout_steps
    timesteps_per_decision = 10.0
    save_interval = args.save_interval
    log_interval = args.log_interval
    max_assignments_per_step = args.max_assignments_per_step
    reward_clip = args.reward_clip
    reward_scale = args.reward_scale
    warmup_iters = args.warmup_iters
    warmup_mix = args.warmup_mix
    warmup_reward_offset = args.warmup_reward_offset
    warmup_penalty_scale = args.warmup_penalty_scale
    warmup_consumption_scale = args.warmup_consumption_scale
    warmup_entropy_mult = args.warmup_entropy_mult

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Use output directory from setup
    checkpoint_dir = output_dir / "checkpoints"

    logger.info("\n" + "=" * 80)
    logger.info("GAPO TRAINING - Hospital Robot Task Assignment")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"De-biasing Enabled: {use_debiasing}")
    logger.info(f"Hidden Dimension: {hidden_dim}")
    logger.info(f"Attention Heads: {num_attention_heads}")
    logger.info(f"Learning Rate: {lr}")
    logger.info(f"Actor LR: {actor_lr}")
    logger.info(f"Critic LR: {critic_lr}")
    logger.info(f"De-bias Lambda: {lambda_debias}")
    logger.info(f"Max Iterations: {max_training_iterations}")
    logger.info(f"Rollout Steps: {rollout_steps}")
    logger.info(f"Save Interval: {save_interval}")
    logger.info(f"Log Interval: {log_interval}")
    logger.info(f"Critic Coef: {critic_coef}")
    logger.info(f"Entropy Coef: {entropy_coef}")
    logger.info(f"Min Advantage Std: {min_adv_std}")
    logger.info(f"Warmup: iters={warmup_iters} mix={warmup_mix} reward_offset={warmup_reward_offset} penalty_scale={warmup_penalty_scale} consumption_scale={warmup_consumption_scale} entropy_mult={warmup_entropy_mult}")
    logger.info(f"Eval Interval: {args.eval_interval}")
    logger.info(f"Eval Steps: {args.eval_steps}")
    logger.info(f"Eval Seeds: {args.eval_seeds}")

    if use_curriculum:
        logger.info(f"\nCurriculum Learning: Enabled ({curriculum_schedule} schedule)")
        logger.info(f"  Stages: {len(curriculum_configs)} configurations")
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
    if getattr(env.graph_state, "category_order", None):
        base_node_dim = env.graph_state.get_node_features_with_category_stats()[0].shape[1]
        sku_feat_dim = env.graph_state.get_node_sku_features()[0].shape[2]
    else:
        base_node_dim = 8
        sku_feat_dim = None
    edge_feat_dim = env.graph_state.get_edge_features_complete()[0].shape[1]
    sku_embed_dim = 16
    node_continuous_dim = base_node_dim + (sku_embed_dim if sku_feat_dim is not None else 0)

    # Determine number of departments (if category stats are available)
    num_departments = len(env.graph_state.department_order) if hasattr(env.graph_state, 'department_order') and env.graph_state.department_order else 10

    # Create GAPO PPO with fine-grained categorical embeddings
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
        robot_feat_dim=20,
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
        device=device,
        logger=logger
    )
    policy_params = sum(p.numel() for p in ppo.policy.parameters())
    policy_old_params = sum(p.numel() for p in ppo.policy_old.parameters())
    edge_cost_params = 0
    if hasattr(env, "edge_cost_manager") and getattr(env.edge_cost_manager, "model", None) is not None:
        edge_cost_params = sum(p.numel() for p in env.edge_cost_manager.model.parameters())
    total_params = policy_params + policy_old_params + edge_cost_params
    trainable_params = sum(p.numel() for p in ppo.policy.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: policy={policy_params:,} "
        f"policy_old={policy_old_params:,} edge_cost={edge_cost_params:,} "
        f"total={total_params:,} trainable(policy)={trainable_params:,}"
    )

    # Training metrics
    memory = Memory()
    running_reward = 0
    iteration_rewards = []
    config_rewards = []
    total_timesteps = 0
    iterations_on_current_config = 0
    clip_fraction_history = deque(maxlen=50)
    grad_norm_history = deque(maxlen=50)
    entropy_history = deque(maxlen=50)
    completion_time_history = deque(maxlen=1000)

    logger.info("\nStarting Training...")
    logger.info("-" * 80)

    # Log initial GPU memory
    log_gpu_memory(logger)

    start_time = time.time()
    prev_process_time = time.process_time()

    # Initialize environment
    state_dict = env.reset()
    
    for iteration in range(1, max_training_iterations + 1):
        print(iteration)
        if iteration <= warmup_iters:
            apply_consumption_scale(env, warmup_consumption_scale)
            ppo.entropy_coef = entropy_coef * warmup_entropy_mult
        else:
            apply_consumption_scale(env, 1.0)
            ppo.entropy_coef = entropy_coef
        iterations_on_current_config += 1
        iteration_reward = 0.0
        num_assignments = 0
        iteration_start_time = time.time()
        buffer_size = 0
        iteration_completion_times = []
        
        # Collect rollout_steps timesteps of experience
        for step in range(rollout_steps):
            total_timesteps += 1
            
            # Re-rank pending tasks using learned scorer (manual priority as secondary)
            if env.pending_tasks:
                task_features = np.stack([t.get_features(env.current_time) for t in env.pending_tasks])
                task_features_tensor = torch.tensor(task_features, dtype=torch.float32).to(ppo.device)
                with torch.no_grad():
                    scores = ppo.policy_old.score_tasks(task_features_tensor).cpu().numpy()
                scored = list(zip(env.pending_tasks, scores))
                scored.sort(key=lambda x: (-x[1], -x[0].manual_priority))
                env.pending_tasks = [t for t, _ in scored]
                for i, (t, score) in enumerate(scored):
                    t.queue_position = i
                    t.learned_score = float(score)

            # Re-score and re-sort each robot's queued tasks (keep current task fixed)
            for robot in env.robots:
                all_tasks = robot.task_queue[1:] + robot.overflow_queue
                if all_tasks:
                    task_features = np.stack([t.get_features(env.current_time) for t in all_tasks])
                    task_features_tensor = torch.tensor(task_features, dtype=torch.float32).to(ppo.device)
                    with torch.no_grad():
                        scores = ppo.policy_old.score_tasks(task_features_tensor).cpu().numpy()
                    for task, score in zip(all_tasks, scores):
                        task.learned_score = float(score)
                    robot.resort_queue()
                    robot.resort_overflow()

                robot.enforce_capacity_limits()
                robot.promote_from_overflow()

            # Autoregressive task assignment phase
            assignments_this_step = 0
            while len(env.pending_tasks) > 0 and assignments_this_step < max_assignments_per_step:
                task = env.pending_tasks[0]

                # Build robot availability mask
                robot_mask = np.ones(env.num_robots, dtype=bool)

                # Select action
                if iteration <= warmup_iters and np.random.random() < warmup_mix:
                    action = select_nearest_robot(task, env.robots, env.graph_state, robot_mask)
                    state_tensor = ppo._state_dict_to_tensor(state_dict)
                    mask_tensor = torch.tensor(robot_mask, dtype=torch.bool).to(ppo.device)
                    with torch.no_grad():
                        logprob, _, _ = ppo.policy_old.evaluate_actions(
                            [state_tensor],
                            torch.tensor([action], dtype=torch.long).to(ppo.device),
                            [mask_tensor]
                        )
                    memory.state_dicts.append(state_dict)
                    memory.actions.append(action)
                    memory.logprobs.append(logprob.item())
                    memory.robot_masks.append(robot_mask)
                    ppo.policy.record_action(action)
                else:
                    action = ppo.select_action(state_dict, memory, robot_mask)

                # Assign task to robot
                success = env.assign_task_to_robot(action, task)
                if success:
                    num_assignments += 1
                    assignments_this_step += 1
                    reward = 0.0
                    state_dict = env._get_state_dict()
                else:
                    reward = -1.0
                    if iteration <= warmup_iters:
                        reward *= warmup_penalty_scale
                        reward += warmup_reward_offset

                iteration_reward += reward
                memory.rewards.append(reward)
                memory.is_terminals.append(False)

            # Simulation time step
            prev_completed = len(env.completed_tasks)
            state_dict, step_reward, done, info = env.step(dt=timesteps_per_decision)
            if reward_clip is not None and reward_clip > 0:
                step_reward = float(np.clip(step_reward, -reward_clip, reward_clip))
            if iteration <= warmup_iters:
                if step_reward < 0:
                    step_reward *= warmup_penalty_scale
                step_reward += warmup_reward_offset
            step_reward *= reward_scale
            if len(env.completed_tasks) > prev_completed:
                for task in env.completed_tasks[prev_completed:]:
                    if task.arrival_time is not None:
                        iteration_completion_times.append(env.current_time - task.arrival_time)

            if step_reward != 0:
                iteration_reward += step_reward
                if len(memory.rewards) > 0:
                    memory.rewards[-1] += step_reward

            if done:
                state_dict = env.reset()
            if (done or step == rollout_steps - 1) and len(memory.is_terminals) > 0:
                memory.is_terminals[-1] = True

        buffer_size = len(memory.actions)

        # Update policy
        next_state_dict = state_dict
        ppo.update(memory, next_state_dict)
        memory.clear_memory()
        # Batch by reshaping, batch by reshaping, batch by reshaping
        # Metrics
        iteration_time = time.time() - iteration_start_time
        process_time_now = time.process_time()
        cpu_time_delta = process_time_now - prev_process_time
        prev_process_time = process_time_now
        cpu_util = (cpu_time_delta / iteration_time * 100.0) if iteration_time > 0 else 0.0
        iteration_rewards.append(iteration_reward)
        config_rewards.append(iteration_reward)
        running_reward = 0.05 * iteration_reward + (1 - 0.05) * running_reward
        if iteration_completion_times:
            completion_time_history.extend(iteration_completion_times)

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
            logger.info(f"\n{'='*80}")
            logger.info(f"CURRICULUM ADVANCE: Moving to config {current_config_idx + 1}/{len(curriculum_configs)}")
            logger.info(f"  Previous config avg reward: {np.mean(config_rewards[-20:]):.2f}")
            logger.info(f"{'='*80}\n")

            env, current_config_desc = create_env_with_config(current_config_idx)
            state_dict = env.reset()

            config_rewards = []
            iterations_on_current_config = 0

        # Logging
        if iteration % log_interval == 0:
            avg_reward = np.mean(iteration_rewards[-log_interval:])
            loss_info = ppo.get_last_loss_info()

            logger.info(f"\nIteration {iteration}/{max_training_iterations} ({100*iteration/max_training_iterations:.1f}%)")
            if iteration <= warmup_iters:
                logger.info(f"  Warmup: active ({iteration}/{warmup_iters})")
            logger.info(f"  Buffer size (actions): {buffer_size}")
            if use_curriculum:
                logger.info(f"  Config: {current_config_idx + 1}/{len(curriculum_configs)} - {current_config_desc}")
                logger.info(f"  Iterations on config: {iterations_on_current_config}")
            logger.info(f"  Iteration Reward: {iteration_reward:.2f}")
            logger.info(f"  Avg Reward ({log_interval} iters): {avg_reward:.2f}")
            logger.info(f"  Running Reward: {running_reward:.2f}")
            logger.info(f"  Tasks Assigned: {num_assignments}")
            logger.info(f"  Total Timesteps: {total_timesteps}")
            logger.info(f"  Simulation Time: {env.current_time / 3600:.2f} hours")
            logger.info(f"  Pending: {len(env.pending_tasks)} | Completed: {len(env.completed_tasks)}")
            logger.info(f"  Iteration Time: {iteration_time:.2f}s")
            logger.info(f"  CPU Time: {cpu_time_delta:.2f}s | CPU Util (proc): {cpu_util:.1f}%")
            if iteration_completion_times:
                avg_iter_completion = float(np.mean(iteration_completion_times))
                logger.info(f"  Avg Completion Time (iter): {avg_iter_completion/60:.2f} min")
            if completion_time_history:
                avg_completion = float(np.mean(completion_time_history))
                logger.info(f"  Avg Completion Time (rolling): {avg_completion/60:.2f} min")

            idle_reasons = []
            for robot, simulator in zip(env.robots, env.robot_simulators):
                if robot.num_queued_tasks == 0:
                    continue
                if simulator.path_queue or simulator.current_target_node is not None:
                    continue
                current_task = robot.current_task
                if not current_task:
                    continue
                parent_id = getattr(current_task, "parent_task_id", None)
                pickup_done = robot.is_pickup_complete(parent_id) if parent_id is not None else None
                idle_reasons.append(
                    f"R{robot.robot_id} leg={getattr(current_task, 'leg_type', None)} "
                    f"parent={parent_id} pickup_done={pickup_done} "
                    f"cur_node={robot.current_node_index} target={simulator.current_target_node} "
                    f"pathq={len(simulator.path_queue)} q={robot.num_queued_tasks} ov={len(robot.overflow_queue)}"
                )
            if idle_reasons:
                logger.info("  Idle Diagnostics:")
                for line in idle_reasons[:10]:
                    logger.info(f"    {line}")

            if loss_info:
                logger.info(f"  Loss - Total: {loss_info.get('total_loss', 0):.4f} | Actor: {loss_info.get('actor_loss', 0):.4f} | Critic: {loss_info.get('critic_loss', 0):.4f}")
                if use_debiasing:
                    logger.info(f"  De-bias: {loss_info.get('debias_loss', 0):.4f} (delta: {loss_info.get('state_delta', 0):.4f}, cons: {loss_info.get('consistency', 0):.4f})")
                if 'clip_fraction' in loss_info:
                    logger.info(f"  PPO Clip Fraction: {loss_info.get('clip_fraction', 0):.4f}")
                if 'entropy_loss' in loss_info:
                    logger.info(f"  Entropy Loss: {loss_info.get('entropy_loss', 0):.4f}")
                if 'grad_norm' in loss_info:
                    logger.info(f"  Grad Norm: {loss_info.get('grad_norm', 0):.4f}")
                if use_debiasing and 'debias_sim_pairs' in loss_info:
                    logger.info(
                        f"  Debias Stats: steps={loss_info.get('debias_seq_len', 0)}, "
                        f"pairs={loss_info.get('debias_sim_pairs', 0)}, "
                        f"sim_mean={loss_info.get('debias_sim_mean', 0):.4f}, "
                        f"sim_max={loss_info.get('debias_sim_max', 0):.4f}"
                    )
                if 'clip_fraction' in loss_info:
                    clip_fraction_history.append(loss_info.get('clip_fraction', 0))
                if 'grad_norm' in loss_info:
                    grad_norm_history.append(loss_info.get('grad_norm', 0))
                if 'entropy_loss' in loss_info:
                    entropy_history.append(loss_info.get('entropy_loss', 0))

                if clip_fraction_history:
                    clip_mean = float(np.mean(clip_fraction_history))
                    if clip_mean < 0.01:
                        logger.warning(f"  Warning: clip_fraction mean very low ({clip_mean:.3f}) - policy update may be too small.")
                    elif clip_mean > 0.5:
                        logger.warning(f"  Warning: clip_fraction mean high ({clip_mean:.3f}) - policy updates may be too large.")

                if grad_norm_history:
                    grad_mean = float(np.mean(grad_norm_history))
                    if grad_mean < 1e-3:
                        logger.warning(f"  Warning: grad_norm mean very low ({grad_mean:.6f}) - possible vanishing gradients.")
                    elif grad_mean > 10.0:
                        logger.warning(f"  Warning: grad_norm mean high ({grad_mean:.3f}) - possible exploding gradients.")

                if entropy_history:
                    ent_mean = float(np.mean(entropy_history))
                    if ent_mean > -0.001:
                        logger.warning(f"  Warning: entropy loss near zero ({ent_mean:.4f}) - policy may be over-confident.")

            if args.eval_interval > 0 and iteration % args.eval_interval == 0:
                eval_metrics = []
                baseline_metrics = []
                for seed in args.eval_seeds:
                    eval_env, _ = create_env_with_config(current_config_idx)
                    metrics = run_evaluation(
                        eval_env,
                        args.eval_steps,
                        seed,
                        ppo,
                        max_assignments_per_step,
                        timesteps_per_decision,
                        reward_clip,
                        use_baseline=False
                    )
                    eval_metrics.append(metrics)
                    baseline_env, _ = create_env_with_config(current_config_idx)
                    baseline = run_evaluation(
                        baseline_env,
                        args.eval_steps,
                        seed,
                        ppo,
                        max_assignments_per_step,
                        timesteps_per_decision,
                        reward_clip,
                        use_baseline=True
                    )
                    baseline_metrics.append(baseline)
                if eval_metrics:
                    avg_eval_reward = float(np.mean([m["reward"] for m in eval_metrics]))
                    avg_eval_completed = float(np.mean([m["completed"] for m in eval_metrics]))
                    avg_eval_pending = float(np.mean([m["pending"] for m in eval_metrics]))
                    avg_eval_hours = float(np.mean([m["sim_hours"] for m in eval_metrics]))
                    avg_eval_on_time = float(np.mean([m["completed_on_time"] for m in eval_metrics]))
                    avg_eval_late = float(np.mean([m["completed_late"] for m in eval_metrics]))
                    avg_eval_late_assign = float(np.mean([
                        (m["late_at_assignment"] / max(m["assigned_total"], 1)) for m in eval_metrics
                    ]))
                    rewards = np.array([m["reward"] for m in eval_metrics])
                    p10 = float(np.percentile(rewards, 10))
                    p50 = float(np.percentile(rewards, 50))
                    p90 = float(np.percentile(rewards, 90))
                    baseline_reward = float(np.mean([m["reward"] for m in baseline_metrics])) if baseline_metrics else 0.0
                    reward_delta = avg_eval_reward - baseline_reward
                    logger.info(
                        f"  Eval (fixed seeds): reward={avg_eval_reward:.2f} "
                        f"completed={avg_eval_completed:.1f} pending={avg_eval_pending:.1f} "
                        f"on_time={avg_eval_on_time:.1f} late={avg_eval_late:.1f} "
                        f"late_at_assign={avg_eval_late_assign:.3f} sim_hours={avg_eval_hours:.2f}"
                    )
                    logger.info(
                        f"  Eval Reward Dist: p10={p10:.2f} p50={p50:.2f} p90={p90:.2f} "
                        f"baseline={baseline_reward:.2f} delta={reward_delta:.2f}"
                    )

            # Log GPU memory periodically
            if iteration % (log_interval * 10) == 0:
                log_gpu_memory(logger)

            logger.info("-" * 80)

        # Save model
        if iteration % save_interval == 0:
            save_path = checkpoint_dir / f"gapo_iter{iteration}.pth"
            ppo.save(str(save_path))
            logger.info(f"Model saved to {save_path}")

    total_time = time.time() - start_time

    logger.info("\n" + "=" * 80)
    logger.info("Training Completed!")
    logger.info("=" * 80)
    logger.info(f"Total Training Time: {total_time / 3600:.2f} hours")
    logger.info(f"Total Timesteps: {total_timesteps}")
    logger.info(f"Avg Iteration Reward: {np.mean(iteration_rewards):.2f}")
    logger.info(f"Final Running Reward: {running_reward:.2f}")
    logger.info(f"Final Simulation Time: {env.current_time / 3600:.2f} hours")
    logger.info(f"Total Completed Tasks: {len(env.completed_tasks)}")
    logger.info("=" * 80)

    # Save final model
    final_model_path = checkpoint_dir / "gapo_final.pth"
    ppo.save(str(final_model_path))
    logger.info(f"\nFinal model saved to {final_model_path}")

    # Save training metrics
    metrics_path = checkpoint_dir / "training_metrics.npz"
    np.savez(
        metrics_path,
        iteration_rewards=np.array(iteration_rewards),
        running_reward=running_reward,
        total_timesteps=total_timesteps
    )
    logger.info(f"Training metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()
