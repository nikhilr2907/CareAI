# CareRobotics: ILC Pilot (`pilot_trial`)

Multi-agent PPO task allocation for a single restocking robot at the **Independent Learning Centre (Sixth Form)**, pilot window **June 6–19 2026**. The policy assigns restocking and pickup tasks to robots over a small graph of storage, shelf and corridor nodes, using a school-day demand schedule.

This branch carries the ILC layout. The layout-agnostic code is shared with `main`, which runs the hospital environment.

## Status

| Area | State |
|---|---|
| Simulation training (`run_training.py`) | Works. Verified with a 2-iteration smoke run. |
| Simulation deployment (`run_deployment.py`) | Works headless, with or without a trained checkpoint. |
| Real-robot training and deployment (`run_training_real.py`, `run_real_deployment.py`) | Imports and CLI verified. **Not run against hardware.** Needs the ROS 2 bridge. |
| Digital twin (`--twin` flag) | Code present. Not verified in a browser. |
| Reliability layer | Phase 1 only: telemetry noise (`src/reliability/telemetry_effects.py`), enabled in `gapo_env.py`. Stale-state handling, movement failures, network effects and battery-model phases are **not implemented**. |
| Tests | Removed from this branch by request. |

**No training data is needed.** Simulation generates its own tasks and consumption from the config. Real-robot runs take live telemetry and inventory from the ROS bridge, not from files.

## Quick start

Python 3.13 was used for verification. `requirements.txt` pins PyTorch 2.9.0 and torch-geometric 2.7.0.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

**Train in simulation** (defaults to `configs/ilc_pilot_v1_care_robotics.json`):

```bash
python run_training.py --config configs/ilc_pilot_v1_care_robotics.json \
    --iterations 10000 --device auto --seed 42 --output-dir outputs --exp-name pilot_sim
```

**Run the trained policy headless in simulation:**

```bash
python run_deployment.py --config configs/ilc_pilot_v1_care_robotics.json \
    --use-trained --checkpoint outputs/pilot_sim/checkpoints/gapo_final.pth \
    --headless --max-runtime 3600
```

Drop `--headless` to get the pygame view. Add `--twin` to publish state to the digital twin (WebSocket on 8766, browser page on 8767).

## Repository layout

```
run_training.py              Simulation PPO training (curriculum, eval, checkpoints)
run_deployment.py            Simulation inference, optional pygame view and twin
run_training_real.py         Training against a live robot via the ROS 2 bridge (1 Hz pacing)
run_real_deployment.py       Greedy inference against a live robot
configs/                     Layout and demand configs (ILC pilot + earlier hospital variants)
digital_twin/                Browser twin (index.html, graph.json)
src/environment/             Graph, robots, tasks, Gym-style env (gapo_env.py, gapo_env_real.py)
src/multi_agent_ppo/         Policy, GNN encoders, attention, PPO, debiasing, learned edge cost
src/deployment/              ROS bridge backend, WebSocket client, state sync, twin publisher
src/reliability/             Sim-to-real effects (Phase 1 telemetry noise)
src/analytics/               Realtime analytics collector and metrics
src/utils/                   Logging (tasks, decisions, ranking, robot samples, fleet events),
                             training and deployment helpers
```

## Configuration

This branch loads only ILC pilot configs (`metadata.environment_type: ilc_open_space`). The loader rejects anything else.

| File | Purpose |
|---|---|
| `ilc_pilot_v1_care_robotics.json` | The pilot config: 1 robot, capacity 2, 3 nodes, 4 SKUs, 8.5-hour school day |

The hospital configs (`hospital_3nodes_consumables.json`, `revised_hospital_config_v3*.json`) are still in `configs/` but this branch rejects them. Use `main` for hospital runs.

The school day is in `school_schedule` (periods, `operating_days`, per-SKU `school_period_profile`). Demand is scaled by each node's `foot_traffic_weight`.

Key training flags: `--iterations`, `--rollout-steps`, `--lr`, `--hidden-dim`, `--curriculum {none,fixed,adaptive,random}` with `--configs-dir` or `--config-list`, `--max-assignments-per-step`, `--seed`, `--output-dir`, `--exp-name`. Run `python run_training.py --help` for the full list.

## Real-robot operation

The robot is reached over a WebSocket ROS 2 bridge. Set the URL in a `.env` file, or pass `--bridge-url`:

```
ROS_BRIDGE_URL=ws://<robot-host>:8765
```

The code default is `ws://localhost:8765`. Note that the `ROSBridgeRobotBackend` docstring still says 9090. The code is authoritative.

```bash
python run_training_real.py --config configs/ilc_pilot_v1_care_robotics.json \
    --bridge-url ws://<robot-host>:8765
python run_real_deployment.py --config configs/ilc_pilot_v1_care_robotics.json \
    --use-trained --checkpoint outputs/<exp>/checkpoints/gapo_real_final.pth \
    --bridge-url ws://<robot-host>:8765 --seed 42
```

The ROS 2 side of the bridge is not in this repository. Before a real run, check the bridge's message schema against `src/deployment/robot_bridge_websocket_client.py` and `robot_backend.py`. The bridge has not been exercised with hardware.

## Outputs

Each run writes under `--output-dir/<exp-name>/`:

- `checkpoints/` (`gapo_iter<N>.pth`, `gapo_final.pth`, `training_metrics.npz`)
- `logs/` (task, decision, ranking and robot-sample logs)
- `analytics/` and `metrics_final/` (plots and metric JSON)

Deployment runs write to `outputs/deploy_<config>_<timestamp>/`. The `outputs/` folder is git-ignored.

## Cluster training

`train_gpu.sh` is a Slurm job for the Imperial College GPU cluster. It runs `run_training.py` on the hospital config and expects a Python venv at a path specific to that cluster. Adjust `PROJECT_DIR`, `OUTPUT_DIR` and the `source` line before submitting. `.github/workflows/deploy.yml` rsyncs the repo to the cluster on pushes to `main`. This branch is not deployed by that workflow.

## Known gaps

- Real-robot scripts are untested against hardware and need the ROS 2 bridge.
- Reliability effects beyond telemetry noise (Phases 2–3) are not implemented, so simulation is cleaner than the real robot will be.
- The PPO core (`gapo_ppo.py`, `gapo_gnn_encoders.py`) is built around the school-period features. It does not run on the hospital config from `main`.
- Edge-cost learning (`learned_edge_cost.py`) trains on linear-congestion simulator records. Real traversal times will be multimodal, so expect mispredictions until it is retrained on real records.
