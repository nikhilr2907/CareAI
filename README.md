# CareRobotics

[Company description: to be written.]

This repository holds the task-allocation software for CareRobotics robots. This branch, `pilot_trial`, is configured for the Independent Learning Centre (Sixth Form) pilot. The `main` branch is configured for hospital environments.

## Overview

A multi-agent PPO policy (GAPO) assigns restocking and pickup tasks to robots over a graph of storage, shelf and corridor nodes. Demand follows a school-day schedule. The code runs in simulation and can drive a live robot through a ROS 2 bridge.

## Status

| Area | State |
| --- | --- |
| Simulation training (`run_training.py`) | Working |
| Simulation deployment (`run_deployment.py`) | Working, headless or with the pygame view |
| Real-robot training and deployment (`run_training_real.py`, `run_real_deployment.py`) | Not yet run against hardware. Requires the ROS 2 bridge. |
| Digital twin (`--twin`) | Implemented, not yet checked in a browser |
| Reliability layer | Phase 1 only (telemetry noise, `src/reliability/`) |

Simulation generates its own tasks and consumption from the config, so no training data is required.

## Setup

Requires Python 3.13. Dependencies are pinned in `requirements.txt`.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Usage

Train in simulation:

```bash
python run_training.py --config configs/ilc_pilot_v1_care_robotics.json \
    --iterations 10000 --device auto --seed 42 --output-dir outputs --exp-name pilot_sim
```

Run a trained policy headless in simulation:

```bash
python run_deployment.py --config configs/ilc_pilot_v1_care_robotics.json \
    --use-trained --checkpoint outputs/pilot_sim/checkpoints/gapo_final.pth \
    --headless --max-runtime 3600
```

Drop `--headless` for the pygame view. Add `--twin` to publish state to the digital twin (WebSocket on port 8766, browser page on 8767).

Run `python run_training.py --help` for the full list of training flags.

## Repository layout

```text
run_training.py              Simulation PPO training
run_deployment.py            Simulation inference, optional pygame view and twin
run_training_real.py         Training against a live robot via the ROS 2 bridge
run_real_deployment.py       Inference against a live robot
configs/                     Pilot layout and demand configuration
digital_twin/                Browser digital twin
src/environment/             Graph, robots, tasks and the environment
src/multi_agent_ppo/         Policy, encoders, attention, PPO, learned edge cost
src/deployment/              ROS bridge backend, WebSocket client, twin publisher
src/reliability/             Sim-to-real effects (telemetry noise)
src/analytics/               Realtime analytics and metrics
src/utils/                   Logging and helpers
```

## Configuration

This branch loads only pilot configs (`metadata.environment_type: ilc_open_space`). Hospital configs are in `configs/` but are rejected here. Use `main` for hospital runs.

The school day is defined in `school_schedule` (periods, operating days, and per-SKU demand profiles). Demand scales with each node's `foot_traffic_weight`.

## Real-robot operation

Live robots are reached over a WebSocket ROS 2 bridge. Set the URL in `.env` or pass `--bridge-url`:

```text
ROS_BRIDGE_URL=ws://<robot-host>:8765
```

[ROS bridge integration: to be written.]

## Outputs

Each run writes to `--output-dir/<exp-name>/`:

- `checkpoints/`: model checkpoints and training metrics
- `logs/`: task, decision, ranking and robot-sample logs
- `analytics/` and `metrics_final/`: plots and metric files

Deployment runs write to `outputs/`, which is git-ignored.

## Cluster training

`train_gpu.sh` is a Slurm job for a GPU cluster. Set the project path, output path and Python environment for your cluster before submitting it.

## Known limitations

- Real-robot operation has not been tested with hardware.
- Only telemetry noise from the reliability layer is implemented.
- The PPO core depends on school-period features and does not run on the hospital config on `main`.

## Contact

[Contact details: to be written.]
