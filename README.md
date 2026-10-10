<p align="center">
  <img src="assets/logo.png" alt="CareRobotics logo" width="120">
</p>

# CareRobotics

Website: [thecarerobotics.com](https://www.thecarerobotics.com/)

CareRobotics is building the automation layer that gives hospital staff their time back. Ward nurses spend a large part of each shift searching for supplies and equipment, and missing stock delays care. CareRobotics is developing robots that restock wards, with the task-allocation software in this repository deciding which robot does which job. The company is supported by Imperial College London and UCL through the Imperial VCC Accelerator and the UCL Venture Builder. A full-size prototype is in development, and a hospital pilot is under discussion with HCA UK.

This repository holds the task-allocation and fleet orchestration software/model code for CareRobotics robots. This branch, `pilot_trial`, is configured for the inaugural pilot, aiming to replicate the replenishment logistics in a hospital in a school environment. The `main` branch is configured for tests in hospital environments, the end goal being a fully functional multi-robot system that can revolutionise patient care in hospitals, by saving clinical staff time previously attributed to laborious manual replenishment and other logistics tasks.

## Overview

A multi-agent PPO policy (GAPO) decides which robot takes each restocking task. The environment simulates shelves consuming stock, raises a replenishment task when a shelf falls to its reorder point, and moves robots along a graph of storage and shelf nodes to pick up and deliver. The same code can drive a live robot through a ROS 2 bridge.

Both branches run the same learning and environment code. They differ only in the layout and demand layer: the config format and loader, the graph and task features, the demand model, and a few layout-specific rewards. See [Environments](#environments).

## Architecture

```mermaid
flowchart LR
    CFG[Config JSON<br/>layout, SKUs, demand] --> GS[GraphState<br/>nodes, edges, features]
    GS --> ENV[GAPOTaskAssignmentEnv]
    TG[Task generator<br/>replenishment tasks] --> ENV
    TCA[TaskCreationActor<br/>proactive tasks] --> ENV
    ENV --> SIM[RobotSimulator<br/>one per robot]
    SIM --> NOISE[Telemetry noise<br/>src/reliability]
    NOISE --> ENV
    ENV --> POL[GAPOPolicyNetwork<br/>graph encoder, robot attention, scorer]
    POL --> PPO[GAPOPPO<br/>PPO + ranking loss]
    PPO -->|assignments| ENV
    ENV -->|rewards| PPO
    ENV -->|traversal records| EC[EdgeCostManager<br/>learned path costs]
    EC -->|path costs| ENV
    PPO --> OUT[Checkpoints, logs, analytics]
    POL -.->|inference| ENVR[GAPOTaskAssignmentEnvReal]
    ENVR --> BE[ROSBridgeRobotBackend]
    BE <-->|WebSocket /ws| ROS[ROS 2 bridge<br/>external]
```

## Environments

| | ILC pilot (`pilot_trial`, this branch) | Hospital (`main`) |
| --- | --- | --- |
| Config | `configs/ilc_pilot_v1_care_robotics.json` | `configs/revised_hospital_config_v3.json` (and a reduced-consumption variant) |
| Layout | 3 nodes: 1 storage, 2 shelves | 12 nodes: 3 storage, 9 recovery wards |
| Stock | 4 SKUs | 132 SKUs (81 to 132 per node) |
| Robots | 1, capacity 2 | 5 by default, capacity 12 |
| Demand per SKU | daily mean × school-period profile × weekday × node foot traffic | daily mean × intraday profile × weekday × served beds × department-category multiplier |
| Time features | school periods from the config | 4 shift periods, intraday and weekday weights |
| Task types | replenishment | replenishment, ad hoc, returns, emergency |
| Task features | 10 | 15, including task-type flags |
| Layout-specific rewards | restock-before-break bonus, movement-during-break penalty | robot collision tracking, pickup load bonus, ad-hoc delivery bonus |

Shared by both: the GAPO policy and PPO, attention and debiasing, the learned edge-cost model, the task-creation actor, charging (robots charge at hub or storage nodes), the real-robot loop, the loggers and analytics, and telemetry noise.

## Status

| Area | State |
| --- | --- |
| Simulation training (`run_training.py`) | Working |
| Simulation deployment (`run_deployment.py`) | Working, headless or with the pygame view |
| Real-robot training and deployment (`run_training_real.py`, `run_real_deployment.py`) | Not yet run against hardware. Requires the ROS 2 bridge. |
| Reliability layer | Phase 1 only: telemetry noise (`src/reliability/`) |

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

Drop `--headless` for the pygame view. Run any script with `--help` for its full list of flags.

ILC demand is sparse: with the default settings (no ad-hoc tasks, consumption scaled to 60% during warm-up), no shelf reaches its reorder point in the first few simulated hours. Add `--stochastic-tasks-per-hour 2` for a quick check that assignment and training run.

## Repository layout

```text
run_training.py              Simulation PPO training
run_deployment.py            Simulation inference, optional pygame view
run_training_real.py         Training against a live robot via the ROS 2 bridge
run_real_deployment.py       Inference against a live robot
configs/                     ILC layout and demand configuration
src/environment/             Graph, robots, tasks and the environment
src/multi_agent_ppo/         Policy, encoders, attention, PPO, learned edge cost, task creation
src/deployment/              ROS bridge backend and WebSocket client
src/reliability/             Sim-to-real effects (telemetry noise)
src/analytics/               Realtime analytics and metrics
src/utils/                   Logging and helpers
shell/                       Cluster (Slurm) job scripts
tools/                       Branch audit and demand probe
```

## Configuration

This branch loads ILC configs (`metadata.environment_type: ilc_open_space`).

- `school_schedule` defines the day: periods, operating days and break periods.
- Each SKU in `sku_database` has a consumption model with a daily mean and a per-period profile.
- Each node's `foot_traffic_weight` scales its demand.
- `metadata.num_robots` and `metadata.robot_max_capacity` set the fleet.

## Real-robot operation

The policy drives live robots through an external ROS 2 bridge, which handles Nav2 and publishes robot state. The bridge server is not in this repository.

**Connection.** Set the bridge URL with `--bridge-url` or `ROS_BRIDGE_URL` in `.env`. The default is `ws://localhost:8765`. The client appends `/ws` to the URL. `ROSBridgeRobotBackend` opens one WebSocket client per robot and reconnects with exponential backoff (1 s, doubling, capped at 30 s).

**Messages from the bridge** (JSON, keyed by `message_type`):

| Type | Contents | Effect |
| --- | --- | --- |
| `robot_state` | position, orientation, velocity, battery, capacity, active task | Updates the robot's latest telemetry |
| `task_status` | task ID and status (`completed`, `in_progress`, `failed`, `canceled`) | Completion or failure handling |
| `location_inventory` | per-SKU stock at each location | Stored; the real environment overwrites simulated shelf stock with it |
| `system_state` | full snapshot including inventory | Stored as the latest system state |
| `emergency_stop` | `active` flag | Sets `emergency_stop_triggered` |
| `task_response`, `connected`, `heartbeat` | acknowledgements | Logged only |

**Messages to the bridge:** `task_command` (task ID, robot ID, waypoints, task metadata) and `task_cancel`.

**Failures.** A failed or cancelled goal is retried up to 3 times. After that, the task is dropped and the robot freed. The next inventory check recreates the task if the demand still exists.

```bash
python run_training_real.py --config configs/ilc_pilot_v1_care_robotics.json --bridge-url ws://<robot-host>:8765
python run_real_deployment.py --config configs/ilc_pilot_v1_care_robotics.json \
    --use-trained --checkpoint outputs/<exp>/checkpoints/gapo_real_final.pth --bridge-url ws://<robot-host>:8765
```

## Outputs

Each run writes to `--output-dir/<exp-name>/`:

- `checkpoints/`: policy, optimiser, task-creation actor and edge-cost model, plus training metrics
- `logs/`: task, decision, ranking, SKU and robot-sample logs
- `analytics/` and `metrics_final/`: plots and metric files

Deployment runs write to `outputs/`, which is git-ignored.

## Known limitations

- Real-robot operation has not been tested with hardware.
- Only telemetry noise from the reliability layer is implemented.
- Checkpoints are not interchangeable between branches, because the feature sizes differ.

## Contact

For hospitals, investors and engineers, use the "Get in touch" form on [thecarerobotics.com](https://www.thecarerobotics.com/).

For questions about this code, email [nikhil.raghavan25@imperial.ac.uk](mailto:nikhil.raghavan25@imperial.ac.uk).
