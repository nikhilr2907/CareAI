<p align="center">
  <img src="assets/logo.png" alt="CareRobotics logo" width="120">
</p>

# CareRobotics

Website: [thecarerobotics.com](https://www.thecarerobotics.com/)

CareRobotics is building the automation layer that gives hospital staff their time back. Ward nurses spend a large part of each shift searching for supplies and equipment, and missing stock delays care. CareRobotics is developing robots that restock wards, with the task-allocation software in this repository deciding which robot does which job. The company is supported by Imperial College London and UCL through the Imperial VCC Accelerator and the UCL Venture Builder. A full-size prototype is in development, and a hospital pilot is under discussion with HCA UK.

This repository holds the task-allocation software for CareRobotics robots. This branch, `main`, is configured for hospital environments. The `pilot_trial` branch is configured for the Independent Learning Centre (Sixth Form) pilot.

## Overview

A multi-agent PPO policy (GAPO) assigns restocking and pickup tasks to robots over a graph of storage, recovery and corridor nodes. The environment is simulated from a hospital configuration. No training data is required, because tasks and consumption are generated from the config.

## Status

| Area | State |
| --- | --- |
| Simulation training (`run_training.py`) | Working |
| Headless deployment (`run_deployment.py`) | Working with a trained checkpoint |
| Real-robot scripts | Not on this branch. The ROS bridge client is included but not yet wired into a runner. |
| Reliability layer | Phase 1 telemetry noise is enabled in the environment. Phases 2 and 3 are not implemented. |

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
python run_training.py --config configs/hospital_3nodes_consumables.json \
    --iterations 10000 --device auto --seed 42 --output-dir outputs --exp-name hospital_sim
```

Run a trained policy headless:

```bash
python run_deployment.py --use-trained \
    --checkpoint outputs/hospital_sim/checkpoints/gapo_final.pth \
    --headless --max-runtime 3600
```

Drop `--headless` for the pygame view.

Run `python run_training.py --help` for the full list of training flags.

## Repository layout

```text
run_training.py              Simulation PPO training
run_deployment.py            Simulation inference, optional pygame view
configs/                     Hospital configuration
examples/                    Config examples and a usage script
src/environment/             Graph, robots, tasks and the environment
src/multi_agent_ppo/         Policy, encoders, attention, PPO, debiasing
src/deployment/              Mock bridge, task source, ROS bridge client
src/reliability/             Sim-to-real effects (telemetry noise)
src/analytics/               Realtime analytics and metrics
src/utils/                   Logging and helpers
shell/                       Cluster (Slurm) job scripts
```

## Configuration

`configs/hospital_3nodes_consumables.json` is the hospital configuration. It describes three nodes, the robots, and the SKUs with their consumption patterns.

`run_deployment.py` loads the first JSON file in `configs/`, so keep one configuration there.

## Outputs

Each training run writes to `--output-dir/<exp-name>/`:

- `checkpoints/`: model checkpoints and training metrics
- `logs/`: task, decision, ranking and robot-sample logs
- `analytics/` and `metrics_final/`: plots and metric files

The `outputs/` folder is git-ignored.

## Cluster training

The Slurm jobs are in `shell/`. Submit them from the repository root:

```bash
sbatch shell/train.sh          # one training run
sbatch shell/train_long.sh     # longer run with a larger iteration budget
sbatch shell/sweep.sh          # learning-rate and hidden-dim sweep (9 array jobs)
```

Settings are environment variables, for example `CONFIG=... ITERATIONS=... sbatch shell/train.sh`. Set `MODULES` (environment modules to load) and `VENV_DIR` (a virtualenv to activate) if your cluster needs them. See `shell/env.sh`.

`.github/workflows/deploy.yml` syncs this branch to the cluster on pushes to `main`. Its hosts, username and remote path come from GitHub variables and secrets, listed at the top of the file.

## Known limitations

- Simulated consumption is a model. Validate it against real records before relying on learned policies.
- The real-robot scripts live on `pilot_trial`, not here.

## Contact

[Contact details: to be written.]
