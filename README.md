# CareRobotics: Hospital Restocking (`main`)

Multi-agent PPO (GAPO) task allocation for a fleet of restocking robots in a hospital consumables environment. Robots are assigned pickup and delivery tasks over a graph of storage, recovery and corridor nodes. The environment is simulated: tasks and consumption come from the hospital config, so no data is needed to train.

This branch is the hospital environment. The ILC pilot (school-day schedule, floorplan, ROS bridge runners) lives on `pilot_trial`. The layout-agnostic modules from that branch are synced here.

## Status

| Area | State |
| --- | --- |
| Simulation training (`run_training.py`) | Works. Verified with a 2-iteration smoke run on `hospital_3nodes_consumables.json`. |
| Headless deployment (`run_deployment.py`) | Works with a trained checkpoint (`--use-trained`). Uses the mock bridge. |
| Real-robot scripts | None on this branch. The ROS bridge client is synced but not wired into any runner. |
| Reliability layer | Synced from `pilot_trial` (Phase 1 telemetry noise). Not enabled in the training or deployment scripts yet. |
| Tests | None (removed by request). |

## Quick start

Python 3.13 was used for verification. `requirements.txt` pins PyTorch 2.9.0 and torch-geometric 2.7.0.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

**Train:**

```bash
python run_training.py --config configs/hospital_3nodes_consumables.json \
    --iterations 10000 --device auto --seed 42 --output-dir outputs --exp-name hospital_sim
```

**Run a trained policy headless:**

```bash
python run_deployment.py --use-trained \
    --checkpoint outputs/hospital_sim/checkpoints/gapo_final.pth \
    --headless --max-runtime 3600
```

Drop `--headless` for the pygame view.

## Repository layout

```text
run_training.py              Simulation PPO training (curriculum, checkpoints, logs)
run_deployment.py            Inference in simulation, optional pygame view
configs/                     Hospital config (nodes, shelves, SKUs, consumption patterns)
examples/                    Config examples and a usage script
src/environment/             Graph, robots, tasks and the Gym-style env (gapo_env.py)
src/multi_agent_ppo/         Policy, GNN encoders, attention, PPO, debiasing
src/deployment/              Mock bridge and task source; synced ROS bridge client (not wired in)
src/reliability/             Sim-to-real effects (Phase 1 telemetry noise), not yet enabled
src/analytics/               Realtime analytics collector and metrics (synced, not wired in)
src/utils/                   Logging and helpers
train_gpu.sh                 Slurm job: one training run on the cluster
train_gpu_long.sh            Longer Slurm run
train_gpu_sweep.sh           Hyperparameter sweep Slurm job
```

## Configuration

`configs/hospital_3nodes_consumables.json` is the only config on this branch. It describes 3 nodes (1 storage, 2 recovery), the robots, and the SKUs with their consumption patterns. Controlled substances are excluded.

`examples/` has an extended config format, a mock config, and `hospital_config_usage.py` showing how to load one.

Key training flags: `--config`, `--configs-dir` or `--config-list` (curriculum), `--curriculum {none,fixed,adaptive,random}`, `--iterations`, `--lr`, `--hidden-dim`, `--num-robots`, `--max-episode-time`, `--max-assignments-per-step`, `--seed`, `--output-dir`, `--exp-name`, `--device`. Run `python run_training.py --help` for the full list.

`run_deployment.py` takes no `--config`. It loads the first JSON file in `configs/`, so keep a single config there or the choice becomes implicit.

## Outputs

Each training run writes under `--output-dir/<exp-name>/`:

- `checkpoints/` (`gapo_iter<N>.pth`, `gapo_final.pth`, `training_metrics.npz`)
- `logs/` (task, decision, ranking and robot-sample logs)
- `analytics/` and `metrics_final/` (plots and metric files)

The `outputs/` folder is git-ignored.

## Cluster training

`train_gpu.sh` runs one training job on the Imperial College GPU cluster on the hospital config. The other two scripts are a longer run and a sweep. They expect a venv and project path specific to that cluster. Adjust `PROJECT_DIR`, `OUTPUT_DIR` and the `source` line before submitting.

`.github/workflows/deploy.yml` rsyncs this branch to the cluster on every push to `main`. It needs the `SSH_KEY` secret. A `RUN_SBATCH=true` secret also submits `train_gpu.sh`.

## Known gaps

- No real-robot runners on this branch. Use `pilot_trial` for ROS bridge work.
- The reliability and analytics modules are synced but not enabled in the scripts.
- `run_deployment.py` picks its config implicitly (first JSON in `configs/`).
- Simulated consumption is a model. Validate it against real restocking records before trusting learned policies.
