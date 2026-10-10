# Getting started

## Requirements

Use the existing `robotics` Conda environment with Python 3.13. Dependencies are pinned in `requirements.txt`.

```bash
conda activate robotics
python -m pip install -r requirements.txt
```

## Train a policy in simulation

```bash
python run_training.py --config configs/revised_hospital_config_v3.json \
    --iterations 10000 --device auto --seed 42 \
    --output-dir outputs --exp-name hospital_sim
```

Training generates its own tasks and stock consumption from the config, so no external dataset is required. Use `configs/revised_hospital_config_v3_consumption_reduced.json` for an easier early run, since its demand is around 35% lower.

On `pilot_trial`, use the pilot config and a separate experiment name.

```bash
python run_training.py --config configs/ilc_pilot_v1_care_robotics.json \
    --iterations 10000 --device auto --seed 42 \
    --output-dir outputs --exp-name ilc_sim
```

## Run a trained policy in simulation

```bash
python run_deployment.py --config configs/revised_hospital_config_v3.json \
    --use-trained --checkpoint outputs/hospital_sim/checkpoints/gapo_final.pth \
    --headless --max-runtime 3600
```

For the pilot, use `configs/ilc_pilot_v1_care_robotics.json` and the checkpoint from `outputs/ilc_sim/checkpoints/`. Use a checkpoint trained on the matching branch because task feature widths differ.

Drop `--headless` to watch the run in a pygame window. Pass `--help` to any script for its full set of flags.

## What each run produces

Every training or deployment run writes to a folder under `--output-dir`, named after `--exp-name`.

| Folder | Contents |
| --- | --- |
| `checkpoints/` | Policy, optimiser, task creation actor, edge cost model, training metrics |
| `logs/` | Task, decision, ranking, SKU and robot sample logs |
| `analytics/` and `metrics_final/` | Plots and metric files |

Deployment runs on their own write to `outputs/`, which git ignores.

## Next step

Continue with [Environments](environments.md) to see how the hospital layout and the ILC pilot layout differ, or jump to [Core interface](core-interface.md) if you already know the config and want the environment contract.

## Build the documentation

```bash
conda activate robotics
python -m pip install -r docs/requirements.txt
python -m mkdocs serve
```

Run `python -m mkdocs build --strict` to check the publishable site. Read the Docs uses `.readthedocs.yaml` to install the documentation dependencies and build it.
