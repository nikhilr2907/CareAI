# Environments

This codebase supports two layouts on separate branches. Both run the same learning and environment code. They differ only in the layout and demand layer, meaning the config format and loader, the graph and task features, the demand model, and a handful of layout specific rewards.

| | Hospital (`main`) | ILC pilot (`pilot_trial`) |
| --- | --- | --- |
| Config | `configs/revised_hospital_config_v3.json` | `configs/ilc_pilot_v1_care_robotics.json` |
| Layout | 12 nodes, 3 storage and 9 recovery wards | 3 nodes, 1 storage and 2 shelves |
| Stock | 132 SKUs, 81 to 132 per node | 4 SKUs |
| Robots | 5 by default, capacity 12 | 1, capacity 2 |
| Demand per SKU | Daily mean times intraday profile times weekday times served beds times department multiplier | Daily mean times school period profile times weekday times node foot traffic |
| Task types | Replenishment, ad hoc, returns, emergency | Replenishment |
| Task features | 15, including task type flags | 10 |
| Layout specific rewards | Robot collision tracking, pickup load bonus, ad hoc delivery bonus | Restock before break bonus, movement during break penalty |

Shared by both layouts are the GAPO policy and PPO trainer, attention and debiasing, the learned edge cost model, the task creation actor, charging at hub or storage nodes, the loggers and analytics, and telemetry noise.

## The config file

`revised_hospital_config_v3.json` is the default hospital config, at format version 3.0. It provides four sections.

`nodes` lists operational locations, each carrying its served bed count and department tags.

`shelves` lists per SKU bins, each with a par level and a reorder level.

`item_database` gives a consumption model per SKU, made of a daily mean, an intraday profile and weekday multipliers.

`metadata.demand_profiles` gives department category demand multipliers.

`metadata.num_robots` and `metadata.robot_max_capacity` set the fleet size. When absent, the environment defaults to 5 robots with capacity 12.

Older hospital config formats still load, but carry no per SKU demand model, so they produce little or no consumption.

## Next step

Continue with [Core interface](core-interface.md) for how the environment turns a config into an observation and a reward each step.
