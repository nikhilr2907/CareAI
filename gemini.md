1) **What we’re building***
 
 
A continuous, multi-robot decision system to run autonomous intra-hospital transfers (primarily consumables logistics):
 
Central supply hub (often 1 per hospital; sometimes 2) holds stock on shelves.
Wards hold local “point-of-use” stock.
A fleet of robotic trolleys/carts performs hub → ward replenishment (the main workload), plus smaller ward ↔ ward / pharmacy ↔ ward ad-hoc jobs and returns-to-hub.
 
 
The MAPPO layer is responsible for high-level fleet orchestration (who does what, when), not fine motor control.
 
 
 
2) ***Key sizing assumption to ground discussion (consumables per ward)***
 
 
Public, ward-level “how many consumables does a cardiology ward carry?” figures aren’t standardised and vary a lot by hospital, unit type, and how stock is defined (SKUs vs locations vs non-stock items). I couldn’t find a reliable single “cardiology ward” number.
 
What is available are published nursing-unit case studies. One study reporting five nursing units found ~193 to ~887 total distinct products per unit (with “inventory” products reported as 187–664, plus additional non-stock items). A simple average across those five units is about ~511 total products per nursing unit. 
 
Practical takeaway for your model: using ~500 product-lines per ward as a baseline assumption is defensible for simulation, with a wide range (≈200–900) to stress-test robustness. 
 
 
 
3) ***Core responsibilities of the system***
 
 
 
A) Continuous operation (end goal)
 
 
The deployed system runs continuously (not episodic), with a rolling stream of tasks and changing conditions.
Training can be episodic/curriculum-based if it transfers well, but the target behaviour must work in a non-terminating, real-time setting.
 
 
 
B) Scalable scope and configuration
 
We want the same software to work across:
 
Different hospitals (layouts, ward counts, hub locations, constraints like “no-go zones”).
Different fleet sizes (e.g., 50 robots → 500 robots).
Evolving inputs (new sensors/data fields, new task types).
 
 
This implies:
 
A clean configuration layer (UI / admin tools) to “plug in” hospital specifics instead of hard-coding per-site logic.
A stable interface contract between simulation, MAPPO policy, and real robot telemetry.
 
 
 
C) Rich but optional context inputs
 
We should be able to incorporate (as available):
 
Time-of-day / day-of-week / seasonality signals.
Planned activity (e.g., theatre schedules).
Preference patterns (e.g., surgeon-specific consumable preferences).
Potentially non-identifiable clinical categories (no names/IDs; ideally aggregated demand drivers), subject to UK governance.
 
 
The philosophy: start with minimal signals; allow “drop-in” features later without redesigning the whole system.
 
 
 
4) ***System boundaries and architecture (important for feasibility)***
 
 
Three-layer separation (recommended):
 
MAPPO / fleet allocator (this project)
Navigation / routing layer
Robot control layer

 
 
MAPPO should assume the lower layers can fail occasionally; the policy should be robust (timeouts, retries, reassignment).
 
 
 
5) ***Task types we must support***
 
 
Hub → ward replenishment (dominant flow)
Ward → hub returns
Ad-hoc courier jobs (lower volume, high variance)

 
 
 
 
6) Outputs we care about (why MAPPO exists)
 
 
The policy output is used in two ways:
 
Operations: run the real fleet effectively day-to-day.
Simulation / planning: quantify:

 
 
 
 
7) ***RL formulation sketch (for discussion)**
 
 
 
State (examples)
 
 
Robot: location, availability, battery, payload/cargo type(s), current assignment.
Environment: hospital graph, blocked zones, congestion proxies.
Demand: outstanding tasks, ward stock levels (or predicted “time-to-stockout”), deadlines, priorities.
 
 
 
Actions (examples)
 
 
Assign robot i to task k, or reassign; choose pickup/dropoff sequencing; choose “wait/hold” under congestion.
(If routing is separate) output intent-level decisions rather than step-by-step moves.
 
 
 
Reward (examples)
 
 
Positive for on-time delivery, weighted by priority (1–5).
Penalties for lateness, stockouts, excessive travel time, congestion creation, and “thrashing” (too many reassignments).
Mild penalty for idling when urgent tasks exist; battery-safe behaviour encouraged.
 
 
 
 
8) My implementation learnings so far (what I’d do differently)
 
 
You can’t prompt your way out of missing specs. MAPPO quality depends heavily on having explicit requirements: state/action definitions, reward design, constraints, and success metrics.
Curriculum/staging helps: start with a small, “obviously learnable” environment, then unlock complexity (bigger map, more agents, more tasks, tighter deadlines).
It may help to train the concept first (pickup → carry → dropoff) in a simple setting, then transfer. I didn’t fully crack that transfer pipeline yet, but conceptually it’s the right direction.
Map realism early pays off: it trains slower, but you can run overnight; and you avoid rebuilding once behaviour already overfits to toy layouts.
 
 
 
 
9) Priority system and “predictable delivery” requirement
 
 
Nurses should be able to set priority 1–5. The system should learn to:
 
Treat priority 5 as urgent (pre-empt lower priorities where appropriate).
Provide an ETA-style commitment (e.g., “worst-case 23 minutes”) based on current fleet state and congestion, and then behave consistently with that.
 
How many robots are needed for a given service level?
What redundancy is required so “everything doesn’t go sideways” when disruptions occur?
What happens under surges (e.g., unusual demand patterns)?
Ward ↔ ward, pharmacy ↔ ward, documents/small items.
Unused/overstock sent back for sorting centrally.
Maintain ward stock above safety thresholds.
Minimise stockouts and emergency runs.
Low-level motion, docking, manipulation, safety behaviours.
Converts assignment into routes (graph-based, dynamic re-routing, congestion handling).
Chooses: task assignment, prioritisation, dispatch timing, queueing logic.
Works with estimated travel/service times and constraints.