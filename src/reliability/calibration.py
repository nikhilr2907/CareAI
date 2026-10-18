import numpy as np

from ..environment.graph.edge import GraphEdge


def calibrate_congestion_constants(records: list) -> dict:
    """Back out jam_density_per_m2 and person_equivalence from traversal records via least squares."""
    rows = [
        r
        for r in records
        if not r.is_lift and not r.emergency_active and r.exit_time > r.entry_time and r.max_v_ms > 0
    ]
    if len(rows) < 10:
        return {"status": "insufficient_data", "n": len(rows)}

    actual_speed = np.array([r.distance_m / (r.exit_time - r.entry_time) for r in rows])
    speed_fraction = actual_speed / np.array([r.max_v_ms for r in rows])
    area_m2 = np.array([r.corridor_width * (1.0 - min(r.clutter_level, 0.95)) * r.distance_m for r in rows])
    occupants = np.array([r.num_robots_on_edge + r.equipment_equivalents for r in rows])
    people = np.array([r.people_count for r in rows])

    y = (1.0 - speed_fraction) * area_m2
    X = np.column_stack([occupants, people])
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)

    if a <= 0:
        return {"status": "degenerate_fit", "n": len(rows)}

    jam_density_per_m2 = 1.0 / a
    person_equivalence = b * jam_density_per_m2

    return {
        "status": "ok",
        "n": len(rows),
        "jam_density_per_m2": {"empirical": jam_density_per_m2, "default": GraphEdge._JAM_DENSITY_PER_M2},
        "person_equivalence": {"empirical": person_equivalence, "default": GraphEdge._PERSON_EQUIVALENCE},
    }
