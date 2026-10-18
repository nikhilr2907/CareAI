import random
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..environment.graph.graph_state import GraphState


class CongestionRandomizer:
    """Per-episode domain randomization of the congestion constants, with ADR-style auto-widening."""

    # name -> (edge attr name, default, absolute_min, absolute_max, widen_step)
    _SPECS = {
        "jam_density_per_m2": ("_JAM_DENSITY_PER_M2", 1.5, 0.8, 2.5, 0.1),
        "person_equivalence": ("_PERSON_EQUIVALENCE", 1.8, 1.0, 3.0, 0.15),
        "bed_equivalence": ("_BED_EQUIVALENCE", 4.0, 2.0, 6.0, 0.3),
    }

    def __init__(
        self,
        enable: bool = True,
        success_window: int = 20,
        widen_threshold: float = 0.7,
        narrow_threshold: float = 0.4,
    ):
        self.enable = enable
        self.widen_threshold = widen_threshold
        self.narrow_threshold = narrow_threshold
        self.ranges = {name: (default, default) for name, (_, default, _, _, _) in self._SPECS.items()}
        self._success_history: deque = deque(maxlen=success_window)
        self.last_params: dict[str, float] = {
            name: default for name, (_, default, _, _, _) in self._SPECS.items()
        }

    def sample_and_apply(self, graph_state: "GraphState") -> dict[str, float]:
        if not self.enable:
            return dict(self.last_params)

        params = {name: random.uniform(lo, hi) for name, (lo, hi) in self.ranges.items()}
        self.last_params = params
        for edge in graph_state.edges:
            for name, value in params.items():
                setattr(edge, self._SPECS[name][0], value)
        return params

    def record_episode_success(self, success: float) -> None:
        if not self.enable:
            return

        self._success_history.append(max(0.0, min(1.0, success)))
        if len(self._success_history) < self._success_history.maxlen:
            return

        avg_success = sum(self._success_history) / len(self._success_history)
        name = random.choice(list(self._SPECS.keys()))
        _, default, abs_min, abs_max, step = self._SPECS[name]
        lo, hi = self.ranges[name]

        if avg_success >= self.widen_threshold:
            self.ranges[name] = (max(abs_min, lo - step), min(abs_max, hi + step))
        elif avg_success <= self.narrow_threshold:
            self.ranges[name] = (min(default, lo + step), max(default, hi - step))

        self._success_history.clear()
