"""
Variable consumption rate patterns for hospital nodes.

Simulates realistic hospital supply usage that varies by:
- Time of day (morning rush, night lull)
- Day of week (weekday vs weekend)
- Day type (regular, holiday, emergency surge)
"""
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class DayType(Enum):
    """Type of day affecting consumption patterns."""
    REGULAR = "regular"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    EMERGENCY_SURGE = "emergency_surge"


@dataclass
class ConsumptionPattern:
    """
    Defines consumption multipliers for different times and conditions.

    Multipliers are applied to the base consumption rate:
        actual_rate = base_rate × hourly_multiplier × day_multiplier
    """
    # Base consumption rate (items/hour at baseline)
    base_rate: float

    # Hourly patterns (24 values, one per hour)
    hourly_multipliers: np.ndarray = None

    # Day type multipliers
    weekday_multiplier: float = 1.0
    weekend_multiplier: float = 0.7
    holiday_multiplier: float = 0.5
    emergency_multiplier: float = 2.5

    # Randomness
    noise_std: float = 0.1  # Standard deviation for random variation

    def __post_init__(self):
        """Initialize default hourly pattern if not provided."""
        if self.hourly_multipliers is None:
            self.hourly_multipliers = self._get_default_hourly_pattern()

    @staticmethod
    def _get_default_hourly_pattern() -> np.ndarray:
        """
        Default hourly consumption pattern for a hospital ward.

        Pattern reflects typical hospital activity:
        - Low at night (00:00-06:00): 0.3-0.5×
        - Morning rush (06:00-09:00): 1.5-2.0×
        - Steady day (09:00-17:00): 1.0-1.2×
        - Evening decline (17:00-22:00): 0.8-1.0×
        - Night (22:00-24:00): 0.5-0.7×

        Returns:
            Array of 24 multipliers (one per hour)
        """
        return np.array([
            # 00:00-05:00 (night - minimal activity)
            0.4, 0.3, 0.3, 0.35, 0.4, 0.5,

            # 06:00-08:00 (morning rush - shift change, rounds, breakfast)
            1.5, 2.0, 1.8,

            # 09:00-11:00 (morning procedures)
            1.2, 1.3, 1.2,

            # 12:00-13:00 (lunch)
            1.4, 1.3,

            # 14:00-16:00 (afternoon steady)
            1.1, 1.0, 1.1,

            # 17:00-19:00 (evening shift change, dinner)
            1.3, 1.4, 1.2,

            # 20:00-21:00 (evening wind down)
            1.0, 0.8,

            # 22:00-23:00 (night shift begins)
            0.7, 0.5
        ], dtype=np.float32)


class VariableConsumptionSystem:
    """
    Manages variable consumption rates for all nodes.

    Usage:
        system = VariableConsumptionSystem()

        # Set patterns for different node types
        system.set_pattern('recovery', ConsumptionPattern(base_rate=10.0))

        # Get current rate for a node
        rate = system.get_current_rate(
            node_id='Ward_10817',
            current_time=3600.0,  # 1 hour into simulation
            sim_start_day=0,      # Monday
            day_type=DayType.REGULAR
        )
    """

    def __init__(self):
        """Initialize consumption system."""
        # Store patterns by node ID
        self.node_patterns: Dict[str, ConsumptionPattern] = {}

        # Store node type patterns (fallback)
        self.type_patterns: Dict[str, ConsumptionPattern] = {
            'recovery': self._get_recovery_ward_pattern(),
            'storage': ConsumptionPattern(base_rate=0.0),  # Storage doesn't consume
            'corridor': ConsumptionPattern(base_rate=0.0),
            'hub': ConsumptionPattern(base_rate=0.0)
        }

    @staticmethod
    def _get_recovery_ward_pattern() -> ConsumptionPattern:
        """Default pattern for recovery wards."""
        return ConsumptionPattern(
            base_rate=10.0,  # 10 items/hour baseline
            weekday_multiplier=1.0,
            weekend_multiplier=0.7,
            holiday_multiplier=0.5,
            emergency_multiplier=2.5,
            noise_std=0.15
        )

    @staticmethod
    def _get_icu_pattern() -> ConsumptionPattern:
        """Pattern for ICU (more constant, less variation)."""
        # ICU has more constant consumption (24/7 critical care)
        hourly = np.ones(24, dtype=np.float32)
        hourly[6:9] = 1.3   # Shift change
        hourly[18:21] = 1.3  # Shift change
        hourly[0:6] = 0.9   # Slightly lower at night

        return ConsumptionPattern(
            base_rate=15.0,  # Higher base rate
            hourly_multipliers=hourly,
            weekday_multiplier=1.0,
            weekend_multiplier=0.95,  # Less weekend variation
            holiday_multiplier=0.9,
            emergency_multiplier=3.0,
            noise_std=0.1  # Less noise (more predictable)
        )

    @staticmethod
    def _get_emergency_dept_pattern() -> ConsumptionPattern:
        """Pattern for emergency department (high variability)."""
        hourly = np.array([
            # Night: moderate (drunk/accident cases)
            0.8, 0.7, 0.6, 0.6, 0.7, 0.9,
            # Morning: low
            0.7, 0.6, 0.7,
            # Day: steady increase
            0.9, 1.0, 1.1, 1.2, 1.2, 1.3,
            # Afternoon/evening: peak (people leave work)
            1.4, 1.5, 1.6, 1.7, 1.8,
            # Late evening: high
            1.5, 1.3, 1.1, 1.0
        ], dtype=np.float32)

        return ConsumptionPattern(
            base_rate=20.0,
            hourly_multipliers=hourly,
            weekday_multiplier=1.0,
            weekend_multiplier=1.2,  # Higher on weekends!
            holiday_multiplier=1.5,  # Holidays can be busy
            emergency_multiplier=3.5,
            noise_std=0.25  # High variability
        )

    def set_pattern(self, node_id: str, pattern: ConsumptionPattern):
        """Set consumption pattern for a specific node."""
        self.node_patterns[node_id] = pattern

    def set_type_pattern(self, node_type: str, pattern: ConsumptionPattern):
        """Set default pattern for a node type."""
        self.type_patterns[node_type] = pattern

    def get_pattern(self, node_id: str, node_type: str = None) -> ConsumptionPattern:
        """
        Get consumption pattern for a node.

        Lookup order:
        1. Specific node pattern (if set)
        2. Node type pattern (if type provided)
        3. Default recovery pattern
        """
        if node_id in self.node_patterns:
            return self.node_patterns[node_id]

        if node_type and node_type in self.type_patterns:
            return self.type_patterns[node_type]

        # Default fallback
        return ConsumptionPattern(base_rate=0.0)

    def get_current_rate(
        self,
        node_id: str,
        current_time: float,
        sim_start_day: int = 0,
        day_type: DayType = DayType.REGULAR,
        node_type: str = None
    ) -> float:
        """
        Calculate current consumption rate for a node.

        Args:
            node_id: Node identifier
            current_time: Current simulation time (seconds since start)
            sim_start_day: Day of week simulation started (0=Monday, 6=Sunday)
            day_type: Type of day (regular, weekend, holiday, emergency)
            node_type: Node type (optional, for fallback)

        Returns:
            Current consumption rate (items/hour)
        """
        pattern = self.get_pattern(node_id, node_type)

        # Base rate
        rate = pattern.base_rate

        if rate == 0.0:
            return 0.0  # No consumption for storage/corridors

        # 1. Hour of day multiplier
        hour_of_day = int((current_time / 3600.0) % 24)
        hourly_mult = pattern.hourly_multipliers[hour_of_day]

        # 2. Day type multiplier
        if day_type == DayType.WEEKEND:
            day_mult = pattern.weekend_multiplier
        elif day_type == DayType.HOLIDAY:
            day_mult = pattern.holiday_multiplier
        elif day_type == DayType.EMERGENCY_SURGE:
            day_mult = pattern.emergency_multiplier
        else:  # REGULAR
            day_mult = pattern.weekday_multiplier

        # 3. Random noise (realistic variation)
        noise = np.random.normal(1.0, pattern.noise_std)
        noise = max(0.5, min(1.5, noise))  # Clamp to [0.5, 1.5]

        # Combine all factors
        final_rate = rate * hourly_mult * day_mult * noise

        return max(0.0, final_rate)

    @staticmethod
    def get_day_type(
        current_time: float,
        sim_start_day: int = 0,
        holidays: list = None,
        emergency_mode: bool = False
    ) -> DayType:
        """
        Determine the day type based on current time.

        Args:
            current_time: Current simulation time (seconds)
            sim_start_day: Day of week simulation started (0=Monday, 6=Sunday)
            holidays: List of day numbers that are holidays
            emergency_mode: Whether in emergency surge mode

        Returns:
            DayType enum
        """
        if emergency_mode:
            return DayType.EMERGENCY_SURGE

        # Calculate current day
        days_elapsed = int(current_time / 86400.0)  # 86400 seconds per day
        current_day = (sim_start_day + days_elapsed) % 7

        if holidays and days_elapsed in holidays:
            return DayType.HOLIDAY

        if current_day >= 5:  # Saturday (5) or Sunday (6)
            return DayType.WEEKEND

        return DayType.REGULAR


def create_realistic_ward_patterns(
    num_wards: int,
    base_rate_range: Tuple[float, float] = (5.0, 15.0)
) -> Dict[str, ConsumptionPattern]:
    """
    Create diverse consumption patterns for multiple wards.

    Each ward gets slightly different:
    - Base consumption rate
    - Peak hours (some wards busier at different times)
    - Noise levels

    Args:
        num_wards: Number of wards to create patterns for
        base_rate_range: Range of base consumption rates (min, max)

    Returns:
        Dictionary mapping ward IDs to consumption patterns
    """
    patterns = {}

    for i in range(num_wards):
        # Random base rate
        base_rate = np.random.uniform(*base_rate_range)

        # Random variation in hourly pattern (shift peaks slightly)
        hourly = ConsumptionPattern._get_default_hourly_pattern()

        # Add some variation (shift peak hours ±2 hours)
        shift = np.random.randint(-2, 3)
        hourly = np.roll(hourly, shift)

        # Random noise level
        noise_std = np.random.uniform(0.1, 0.2)

        pattern = ConsumptionPattern(
            base_rate=base_rate,
            hourly_multipliers=hourly,
            noise_std=noise_std
        )

        patterns[f"node_{i}"] = pattern

    return patterns


# Example usage and testing
if __name__ == '__main__':
    print("Variable Consumption System Demo\n")

    system = VariableConsumptionSystem()

    # Simulate 24 hours for a recovery ward
    print("Recovery Ward - 24 Hour Pattern (Regular Weekday):")
    print("Hour | Consumption Rate")
    print("-" * 30)

    for hour in range(24):
        current_time = hour * 3600.0
        rate = system.get_current_rate(
            node_id='Ward_A',
            current_time=current_time,
            sim_start_day=0,  # Monday
            day_type=DayType.REGULAR,
            node_type='recovery'
        )
        print(f"{hour:02d}:00 | {rate:.2f} items/hour")

    # Compare weekday vs weekend
    print("\n\nWeekday vs Weekend Comparison (at 10:00 AM):")
    current_time = 10 * 3600.0

    weekday_rate = system.get_current_rate('Ward_A', current_time, 0, DayType.REGULAR, 'recovery')
    weekend_rate = system.get_current_rate('Ward_A', current_time, 5, DayType.WEEKEND, 'recovery')

    print(f"Weekday:  {weekday_rate:.2f} items/hour")
    print(f"Weekend:  {weekend_rate:.2f} items/hour")
    print(f"Ratio:    {weekend_rate/weekday_rate:.2f}x")
