"""
Simple smoke test for fine-grained categorical embeddings (graph_state only).

Tests the graph_state.get_node_features_with_category_stats() method
to ensure it returns the correct categorical dimensions.
"""
import numpy as np
from src.environment.graph.graph_state import GraphState


def test_graph_state_features():
    """Test that graph_state returns correct categorical dimensions."""
    print("\n" + "="*80)
    print("TEST: Graph State Feature Extraction with Fine-Grained Categoricals")
    print("="*80)

    # Create a simple graph state
    graph_state = GraphState()

    # Test at different times to verify temporal features
    test_times = [
        (0.0, "midnight (00:00)", "night", "weekday"),
        (21600.0, "6:00 AM", "morning", "weekday"),
        (43200.0, "noon (12:00)", "afternoon", "weekday"),
        (64800.0, "6:00 PM", "evening", "weekday"),
        (453600.0, "Saturday 6:00 AM", "morning", "weekend"),  # 5 days + 6 hours
    ]

    shift_names = ['night', 'morning', 'afternoon', 'evening']
    day_names = ['weekday', 'weekend']

    for time_sec, time_desc, expected_shift, expected_day in test_times:
        print(f"\n  Testing at {time_desc} ({time_sec/3600:.1f} hours):")

        graph_state.current_time = time_sec

        # Get features
        continuous, categorical = graph_state.get_node_features_with_category_stats()

        # Verify shapes
        num_nodes = len(graph_state.nodes)
        assert categorical.shape == (num_nodes, 4), \
            f"Expected categorical shape ({num_nodes}, 4), got {categorical.shape}"
        assert continuous.ndim == 2, \
            f"Expected 2D continuous features, got {continuous.ndim}D"

        # Check categorical values for first node
        node_type_id, dept_id, shift_period_id, day_type_id = categorical[0]

        print(f"    [OK] Categorical shape: {categorical.shape}")
        print(f"    [OK] Continuous shape: {continuous.shape}")
        print(f"    [OK] Sample (node 0): node_type={node_type_id}, dept_id={dept_id}, "
              f"shift={shift_names[shift_period_id]}, day={day_names[day_type_id]}")

        # Verify temporal features match expectations
        assert shift_names[shift_period_id] == expected_shift, \
            f"Expected shift '{expected_shift}', got '{shift_names[shift_period_id]}'"
        assert day_names[day_type_id] == expected_day, \
            f"Expected day '{expected_day}', got '{day_names[day_type_id]}'"

        # Verify categorical ranges
        assert categorical[:, 0].max() < 4, "node_type should be 0-3"
        assert categorical[:, 2].max() < 4, "shift_period should be 0-3"
        assert categorical[:, 3].max() < 2, "day_type should be 0-1"

        # Check that continuous features include temporal info
        # Continuous features should now be 24 dimensions (or 24 + 3*N_categories)
        # Base: 22 (removed dept_id float) + 2 temporal (intraday_weight, weekday_mult) = 24
        assert continuous.shape[1] >= 24, \
            f"Expected at least 24 continuous features, got {continuous.shape[1]}"

    print(f"\n" + "="*80)
    print("[OK] ALL TEMPORAL FEATURES WORKING CORRECTLY")
    print("="*80)

    # Test a longer scenario to verify all shift periods
    print(f"\n  Full day verification:")
    for hour in [3, 9, 15, 21]:  # night, morning, afternoon, evening
        time_sec = hour * 3600.0
        graph_state.current_time = time_sec
        _, categorical = graph_state.get_node_features_with_category_stats()
        shift_id = categorical[0, 2]
        print(f"    Hour {hour:02d}:00 -> shift_period={shift_id} ({shift_names[shift_id]})")

    print(f"\n[OK] Graph state feature extraction: PASSED\n")


def test_temporal_helpers():
    """Test the temporal helper functions directly."""
    print("\n" + "="*80)
    print("TEST: Temporal Helper Functions")
    print("="*80)

    graph_state = GraphState()

    # Test shift period calculation
    test_cases_shift = [
        (0.0, 0, "night"),      # 00:00
        (18000.0, 0, "night"),  # 05:00
        (21600.0, 1, "morning"), # 06:00
        (39600.0, 1, "morning"), # 11:00
        (43200.0, 2, "afternoon"), # 12:00
        (61200.0, 2, "afternoon"), # 17:00
        (64800.0, 3, "evening"), # 18:00
        (82800.0, 3, "evening"), # 23:00
    ]

    print("\n  Shift Period ID calculation:")
    for time_sec, expected_id, name in test_cases_shift:
        actual_id = graph_state._get_shift_period_id(time_sec)
        hour = (time_sec % 86400) / 3600
        status = "[OK]" if actual_id == expected_id else "[FAIL]"
        print(f"    {status} {hour:05.1f}h -> {actual_id} ({name}) [expected: {expected_id}]")
        assert actual_id == expected_id, f"Shift period mismatch at {hour}h"

    # Test day type calculation
    test_cases_day = [
        (0.0, 0, "Monday (weekday)"),
        (86400.0, 0, "Tuesday (weekday)"),
        (172800.0, 0, "Wednesday (weekday)"),
        (259200.0, 0, "Thursday (weekday)"),
        (345600.0, 0, "Friday (weekday)"),
        (432000.0, 1, "Saturday (weekend)"),
        (518400.0, 1, "Sunday (weekend)"),
    ]

    print("\n  Day Type ID calculation:")
    for time_sec, expected_id, desc in test_cases_day:
        actual_id = graph_state._get_day_type_id(time_sec)
        status = "[OK]" if actual_id == expected_id else "[FAIL]"
        print(f"    {status} {time_sec/86400:.1f} days -> {actual_id} ({desc}) [expected: {expected_id}]")
        assert actual_id == expected_id, f"Day type mismatch for {desc}"

    # Test intraday weight
    print("\n  Intraday Weight calculation:")
    for hour in [3, 9, 15, 21]:
        time_sec = hour * 3600.0
        weight = graph_state._get_intraday_weight(time_sec)
        print(f"    Hour {hour:02d}:00 -> weight={weight:.3f}")
        assert 0.0 < weight < 1.0, f"Invalid intraday weight: {weight}"

    # Test weekday multiplier
    print("\n  Weekday Multiplier calculation:")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day_idx in range(7):
        time_sec = day_idx * 86400.0
        mult = graph_state._get_weekday_multiplier(time_sec)
        print(f"    {day_names[day_idx]:<9} -> multiplier={mult:.2f}")
        assert 0.9 < mult < 1.1, f"Invalid weekday multiplier: {mult}"

    print(f"\n[OK] Temporal helper functions: PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "#"*80)
    print("# SMOKE TEST: Fine-Grained Categorical Embeddings (Graph State)")
    print("#"*80)

    try:
        # Test temporal helpers
        test_temporal_helpers()

        # Test graph state features
        test_graph_state_features()

        print("\n" + "#"*80)
        print("# ALL TESTS PASSED [OK]")
        print("#"*80 + "\n")

        print("NOTE: To test the full GNN encoder integration, install torch-geometric")
        print("      and run the full test suite with: python test_categorical_embeddings.py\n")

    except Exception as e:
        print("\n" + "#"*80)
        print("# TEST FAILED [FAIL]")
        print("#"*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
