"""
Test script for region-based position mapping.
Verifies that continuous (x, y) positions correctly map to graph nodes/edges.
"""

import sys
import numpy as np
from src.environment.graph.graph_state import GraphState
from src.environment.graph_helpers import (
    get_current_node_index,
    get_current_edge_index,
    compute_edge_progress
)


def test_region_based_positioning():
    """Test that robots at continuous positions map to correct nodes/edges."""

    print("=" * 70)
    print("REGION-BASED POSITION MAPPING TEST")
    print("=" * 70)

    # Create graph with region-based nodes
    graph = GraphState()

    print(f"\nCreated graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    # Display node regions
    print("\n" + "=" * 70)
    print("NODE REGIONS:")
    print("=" * 70)
    for idx, node in enumerate(graph.nodes):
        x_min, y_min, x_max, y_max = node.bounds
        print(f"Node {idx} ({node.node_id}, {node.node_type}):")
        print(f"  Center: ({node.center_x:.1f}, {node.center_y:.1f})")
        print(f"  Size: {node.width:.1f}m × {node.height:.1f}m")
        print(f"  Bounds: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
        print()

    # Test cases: robot positions and expected node assignments
    test_cases = [
        # (x, y, expected_node_idx, description)
        (2.5, 2.5, 0, "Center of node 0 (storage)"),
        (1.0, 1.0, 0, "Inside node 0 region"),
        (4.0, 3.0, 0, "Near edge of node 0"),

        (10.0, 2.5, 1, "Center of node 1 (corridor)"),
        (20.0, 2.5, 2, "Center of node 2 (recovery ward)"),

        (6.0, 7.5, 8, "Center of node 8 (central corridor)"),

        # Test positions in corridors (between nodes)
        (6.0, 2.5, None, "Between nodes 1 and 8 (in corridor)"),
        (15.0, 2.5, None, "Between nodes 1 and 2 (in corridor)"),
    ]

    print("=" * 70)
    print("POSITION-TO-NODE MAPPING TESTS:")
    print("=" * 70)

    passed = 0
    failed = 0

    for robot_x, robot_y, expected_node, description in test_cases:
        node_idx = get_current_node_index(robot_x, robot_y, graph)

        status = "[PASS]" if node_idx == expected_node else "[FAIL]"
        if node_idx == expected_node:
            passed += 1
        else:
            failed += 1

        print(f"\nTest: {description}")
        print(f"  Position: ({robot_x:.1f}, {robot_y:.1f})")
        print(f"  Expected node: {expected_node}")
        print(f"  Actual node: {node_idx}")
        print(f"  {status}")

        # If in corridor, find which edge
        if node_idx is None:
            edge_idx = get_current_edge_index(robot_x, robot_y, graph, tolerance=2.0)
            if edge_idx is not None:
                edge = graph.edges[edge_idx]
                progress = compute_edge_progress(robot_x, robot_y, graph, edge_idx)
                print(f"  On edge {edge_idx}: {edge.from_node} -> {edge.to_node}")
                print(f"  Progress: {progress:.2%}")

    # Test node feature extraction
    print("\n" + "=" * 70)
    print("NODE FEATURE EXTRACTION:")
    print("=" * 70)

    node_features = graph.get_node_features()
    print(f"Node features shape: {node_features.shape}")
    print(f"Expected shape: ({len(graph.nodes)}, 8)")

    if node_features.shape == (len(graph.nodes), 8):
        print("[PASS] Node features have correct shape")
        passed += 1
    else:
        print("[FAIL] Node features have incorrect shape")
        failed += 1

    print(f"\nSample node features (node 0):")
    print(f"  {node_features[0]}")
    print(f"  [center_x, center_y, width, height, stock, consumption, time_to_stockout, occupancy]")

    # Test edge feature extraction
    print("\n" + "=" * 70)
    print("EDGE FEATURE EXTRACTION:")
    print("=" * 70)

    edge_features = graph.get_edge_features()
    print(f"Edge features shape: {edge_features.shape}")
    print(f"Expected shape: ({len(graph.edges)}, 5)")

    if edge_features.shape == (len(graph.edges), 5):
        print("[PASS] Edge features have correct shape")
        passed += 1
    else:
        print("[FAIL] Edge features have incorrect shape")
        failed += 1

    print(f"\nSample edge features (edge 0):")
    print(f"  {edge_features[0]}")
    print(f"  [distance_m, corridor_width, current_weight, has_patient_bed, clutter_level]")

    # Test region containment methods
    print("\n" + "=" * 70)
    print("REGION CONTAINMENT METHODS:")
    print("=" * 70)

    node = graph.nodes[0]
    test_point = (node.center_x, node.center_y)

    if node.contains_point(*test_point):
        print(f"[PASS] Node.contains_point() works for center point")
        passed += 1
    else:
        print(f"[FAIL] Node.contains_point() failed for center point")
        failed += 1

    # Test bounds property
    bounds = node.bounds
    if len(bounds) == 4:
        print(f"[PASS] Node.bounds property returns (x_min, y_min, x_max, y_max)")
        passed += 1
    else:
        print(f"[FAIL] Node.bounds property has incorrect format")
        failed += 1

    # Test area property
    expected_area = node.width * node.height
    if abs(node.area - expected_area) < 0.01:
        print(f"[PASS] Node.area property = {node.area:.2f} m^2")
        passed += 1
    else:
        print(f"[FAIL] Node.area property incorrect")
        failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY:")
    print("=" * 70)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed == 0:
        print("\n[SUCCESS] ALL TESTS PASSED! Region-based positioning is working correctly.")
        return 0
    else:
        print(f"\n[ERROR] {failed} TEST(S) FAILED. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = test_region_based_positioning()
    sys.exit(exit_code)
