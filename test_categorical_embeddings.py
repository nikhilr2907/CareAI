"""
Smoke test for fine-grained categorical embeddings.

Tests:
1. graph_state.get_node_features_with_category_stats() returns correct shapes
2. HospitalGraphEncoder accepts 4 categorical features
3. Forward pass works end-to-end
"""
import torch
import numpy as np
from src.environment.graph.graph_state import GraphState
from src.multi_agent_ppo.gapo_gnn_encoders import HospitalGraphEncoder


def test_graph_state_features():
    """Test that graph_state returns correct categorical dimensions."""
    print("\n" + "="*80)
    print("TEST 1: Graph State Feature Extraction")
    print("="*80)

    # Create a simple graph state
    graph_state = GraphState()
    graph_state.current_time = 28800.0  # 8:00 AM

    # Get features
    continuous, categorical = graph_state.get_node_features_with_category_stats()

    print(f"✓ Continuous features shape: {continuous.shape}")
    print(f"✓ Categorical features shape: {categorical.shape}")

    # Verify shapes
    num_nodes = len(graph_state.nodes)
    assert categorical.shape == (num_nodes, 4), f"Expected categorical shape ({num_nodes}, 4), got {categorical.shape}"
    assert continuous.ndim == 2, f"Expected 2D continuous features, got {continuous.ndim}D"

    # Check categorical values
    print(f"\n  Sample categorical features (first 3 nodes):")
    for i in range(min(3, num_nodes)):
        node_type, dept_id, shift_period, day_type = categorical[i]
        print(f"    Node {i}: node_type={node_type}, dept_id={dept_id}, shift={shift_period}, day={day_type}")

    # Verify categorical ranges
    assert categorical[:, 0].max() < 4, "node_type should be 0-3"
    assert categorical[:, 2].max() < 4, "shift_period should be 0-3"
    assert categorical[:, 3].max() < 2, "day_type should be 0-1"

    print(f"\n✓ All categorical features in valid ranges")
    print(f"✓ Graph state feature extraction: PASSED")

    return continuous, categorical


def test_gnn_encoder():
    """Test that HospitalGraphEncoder handles 4 categorical features."""
    print("\n" + "="*80)
    print("TEST 2: Hospital Graph Encoder with Fine-Grained Embeddings")
    print("="*80)

    # Create encoder with fine-grained embeddings
    encoder = HospitalGraphEncoder(
        node_continuous_dim=24,  # Updated with temporal features
        num_node_types=4,
        num_departments=10,
        num_shift_periods=4,
        num_day_types=2,
        edge_feat_dim=21,
        hidden_dim=64,
        node_type_embedding_dim=8,
        department_embedding_dim=16,
        shift_embedding_dim=4,
        day_type_embedding_dim=4
    )

    print(f"✓ Created HospitalGraphEncoder")
    print(f"  Node input: 24 continuous + 8 node_type + 16 dept + 4 shift + 4 day_type = 56 dims")

    # Create dummy data
    num_nodes = 10
    num_edges = 20

    node_continuous = torch.randn(num_nodes, 24)
    node_categorical = torch.zeros(num_nodes, 4, dtype=torch.long)

    # Fill with valid categorical values
    node_categorical[:, 0] = torch.randint(0, 4, (num_nodes,))  # node_type
    node_categorical[:, 1] = torch.randint(-1, 10, (num_nodes,))  # dept_id (-1 or 0-9)
    node_categorical[:, 2] = torch.randint(0, 4, (num_nodes,))  # shift_period
    node_categorical[:, 3] = torch.randint(0, 2, (num_nodes,))  # day_type

    edge_features = torch.randn(num_edges, 21)
    edge_node_indices = torch.randint(0, num_nodes, (num_edges, 2))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    print(f"✓ Created dummy tensors:")
    print(f"    node_continuous: {node_continuous.shape}")
    print(f"    node_categorical: {node_categorical.shape}")
    print(f"    edge_features: {edge_features.shape}")

    # Forward pass
    node_embeds, edge_embeds, graph_embed = encoder(
        node_continuous,
        node_categorical,
        edge_features,
        edge_node_indices,
        edge_index
    )

    print(f"\n✓ Forward pass successful!")
    print(f"  Node embeddings: {node_embeds.shape}")
    print(f"  Edge embeddings: {edge_embeds.shape}")
    print(f"  Graph embedding: {graph_embed.shape}")

    # Verify shapes
    assert node_embeds.shape == (num_nodes, 64), f"Expected ({num_nodes}, 64), got {node_embeds.shape}"
    assert edge_embeds.shape == (num_edges, 64), f"Expected ({num_edges}, 64), got {edge_embeds.shape}"
    assert graph_embed.shape == (64,), f"Expected (64,), got {graph_embed.shape}"

    print(f"\n✓ Output shapes correct")
    print(f"✓ Hospital Graph Encoder: PASSED")

    return encoder


def test_end_to_end():
    """Test end-to-end with real graph_state data."""
    print("\n" + "="*80)
    print("TEST 3: End-to-End Integration")
    print("="*80)

    # Create graph state
    graph_state = GraphState()
    graph_state.current_time = 43200.0  # 12:00 PM (afternoon)

    # Extract features
    continuous, categorical = graph_state.get_node_features_with_category_stats()

    # Convert to tensors
    node_continuous = torch.tensor(continuous, dtype=torch.float32)
    node_categorical = torch.tensor(categorical, dtype=torch.long)

    # Get edge features
    edge_continuous, edge_node_indices = graph_state.get_edge_features_complete()
    edge_features = torch.tensor(edge_continuous, dtype=torch.float32)
    edge_node_indices_tensor = torch.tensor(edge_node_indices, dtype=torch.long)

    # Build edge_index (bidirectional)
    edge_list = []
    for i, (from_idx, to_idx) in enumerate(edge_node_indices):
        edge_list.append([from_idx, to_idx])
        edge_list.append([to_idx, from_idx])  # Bidirectional
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    print(f"✓ Prepared real data from GraphState:")
    print(f"    Nodes: {node_continuous.shape[0]}")
    print(f"    Edges: {edge_features.shape[0]}")
    print(f"    Continuous dims: {node_continuous.shape[1]}")
    print(f"    Categorical dims: {node_categorical.shape[1]}")

    # Create encoder
    encoder = HospitalGraphEncoder(
        node_continuous_dim=node_continuous.shape[1],
        num_node_types=4,
        num_departments=10,
        num_shift_periods=4,
        num_day_types=2,
        edge_feat_dim=edge_features.shape[1],
        hidden_dim=64,
        node_type_embedding_dim=8,
        department_embedding_dim=16,
        shift_embedding_dim=4,
        day_type_embedding_dim=4
    )

    # Forward pass
    node_embeds, edge_embeds, graph_embed = encoder(
        node_continuous,
        node_categorical,
        edge_features,
        edge_node_indices_tensor,
        edge_index
    )

    print(f"\n✓ End-to-end forward pass successful!")
    print(f"  Node embeddings: {node_embeds.shape}")
    print(f"  Edge embeddings: {edge_embeds.shape}")
    print(f"  Graph embedding: {graph_embed.shape}")

    # Check that embeddings are not NaN
    assert not torch.isnan(node_embeds).any(), "Node embeddings contain NaN"
    assert not torch.isnan(edge_embeds).any(), "Edge embeddings contain NaN"
    assert not torch.isnan(graph_embed).any(), "Graph embedding contains NaN"

    print(f"\n✓ No NaN values in embeddings")
    print(f"✓ End-to-End Integration: PASSED")

    # Print sample categorical values to verify temporal features
    print(f"\n  Temporal categorical features (current_time = {graph_state.current_time / 3600:.1f} hours):")
    shift_names = ['night', 'morning', 'afternoon', 'evening']
    day_names = ['weekday', 'weekend']
    shift_id = categorical[0, 2]
    day_id = categorical[0, 3]
    print(f"    Shift period: {shift_names[shift_id]} (id={shift_id})")
    print(f"    Day type: {day_names[day_id]} (id={day_id})")


def main():
    """Run all tests."""
    print("\n" + "#"*80)
    print("# SMOKE TEST: Fine-Grained Categorical Embeddings")
    print("#"*80)

    try:
        # Test 1: Graph state features
        continuous, categorical = test_graph_state_features()

        # Test 2: GNN encoder
        encoder = test_gnn_encoder()

        # Test 3: End-to-end
        test_end_to_end()

        print("\n" + "#"*80)
        print("# ALL TESTS PASSED ✓")
        print("#"*80 + "\n")

    except Exception as e:
        print("\n" + "#"*80)
        print("# TEST FAILED ✗")
        print("#"*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
