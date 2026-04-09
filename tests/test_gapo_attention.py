import torch

from src.multi_agent_ppo.gapo_attention import (
    GAPOAttentionModule,
    RobotScorer,
    TaskNodeAttention,
    TaskRobotAttention,
)


def test_task_robot_attention_supports_unbatched_and_masked_inputs():
    torch.manual_seed(0)
    module = TaskRobotAttention(embed_dim=8, num_heads=2)
    task_embedding = torch.randn(8)
    robot_embeddings = torch.randn(4, 8)
    mask = torch.tensor([True, False, True, False])

    context, weights = module(task_embedding, robot_embeddings, mask)

    assert context.shape == (8,)
    assert weights.shape == (4,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert weights[1].item() == 0.0
    assert weights[3].item() == 0.0


def test_task_robot_attention_supports_batched_inputs():
    torch.manual_seed(1)
    module = TaskRobotAttention(embed_dim=8, num_heads=2)
    task_embedding = torch.randn(2, 1, 8)
    robot_embeddings = torch.randn(2, 5, 8)

    context, weights = module(task_embedding, robot_embeddings)

    assert context.shape == (2, 8)
    assert weights.shape == (2, 5)


def test_task_node_attention_returns_expected_shapes():
    torch.manual_seed(2)
    module = TaskNodeAttention(embed_dim=8, num_heads=2)
    task_embedding = torch.randn(8)
    node_embeddings = torch.randn(3, 8)

    context, weights = module(task_embedding, node_embeddings)

    assert context.shape == (8,)
    assert weights.shape == (3,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_robot_scorer_handles_batched_and_unbatched_embeddings():
    scorer = RobotScorer(embed_dim=8, hidden_dim=16)
    task_embedding = torch.randn(8)
    robot_context = torch.randn(8)
    node_context = torch.randn(8)

    unbatched_scores = scorer(
        torch.randn(4, 8),
        task_embedding,
        robot_context,
        node_context,
    )
    batched_scores = scorer(
        torch.randn(2, 4, 8),
        torch.randn(2, 8),
        torch.randn(2, 8),
        torch.randn(2, 8),
    )

    assert unbatched_scores.shape == (4,)
    assert batched_scores.shape == (2, 4)


def test_gapo_attention_module_returns_attention_info_and_respects_mask():
    torch.manual_seed(3)
    module = GAPOAttentionModule(embed_dim=8, num_heads=2, hidden_dim=16)
    task_embedding = torch.randn(8)
    robot_embeddings = torch.randn(5, 8)
    node_embeddings = torch.randn(6, 8)
    mask = torch.tensor([True, False, True, True, False])

    action_logits, attention_info = module(
        task_embedding,
        robot_embeddings,
        node_embeddings,
        robot_availability_mask=mask,
    )

    assert action_logits.shape == (5,)
    assert set(attention_info) == {
        "robot_context",
        "peer_aware_robots",
        "node_context",
        "robot_attn_weights",
        "peer_attn_weights",
        "node_attn_weights",
    }
    assert attention_info["robot_context"].shape == (8,)
    assert attention_info["peer_aware_robots"].shape == (5, 8)   # per-robot, not global
    assert attention_info["node_context"].shape == (8,)
    assert attention_info["robot_attn_weights"].shape == (5,)
    assert attention_info["peer_attn_weights"].shape == (5, 5)   # [N, N] robot-robot matrix
    assert attention_info["node_attn_weights"].shape == (6,)
    assert attention_info["robot_attn_weights"][1].item() == 0.0
    assert attention_info["robot_attn_weights"][4].item() == 0.0
