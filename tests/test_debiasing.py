import torch
import torch.nn.functional as F

from src.multi_agent_ppo.debiasing import (
    CausalMaskingDebiasing,
    ComprehensiveDebiasing,
    PermutationInvarianceDebiasing,
    StateDeltaDebiasing,
    TemporalConsistencyDebiasing,
)


def test_state_delta_debiasing_returns_zero_for_short_sequences():
    debias = StateDeltaDebiasing(lambda_debias=0.25)

    empty_loss = debias.compute_loss([], [], device=torch.device("cpu"))
    single_loss = debias.compute_loss([torch.tensor([1.0, 2.0])], [0])

    assert empty_loss.shape == torch.Size([])
    assert single_loss.shape == torch.Size([])
    assert empty_loss.item() == 0.0
    assert single_loss.item() == 0.0


def test_state_delta_debiasing_matches_expected_norm():
    debias = StateDeltaDebiasing(lambda_debias=0.5)
    state_sequence = [
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([3.0, 4.0, 0.0]),
        torch.tensor([6.0, 8.0, 0.0]),
    ]

    loss = debias.compute_loss(state_sequence, [0, 1])

    expected_delta = torch.tensor(5.0)
    expected_loss = 0.5 * expected_delta

    assert torch.isclose(loss, expected_loss)


def test_temporal_consistency_debiasing_only_triggers_above_threshold():
    debias = TemporalConsistencyDebiasing(lambda_consistency=0.2)
    logits_sequence = [
        torch.tensor([2.0, 0.0]),
        torch.tensor([0.0, 2.0]),
    ]
    similarity_matrix = torch.tensor([[1.0, 0.69], [0.69, 1.0]])

    below_threshold_loss = debias.compute_loss(logits_sequence, similarity_matrix)
    assert below_threshold_loss.item() == 0.0

    similarity_matrix[0, 1] = 0.9
    similarity_matrix[1, 0] = 0.9
    above_threshold_loss = debias.compute_loss(logits_sequence, similarity_matrix)

    probs_i = F.softmax(logits_sequence[0], dim=-1)
    probs_j = F.softmax(logits_sequence[1], dim=-1)
    expected_kl = F.kl_div(probs_j.log(), probs_i, reduction="batchmean")
    expected_loss = 0.2 * (expected_kl * 0.9)

    assert torch.isclose(above_threshold_loss, expected_loss)


def test_permutation_invariance_debiasing_matches_action_distribution_variance():
    debias = PermutationInvarianceDebiasing(lambda_permute=0.3)
    policy_outputs = [
        (torch.tensor([0.0]), torch.tensor([2.0, 0.0, -1.0])),
        (torch.tensor([1.0]), torch.tensor([0.0, 2.0, -1.0])),
    ]

    loss = debias.compute_loss(policy_outputs, [(0, 0), (1, 1)])

    action_dists = [F.softmax(logits, dim=-1) for _, logits in policy_outputs]
    expected = 0.3 * torch.var(torch.stack(action_dists), dim=0).mean()

    assert torch.isclose(loss, expected)


def test_causal_mask_is_lower_triangular_boolean_mask():
    mask = CausalMaskingDebiasing().create_causal_mask(4)

    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    assert mask.dtype is torch.bool
    assert torch.equal(mask, expected)


def test_comprehensive_debiasing_returns_breakdown_and_causal_mask():
    debias = ComprehensiveDebiasing(
        lambda_state_delta=0.1,
        lambda_consistency=0.05,
        use_causal_mask=False,
    )

    state_sequence = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 0.0]),
    ]
    action_sequence = [0]
    action_logits_sequence = [torch.tensor([1.0, 1.0])]
    task_features_sequence = [torch.tensor([1.0, 0.0])]

    total_loss, breakdown = debias.compute_total_debias_loss(
        state_sequence,
        action_sequence,
        action_logits_sequence,
        task_features_sequence,
    )

    assert total_loss.shape == torch.Size([])
    assert breakdown["debias_seq_len"] == 1
    assert breakdown["debias_sim_pairs"] == 0
    assert breakdown["consistency"] == 0.0
    assert breakdown["total_debias"] == breakdown["state_delta"]
    assert debias.get_causal_mask(3) is None
