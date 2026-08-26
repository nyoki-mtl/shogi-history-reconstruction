import torch

from posdiffusion.reverse.scoring import positions_for_mode, score_candidate_tokens


def test_positions_for_mode_changed_uses_candidate_specific_mask() -> None:
    source = torch.zeros(96, dtype=torch.long)
    candidates = torch.zeros((2, 96), dtype=torch.long)
    candidates[0, 0] = 1
    candidates[1, 95] = 1

    mask = positions_for_mode(mode="changed", source_tokens=source, candidate_tokens=candidates)

    assert mask[0].sum().item() == 1
    assert mask[0, 0]
    assert mask[1].sum().item() == 1
    assert mask[1, 95]


def test_score_candidate_tokens_ranks_higher_log_probability_candidate() -> None:
    source = torch.zeros(96, dtype=torch.long)
    candidates = torch.zeros((2, 96), dtype=torch.long)
    candidates[0, 0] = 1
    candidates[1, 0] = 2
    logits = torch.zeros((96, 40), dtype=torch.float32)
    logits[0, 1] = 3.0
    logits[0, 2] = 1.0

    scores = score_candidate_tokens(logits, source, candidates, mode="changed").scores

    assert scores[0] > scores[1]
