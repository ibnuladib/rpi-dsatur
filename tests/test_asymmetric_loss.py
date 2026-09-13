import pytest
import torch
from labels.build_labels import compute_shell_conflict_density
from model.train import AsymmetricOrdinalLoss


def test_underprediction_costs_more_than_overprediction():
    loss_fn = AsymmetricOrdinalLoss(6)
    logits_4_to_0 = torch.zeros(1, 6)
    logits_4_to_0[0, 0] = 8.0
    logits_0_to_4 = torch.zeros(1, 6)
    logits_0_to_4[0, 4] = 8.0
    under = loss_fn(logits_4_to_0, torch.tensor([4]))
    over = loss_fn(logits_0_to_4, torch.tensor([0]))
    assert under > over


def test_correct_class_beats_far_underprediction():
    loss_fn = AsymmetricOrdinalLoss(6)
    logits_ok = torch.zeros(1, 6)
    logits_ok[0, 4] = 8.0
    logits_bad = torch.zeros(1, 6)
    logits_bad[0, 0] = 8.0
    target = torch.tensor([4])
    assert loss_fn(logits_ok, target) < loss_fn(logits_bad, target)


def test_shell_conflict_density_per_shell_mean():
    x = torch.tensor([
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.00],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.25],
        [0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.50],
    ])
    dens = compute_shell_conflict_density(x, hop_radius=4)
    assert dens[0] == pytest.approx(1.0)
    assert dens[1] == pytest.approx(0.5)
    assert dens[2] == pytest.approx(0.25)
    assert dens[3] == 0.0
    assert dens[4] == 0.0
