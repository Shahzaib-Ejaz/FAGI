import numpy as np
import pytest
import torch

from fagi.model.dense import DenseFAGI
from fagi.training import train_model


def data(seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(64, 12, 6).astype(np.float32)
    A = (rng.rand(64, 12, 12) > 0.7).astype(np.float32)
    y = rng.randint(0, 2, 64)
    M = np.ones((64, 12), dtype=np.float32)
    return X, A, y, M


@pytest.mark.parametrize("adv,dp", [(False, False), (True, False), (False, True), (True, True)])
def test_same_seed_same_weights_regardless_of_rng_history(adv, dp):
    X, A, y, M = data()
    states = []
    for burn in (0, 17):
        torch.manual_seed(1234 + burn)
        np.random.seed(burn)
        _ = torch.randn(burn + 1)
        m = DenseFAGI(6, gnnguard_threshold=0.15)
        train_model(m, X, A, y, node_mask=M, epochs=2, batch_size=16, adv=adv, dp=dp, seed=5, pgd_steps=2)
        states.append({k: v.clone() for k, v in m.state_dict().items()})
    for k in states[0]:
        assert torch.equal(states[0][k], states[1][k]), k


def test_reinit_false_keeps_starting_weights():
    X, A, y, M = data()
    m = DenseFAGI(6, gnnguard_threshold=None)
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(0.01)
    train_model(m, X, A, y, epochs=1, batch_size=64, seed=0, lr=0.0, weight_decay=0.0, reinit=False)
    assert all(torch.allclose(p, torch.full_like(p, 0.01)) for p in m.parameters())
