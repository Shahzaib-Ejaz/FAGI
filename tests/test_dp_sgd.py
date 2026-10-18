import pytest
import torch
import torch.nn.functional as F

from fagi.model.dense import DenseFAGI
from fagi.privacy.dp_sgd import DPSGD, compute_epsilon, make_graph_loss


def setup(B=6):
    torch.manual_seed(0)
    m = DenseFAGI(6, gnnguard_threshold=None)
    m.dropout.p = 0.0
    X = torch.randn(B, 12, 6)
    A = (torch.rand(B, 12, 12) > 0.5).float()
    Y = torch.randint(0, 2, (B,))
    return m, X, A, Y


class TestDPSGD:
    def test_per_sample_grads_match_loop(self):
        m, X, A, Y = setup()
        opt = DPSGD(m, n_train=100, batch_size=10)
        pg = opt.per_sample_grads(make_graph_loss(m), X, A, Y)
        for i in range(len(Y)):
            m.zero_grad()
            F.cross_entropy(m(X[i:i + 1], A[i:i + 1]), Y[i:i + 1]).backward()
            for k, p in m.named_parameters():
                assert torch.allclose(p.grad, pg[k][i], atol=1e-6)

    def test_each_sample_contribution_is_clipped(self):
        m, X, A, Y = setup()
        opt = DPSGD(m, n_train=100, batch_size=10, sigma=0.0, clip_norm=0.5)
        pg = opt.per_sample_grads(make_graph_loss(m), X * 50, A, Y)
        for i in range(len(Y)):
            single = {k: v[i:i + 1] for k, v in pg.items()}
            g = opt.clipped_noisy_grad(single)
            norm = torch.sqrt(sum((v * opt.expected_batch).pow(2).sum() for v in g.values()))
            assert norm <= 0.5 + 1e-4

    def test_noise_scale(self):
        m, _, _, _ = setup()
        opt = DPSGD(m, n_train=1000, batch_size=1000, sigma=1.1, clip_norm=1.0,
                    generator=torch.Generator().manual_seed(0))
        g = opt.clipped_noisy_grad(None)
        flat = torch.cat([v.flatten() for v in g.values()]) * opt.expected_batch
        assert abs(flat.std().item() - 1.1) < 0.05

    def test_accountant_steps_and_epsilon(self):
        m, X, A, Y = setup()
        opt = DPSGD(m, n_train=100, batch_size=10, sigma=1.1)
        for _ in range(3):
            opt.apply(opt.clipped_noisy_grad(opt.per_sample_grads(make_graph_loss(m), X, A, Y)))
        assert opt.steps == 3
        assert opt.epsilon() == pytest.approx(compute_epsilon(1.1, 0.1, 3, 1e-5))

    def test_epsilon_monotonic(self):
        assert compute_epsilon(1.1, 0.01, 2000) > compute_epsilon(1.1, 0.01, 1000)
        assert compute_epsilon(2.0, 0.01, 1000) < compute_epsilon(1.1, 0.01, 1000)
        assert compute_epsilon(1.1, 0.02, 1000) > compute_epsilon(1.1, 0.01, 1000)

    def test_known_value(self):
        assert compute_epsilon(1.1, 0.02, 1250, 1e-5) == pytest.approx(4.02, abs=0.05)

    def test_poisson_batch_rate(self):
        m, _, _, _ = setup()
        opt = DPSGD(m, n_train=10000, batch_size=500, generator=torch.Generator().manual_seed(1))
        sizes = [len(opt.poisson_batch()) for _ in range(20)]
        assert 450 < sum(sizes) / 20 < 550
