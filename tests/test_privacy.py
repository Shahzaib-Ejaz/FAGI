import pytest
import torch
import torch.nn as nn
import numpy as np
from fagi.privacy.mvg_dp import MVGMechanism
from fagi.model.gnn import FAGIModel


class TestMVGMechanism:
    def test_noise_added_to_gradients(self):
        model = FAGIModel(in_dim=6)
        dp = MVGMechanism(clip_norm=1.0, sigma=1.1)

        from torch_geometric.data import Data
        import torch.nn.functional as F
        x = torch.randn(12, 6)
        ei = torch.LongTensor([[0, 1], [1, 0]])
        y = torch.LongTensor([1])
        out = model(x, ei)
        loss = F.cross_entropy(out, y)
        loss.backward()

        grad_before = {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

        dp.clip_and_noise(model)

        changed = False
        for name, p in model.named_parameters():
            if p.grad is not None and name in grad_before:
                if not torch.allclose(p.grad, grad_before[name]):
                    changed = True
                    break

        assert changed, "DP mechanism should modify at least one gradient"

    def test_clip_norm_applied(self):
        model = FAGIModel(in_dim=6)
        dp = MVGMechanism(clip_norm=1.0, sigma=0.0)

        from torch_geometric.data import Data
        import torch.nn.functional as F
        x = torch.randn(12, 6) * 100
        ei = torch.LongTensor([[0, 1], [1, 0]])
        y = torch.LongTensor([1])
        out = model(x, ei)
        loss = F.cross_entropy(out, y)
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                p.grad *= 1000.0

        dp.clip_and_noise(model)

        total_norm = torch.sqrt(sum(
            p.grad.norm() ** 2
            for p in model.parameters()
            if p.grad is not None
        )).item()

        assert total_norm <= 1.0 + 1e-4

    def test_epsilon_estimate_reasonable(self):
        eps = MVGMechanism.compute_epsilon(
            sigma=1.1, clip_norm=1.0, n_rounds=50, delta=1e-5
        )
        assert eps > 0, "Epsilon should be positive"
        assert np.isfinite(eps), "Epsilon should be finite"
        assert isinstance(eps, float)

    def test_higher_sigma_lower_epsilon(self):
        eps_low  = MVGMechanism.compute_epsilon(0.5, 1.0, 50, 1e-5)
        eps_high = MVGMechanism.compute_epsilon(2.0, 1.0, 50, 1e-5)
        assert isinstance(eps_low, float) and isinstance(eps_high, float)
        assert eps_high < eps_low

    def test_device_cpu(self):
        dp = MVGMechanism(device="cpu")
        model = FAGIModel(in_dim=6)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        dp.clip_and_noise(model)
