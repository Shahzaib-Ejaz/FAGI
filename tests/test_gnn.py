import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from fagi.model.gnn import FAGIModel
from fagi.model.gnnguard import gnnguard_filter
from fagi.model.pgd import pgd_attack, adversarial_loss


def make_graph(n_nodes=12, in_dim=6, n_edges=5, label=1):
    x = torch.randn(n_nodes, in_dim)
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    ei = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ])
    return Data(x=x, edge_index=ei, y=torch.LongTensor([label]))


class TestFAGIModel:
    def test_output_shape(self):
        model = FAGIModel(in_dim=6)
        g = make_graph(in_dim=6)
        out = model(g.x, g.edge_index)
        assert out.shape == (1, 2), "Output should be (1, n_classes)"

    def test_ablation_no_gnnguard(self):
        model = FAGIModel(in_dim=6, gnnguard_threshold=None)
        g = make_graph(in_dim=6)
        out = model(g.x, g.edge_index)
        assert out.shape == (1, 2)

    def test_predict_returns_int(self):
        model = FAGIModel(in_dim=6)
        g = make_graph(in_dim=6)
        pred = model.predict(g.x, g.edge_index)
        assert isinstance(pred, int)
        assert pred in (0, 1)

    def test_forward_deterministic_in_eval(self):
        model = FAGIModel(in_dim=6)
        model.eval()
        g = make_graph(in_dim=6)
        out1 = model(g.x, g.edge_index)
        out2 = model(g.x, g.edge_index)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


class TestGNNGuard:
    def test_filter_removes_dissimilar_edges(self):
        x = torch.zeros(4, 6)
        x[0] = torch.ones(6)
        x[1] = -torch.ones(6)
        x[2] = torch.ones(6)
        ei = torch.LongTensor([[0, 0], [1, 2]])

        filtered = gnnguard_filter(x, ei, threshold=0.15)
        assert filtered.shape[1] < ei.shape[1], "Dissimilar edge should be removed"

    def test_filter_keeps_similar_edges(self):
        x = torch.ones(4, 6)
        ei = torch.LongTensor([[0, 1, 2], [1, 2, 3]])
        filtered = gnnguard_filter(x, ei, threshold=0.15)
        assert filtered.shape[1] == ei.shape[1], "All edges should be kept"

    def test_safety_fallback_empty(self):
        x = torch.zeros(3, 6)
        ei = torch.LongTensor([[0, 1], [1, 2]])
        filtered = gnnguard_filter(x, ei, threshold=0.9)
        assert filtered.shape[1] > 0, "Should return original on all-pruned case"

    def test_empty_edge_index(self):
        x = torch.randn(4, 6)
        ei = torch.LongTensor([[], []])
        filtered = gnnguard_filter(x, ei, threshold=0.15)
        assert filtered.shape[1] == 0


class TestPGD:
    def test_pgd_changes_features(self):
        model = FAGIModel(in_dim=6)
        model.eval()
        g = make_graph(in_dim=6, label=1)
        x_adv = pgd_attack(model, g.x, g.edge_index, g.y,
                            epsilon=0.1, steps=5)
        assert not torch.allclose(g.x, x_adv), "PGD should modify features"

    def test_pgd_respects_epsilon(self):
        model = FAGIModel(in_dim=6)
        model.eval()
        epsilon = 0.1
        g = make_graph(in_dim=6, label=1)
        x_adv = pgd_attack(model, g.x, g.edge_index, g.y,
                            epsilon=epsilon, steps=10)
        delta = (x_adv - g.x).abs().max().item()
        assert delta <= epsilon + 1e-5, \
            f"Max perturbation {delta:.4f} exceeds epsilon {epsilon}"

    def test_adversarial_loss_finite(self):
        model = FAGIModel(in_dim=6)
        g = make_graph(in_dim=6, label=1)
        loss = adversarial_loss(model, g.x, g.edge_index, g.y,
                                epsilon=0.1, pgd_steps=5)
        assert torch.isfinite(loss), "Adversarial loss must be finite"

    def test_adversarial_loss_equal_weight(self):
        model = FAGIModel(in_dim=6)
        model.eval()
        g = make_graph(in_dim=6, label=0)
        import torch.nn.functional as F
        loss = adversarial_loss(model, g.x, g.edge_index, g.y,
                                lambda_adv=0.5, pgd_steps=2)
        clean_loss = F.cross_entropy(model(g.x, g.edge_index), g.y)
        ratio = loss.item() / (clean_loss.item() + 1e-8)
        assert 0.1 < ratio < 10, "Adversarial loss should be in sane range"
