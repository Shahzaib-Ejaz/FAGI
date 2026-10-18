import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from fagi.graph.constructor import APIGraphConstructor
from fagi.graph.dense import NodeFeatureScaler, sessions_to_dense
from fagi.model.dense import DenseFAGI, DenseSAGEConv, dense_gnnguard, pgd_attack_dense
from fagi.model.gnn import FAGIModel
from fagi.model.gnnguard import gnnguard_filter


def to_dense(ei, n):
    A = torch.zeros(n, n)
    A.index_put_((ei[1], ei[0]), torch.ones(ei.shape[1]), accumulate=True)
    return A


def copy_weights(dense, sparse):
    for a, b in [(dense.conv1, sparse.conv1), (dense.conv2, sparse.conv2)]:
        a.lin_l.weight.data = b.lin_l.weight.data.clone()
        a.lin_l.bias.data = b.lin_l.bias.data.clone()
        a.lin_r.weight.data = b.lin_r.weight.data.clone()
    dense.fc.load_state_dict(sparse.fc.state_dict())


class TestDenseEquivalence:
    def test_layer_matches_sageconv(self):
        torch.manual_seed(0)
        x = torch.randn(12, 6)
        ei = torch.tensor([[0, 1, 2, 2, 5, 0], [1, 2, 0, 0, 0, 5]])
        s = SAGEConv(6, 8, aggr="mean")
        d = DenseSAGEConv(6, 8)
        d.lin_l.weight.data = s.lin_l.weight.data.clone()
        d.lin_l.bias.data = s.lin_l.bias.data.clone()
        d.lin_r.weight.data = s.lin_r.weight.data.clone()
        assert torch.allclose(s(x, ei), d(x, to_dense(ei, 12)), atol=1e-6)

    def test_model_matches_sparse_model_with_gnnguard(self):
        torch.manual_seed(1)
        sparse = FAGIModel(in_dim=6, gnnguard_threshold=0.15).eval()
        dense = DenseFAGI(6, gnnguard_threshold=0.15).eval()
        copy_weights(dense, sparse)
        for _ in range(5):
            x = torch.randn(12, 6)
            ei = torch.randint(0, 12, (2, 30))
            assert torch.allclose(sparse(x, ei), dense(x[None], to_dense(ei, 12)[None]), atol=1e-5)

    def test_gnnguard_matches_sparse_filter(self):
        torch.manual_seed(2)
        x = torch.randn(12, 6)
        ei = torch.randint(0, 12, (2, 40))
        assert torch.equal(dense_gnnguard(x, to_dense(ei, 12), 0.15), to_dense(gnnguard_filter(x, ei, 0.15), 12))

    def test_sessions_to_dense_matches_constructor(self):
        s = {"calls": [{"dst": "auth"}, {"dst": "user_read", "payload_tokens": ["SELECT"]}, {"dst": "auth"}], "label": 1}
        X, A, M, y = sessions_to_dense([s])
        g = APIGraphConstructor().session_to_graph(s)
        assert np.allclose(X[0], g.x.numpy())
        assert np.allclose(A[0], to_dense(g.edge_index, 12).numpy())
        assert y[0] == 1 and M[0, 0] == 1 and M[0, 2] == 0

    def test_scaler_keeps_inactive_nodes_zero(self):
        X = np.random.rand(10, 12, 6).astype(np.float32)
        M = np.zeros((10, 12), dtype=np.float32)
        M[:, :3] = 1
        X[:, 3:] = 0
        Z = NodeFeatureScaler().fit(X, M).transform(X)
        assert np.all(Z[:, 3:] == 0)


class TestPGDDense:
    def test_perturbation_respects_budget_and_mask(self):
        torch.manual_seed(3)
        m = DenseFAGI(5, gnnguard_threshold=None, readout="center")
        X = torch.randn(8, 9, 5)
        A = torch.ones(8, 9, 9)
        Y = torch.randint(0, 2, (8,))
        mask = torch.zeros(8, 9)
        mask[:, 0] = 1
        xa = pgd_attack_dense(m, X, A, Y, epsilon=0.1, steps=5, node_mask=mask)
        delta = xa - X
        assert delta.abs().max() <= 0.1 + 1e-6
        assert torch.all(delta[:, 1:] == 0)

    def test_attack_does_not_decrease_loss(self):
        torch.manual_seed(4)
        m = DenseFAGI(5, gnnguard_threshold=None).eval()
        X = torch.randn(32, 12, 5)
        A = (torch.rand(32, 12, 12) > 0.7).float()
        Y = torch.randint(0, 2, (32,))
        xa = pgd_attack_dense(m, X, A, Y, epsilon=0.5, steps=10, random_start=False)
        with torch.no_grad():
            assert F.cross_entropy(m(xa, A), Y) >= F.cross_entropy(m(X, A), Y)
