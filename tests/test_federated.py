import pytest
import copy
import torch
import numpy as np
from torch_geometric.data import Data

from fagi.model.gnn import FAGIModel
from fagi.federated.server import FederatedServer
from fagi.federated.client import FederatedClient


def make_simple_graph(label=0):
    x = torch.randn(12, 6)
    ei = torch.LongTensor([[0, 1, 2], [1, 2, 0]])
    return Data(x=x, edge_index=ei, y=torch.LongTensor([label]))


class TestFederatedServer:
    def test_fedavg_averages_weights(self):
        global_model = FAGIModel(in_dim=6)
        server = FederatedServer(global_model)

        client1 = FAGIModel(in_dim=6)
        client2 = FAGIModel(in_dim=6)

        with torch.no_grad():
            for p in client1.parameters():
                p.fill_(1.0)
            for p in client2.parameters():
                p.fill_(3.0)

        server.aggregate([client1, client2])

        for p in global_model.parameters():
            assert torch.allclose(p, torch.full_like(p, 2.0), atol=0.01), \
                "FedAvg should produce mean of client weights"

    def test_single_client_no_change(self):
        global_model = FAGIModel(in_dim=6)
        client = FAGIModel(in_dim=6)
        server = FederatedServer(global_model)

        with torch.no_grad():
            for p in client.parameters():
                p.fill_(5.0)

        server.aggregate([client])

        for gp, cp in zip(global_model.parameters(), client.parameters()):
            assert torch.allclose(gp, cp, atol=0.01)

    def test_byzantine_filter_excludes_outliers(self):
        global_model = FAGIModel(in_dim=6)
        server = FederatedServer(global_model, byzantine_threshold=1.0)

        normal_clients = []
        for _ in range(3):
            m = FAGIModel(in_dim=6)
            with torch.no_grad():
                for p in m.parameters():
                    p.fill_(1.0)
            normal_clients.append(m)

        outlier = FAGIModel(in_dim=6)
        with torch.no_grad():
            for p in outlier.parameters():
                p.fill_(1000.0)

        server.aggregate(normal_clients + [outlier])

        for p in global_model.parameters():
            assert p.abs().mean() < 100, "Byzantine filter should exclude outlier"

    def test_convergence_check(self):
        model = FAGIModel(in_dim=6)
        server = FederatedServer(model)
        prev_model = copy.deepcopy(model)
        assert server.convergence_check(prev_model, tau=1e-4) is True

    def test_convergence_check_not_converged(self):
        model = FAGIModel(in_dim=6)
        server = FederatedServer(model)
        prev_model = copy.deepcopy(model)

        with torch.no_grad():
            for p in model.parameters():
                p += 100.0

        assert server.convergence_check(prev_model, tau=1e-4) is False


class TestFederatedClient:
    def test_local_train_returns_model(self):
        graphs = [make_simple_graph(label=i % 2) for i in range(20)]
        client = FederatedClient("org_0", graphs)
        global_model = FAGIModel(in_dim=6)
        local_model = client.local_train(global_model, epochs=2, batch_size=10)
        assert isinstance(local_model, FAGIModel)

    def test_local_train_modifies_weights(self):
        graphs = [make_simple_graph(label=i % 2) for i in range(30)]
        client = FederatedClient("org_0", graphs)
        global_model = FAGIModel(in_dim=6)
        weights_before = {
            n: p.clone() for n, p in global_model.named_parameters()
        }
        local_model = client.local_train(global_model, epochs=3, batch_size=20)
        changed = any(
            not torch.allclose(p, weights_before[n])
            for n, p in local_model.named_parameters()
        )
        assert changed, "Local training should modify model weights"

    def test_empty_data_does_not_crash(self):
        client = FederatedClient("org_empty", [])
        global_model = FAGIModel(in_dim=6)
        result = client.local_train(global_model, epochs=1, batch_size=10)
        assert isinstance(result, FAGIModel)
