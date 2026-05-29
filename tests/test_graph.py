"""
Unit tests for APIGraphConstructor.
Run with: pytest tests/test_graph.py -v
"""

import pytest
import numpy as np
import torch
from fagi.graph.constructor import APIGraphConstructor, DEFAULT_ENDPOINTS


# ── Fixtures ──────────────────────────────────────────────────────────

def make_benign_session():
    return {
        "calls": [
            {"dst": "auth",       "duration_ms": 420, "response_size": 800,
             "privilege_level": 0, "method": "POST", "payload_tokens": ["ok"]},
            {"dst": "user_read",  "duration_ms": 310, "response_size": 1200,
             "privilege_level": 1, "method": "GET",  "payload_tokens": ["data"]},
            {"dst": "data_query", "duration_ms": 550, "response_size": 3400,
             "privilege_level": 2, "method": "POST", "payload_tokens": ["json"]},
        ],
        "label": 0,
    }


def make_lateral_movement_session():
    return {
        "calls": [
            {"dst": "auth",       "duration_ms": 400, "response_size": 800,
             "privilege_level": 0, "method": "POST", "payload_tokens": []},
            {"dst": "user_write", "duration_ms": 600, "response_size": 1000,
             "privilege_level": 2, "method": "POST", "payload_tokens": ["escalate"]},
            {"dst": "admin_write","duration_ms": 900, "response_size": 1500,
             "privilege_level": 4, "method": "POST", "payload_tokens": ["admin"]},
            {"dst": "data_export","duration_ms": 1200,"response_size": 48000,
             "privilege_level": 3, "method": "GET",  "payload_tokens": ["bulk"]},
        ],
        "label": 1,
    }


# ── Tests ─────────────────────────────────────────────────────────────

class TestAPIGraphConstructor:

    def test_graph_shape_with_payload(self):
        """Graph should have correct node and edge dimensions."""
        constructor = APIGraphConstructor(include_payload=True)
        session = make_benign_session()
        graph = constructor.session_to_graph(session)

        assert graph.x.shape == (len(DEFAULT_ENDPOINTS), 6), \
            "Node features should be (n_endpoints, 6) with payload"
        assert graph.y.item() == 0

    def test_graph_shape_without_payload(self):
        """FAGI−LLM ablation: 5 features per node."""
        constructor = APIGraphConstructor(include_payload=False)
        session = make_benign_session()
        graph = constructor.session_to_graph(session)

        assert graph.x.shape == (len(DEFAULT_ENDPOINTS), 5)

    def test_edges_created_from_call_sequence(self):
        """Sequential calls should create directed edges."""
        constructor = APIGraphConstructor()
        session = make_benign_session()
        graph = constructor.session_to_graph(session)

        # 3 calls → 2 sequential edges → 4 edges (bidirectional)
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] >= 2, "Should have at least one edge"

    def test_empty_calls_handled(self):
        """Empty session should not crash (self-loop fallback)."""
        constructor = APIGraphConstructor()
        graph = constructor.session_to_graph({"calls": [], "label": 0})
        assert graph.edge_index.shape[1] > 0, "Should have fallback self-loop"

    def test_privilege_level_encoded(self):
        """High-privilege endpoint should have high max_privilege feature."""
        constructor = APIGraphConstructor(include_payload=True)
        session = make_lateral_movement_session()
        graph = constructor.session_to_graph(session)

        # admin_write is endpoint index 4 (privilege=4)
        from fagi.graph.constructor import DEFAULT_ENDPOINTS
        admin_idx = DEFAULT_ENDPOINTS.index("admin_write")
        assert graph.x[admin_idx, 3] == 4.0, \
            "admin_write node should have max_privilege = 4"

    def test_payload_suspicion_score(self):
        """Suspicious tokens should increase payload_susp_score."""
        constructor = APIGraphConstructor(include_payload=True)
        # SQL injection session
        session = {
            "calls": [
                {"dst": "data_query", "duration_ms": 400, "response_size": 500,
                 "privilege_level": 2, "method": "POST",
                 "payload_tokens": ["SELECT", "FROM", "UNION"]},
            ],
            "label": 1,
        }
        graph = constructor.session_to_graph(session)
        query_idx = DEFAULT_ENDPOINTS.index("data_query")
        assert graph.x[query_idx, 5] > 0, \
            "SQL tokens should produce non-zero payload_susp_score"

    def test_batch_conversion(self):
        """Batch conversion should handle mixed sessions."""
        constructor = APIGraphConstructor()
        sessions = [make_benign_session(), make_lateral_movement_session()]
        graphs = constructor.sessions_to_graphs(sessions)
        assert len(graphs) == 2
        assert graphs[0].y.item() == 0
        assert graphs[1].y.item() == 1

    def test_avg_duration_normalised(self):
        """Duration should be averaged (not summed) across calls to an endpoint."""
        constructor = APIGraphConstructor(include_payload=False)
        session = {
            "calls": [
                {"dst": "auth", "duration_ms": 400.0, "response_size": 800,
                 "privilege_level": 0, "method": "GET", "payload_tokens": []},
                {"dst": "auth", "duration_ms": 600.0, "response_size": 800,
                 "privilege_level": 0, "method": "GET", "payload_tokens": []},
            ],
            "label": 0,
        }
        graph = constructor.session_to_graph(session)
        auth_idx = DEFAULT_ENDPOINTS.index("auth")
        avg_dur = graph.x[auth_idx, 1].item()
        assert abs(avg_dur - 500.0) < 1.0, \
            f"Average duration should be 500ms, got {avg_dur:.1f}"

    def test_feature_names(self):
        """Feature name list should match dimension count."""
        c_with = APIGraphConstructor(include_payload=True)
        c_without = APIGraphConstructor(include_payload=False)
        assert len(c_with.node_feature_names) == 6
        assert len(c_without.node_feature_names) == 5
