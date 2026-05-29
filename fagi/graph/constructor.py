"""
API Graph Constructor
=====================
Translates raw API gateway log sessions into attributed call graphs
G = (V, E, X) where:
  V = unique API endpoints observed in a detection window W
  E ⊆ V × V = call dependency edges weighted by frequency and latency
  X ∈ ℝ^(|V|×d) = node feature matrix encoding endpoint attributes

Node features per endpoint (d=6):
  [call_count, avg_duration_ms, avg_response_size,
   max_privilege_level, write_ratio, payload_susp_score]

The payload_susp_score is filled by the LLM Embedding Engine;
it is zero when running without the LLM module (FAGI−LLM ablation).
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Optional


# Standard API endpoint names recognised by the constructor.
# In production these are learned from the gateway schema.
DEFAULT_ENDPOINTS = [
    "auth", "user_read", "user_write", "admin_read", "admin_write",
    "data_export", "data_query", "health", "schema",
    "billing", "notify", "search",
]

# Suspicious payload tokens used to compute payload_susp_score.
# Drawn from OWASP API Security Top 10 injection patterns.
SUSPICIOUS_TOKENS = {
    "SELECT", "DROP", "UNION", "INSERT", "DELETE", "UPDATE",
    "escalate", "admin", "bearer", "bulk", "export",
    "enumerate", "'", "OR", "AND", "--", "/*",
}


class APIGraphConstructor:
    """
    Converts a list of API call sessions into PyTorch Geometric Data objects.

    Each session is a list of call dictionaries with fields:
        dst          : str   — destination endpoint name
        duration_ms  : float — call latency in milliseconds
        response_size: float — response payload size in bytes
        privilege_level: int — endpoint privilege level (0–4)
        method       : str   — HTTP method (GET, POST, PUT, DELETE, …)
        payload_tokens: list[str] — tokenised payload content

    Parameters
    ----------
    endpoints : list[str], optional
        Ordered list of known endpoint names. If None, uses DEFAULT_ENDPOINTS.
    include_payload : bool
        Whether to include payload_susp_score as node feature (dim 6 vs 5).
        Set False for the FAGI−LLM ablation.
    """

    def __init__(
        self,
        endpoints: Optional[List[str]] = None,
        include_payload: bool = True,
    ):
        self.endpoints = endpoints or DEFAULT_ENDPOINTS
        self.ep_idx: Dict[str, int] = {e: i for i, e in enumerate(self.endpoints)}
        self.include_payload = include_payload
        self.n_nodes = len(self.endpoints)
        self.feature_dim = 6 if include_payload else 5

    def session_to_graph(self, session: dict) -> Data:
        """
        Convert one API call session to a PyG Data object.

        Parameters
        ----------
        session : dict with keys:
            'calls' : list[dict]  — ordered API calls
            'label' : int         — 0 = benign, 1 = malicious

        Returns
        -------
        torch_geometric.data.Data
            x          : FloatTensor (n_nodes, feature_dim)
            edge_index : LongTensor  (2, n_edges) — bidirectional
            y          : LongTensor  (1,)
        """
        calls = session["calls"]
        label = session.get("label", -1)
        n = self.n_nodes
        nf = np.zeros((n, self.feature_dim), dtype=np.float32)
        edge_src, edge_dst = [], []
        prev_ep = None

        for call in calls:
            di = self.ep_idx.get(call["dst"], 0)

            # Accumulate per-node statistics
            nf[di, 0] += 1                                      # call count
            nf[di, 1] += call.get("duration_ms", 0.0)          # total latency
            nf[di, 2] += call.get("response_size", 0.0)        # total size
            nf[di, 3] = max(nf[di, 3],
                            call.get("privilege_level", 0))     # max privilege
            nf[di, 4] += 1 if call.get("method", "GET") in (
                "POST", "PUT", "DELETE", "PATCH"
            ) else 0                                            # write count

            if self.include_payload:
                tokens = call.get("payload_tokens", [])
                nf[di, 5] += sum(
                    1 for t in tokens if t in SUSPICIOUS_TOKENS
                )                                               # suspicion score

            # Sequential call edges (directed: prev → current)
            if prev_ep is not None:
                pi = self.ep_idx.get(prev_ep, 0)
                edge_src.append(pi)
                edge_dst.append(di)
            prev_ep = call["dst"]

        # Normalise latency and size to per-call averages
        for i in range(n):
            if nf[i, 0] > 0:
                nf[i, 1] /= nf[i, 0]   # avg duration
                nf[i, 2] /= nf[i, 0]   # avg size
                nf[i, 4] /= nf[i, 0]   # write ratio

        # Ensure at least one self-loop to avoid isolated graph
        if not edge_src:
            edge_src = [0]
            edge_dst = [0]

        # Make edges bidirectional for undirected message passing
        ei = torch.LongTensor(
            [edge_src + edge_dst, edge_dst + edge_src]
        )

        return Data(
            x=torch.FloatTensor(nf),
            edge_index=ei,
            y=torch.LongTensor([label]),
        )

    def sessions_to_graphs(self, sessions: List[dict]) -> List[Data]:
        """Batch conversion of sessions to graphs."""
        return [self.session_to_graph(s) for s in sessions]

    @property
    def node_feature_names(self) -> List[str]:
        """Human-readable names for node feature dimensions."""
        names = [
            "call_count",
            "avg_duration_ms",
            "avg_response_size_bytes",
            "max_privilege_level",
            "write_ratio",
        ]
        if self.include_payload:
            names.append("payload_susp_score")
        return names
