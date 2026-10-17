import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Optional


DEFAULT_ENDPOINTS = [
    "auth", "user_read", "user_write", "admin_read", "admin_write",
    "data_export", "data_query", "health", "schema",
    "billing", "notify", "search",
]

SUSPICIOUS_TOKENS = {
    "SELECT", "DROP", "UNION", "INSERT", "DELETE", "UPDATE",
    "escalate", "admin", "bearer", "bulk", "export",
    "enumerate", "'", "OR", "AND", "--", "/*",
}


class APIGraphConstructor:
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
        calls = session["calls"]
        label = session.get("label", -1)
        n = self.n_nodes
        nf = np.zeros((n, self.feature_dim), dtype=np.float32)
        edge_src, edge_dst = [], []
        prev_ep = None

        for call in calls:
            di = self.ep_idx.get(call["dst"], 0)

            nf[di, 0] += 1
            nf[di, 1] += call.get("duration_ms", 0.0)
            nf[di, 2] += call.get("response_size", 0.0)
            nf[di, 3] = max(nf[di, 3],
                            call.get("privilege_level", 0))
            nf[di, 4] += 1 if call.get("method", "GET") in (
                "POST", "PUT", "DELETE", "PATCH"
            ) else 0

            if self.include_payload:
                tokens = call.get("payload_tokens", [])
                nf[di, 5] += sum(
                    1 for t in tokens if t in SUSPICIOUS_TOKENS
                )

            if prev_ep is not None:
                pi = self.ep_idx.get(prev_ep, 0)
                edge_src.append(pi)
                edge_dst.append(di)
            prev_ep = call["dst"]

        for i in range(n):
            if nf[i, 0] > 0:
                nf[i, 1] /= nf[i, 0]
                nf[i, 2] /= nf[i, 0]
                nf[i, 4] /= nf[i, 0]

        if not edge_src:
            edge_src = [0]
            edge_dst = [0]

        ei = torch.LongTensor(
            [edge_src + edge_dst, edge_dst + edge_src]
        )

        return Data(
            x=torch.FloatTensor(nf),
            edge_index=ei,
            y=torch.LongTensor([label]),
        )

    def sessions_to_graphs(self, sessions: List[dict]) -> List[Data]:
        return [self.session_to_graph(s) for s in sessions]

    @property
    def node_feature_names(self) -> List[str]:
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
