import torch
import numpy as np
from typing import List, Optional


class DistilBERTEmbedder:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embedding_dim: int = 128,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            from transformers import DistilBertModel, DistilBertTokenizer
            self._tokenizer = DistilBertTokenizer.from_pretrained(
                self.model_name
            )
            self._model = DistilBertModel.from_pretrained(
                self.model_name
            ).to(self.device)
            self._model.eval()

    def embed_tokens(self, tokens: List[str]) -> np.ndarray:
        self._load()

        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        text = " ".join(str(t) for t in tokens[:64])

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embedding = cls_embedding.squeeze().cpu().numpy()

        return embedding[:self.embedding_dim].astype(np.float32)

    def compute_session_payload_scores(
        self,
        session: dict,
        endpoints: List[str],
    ) -> np.ndarray:
        from collections import defaultdict
        ep_idx = {e: i for i, e in enumerate(endpoints)}
        ep_tokens = defaultdict(list)

        for call in session.get("calls", []):
            dst = call.get("dst", "")
            tokens = call.get("payload_tokens", [])
            ep_tokens[dst].extend(tokens)

        embeddings = np.zeros(
            (len(endpoints), self.embedding_dim), dtype=np.float32
        )
        for ep_name, tokens in ep_tokens.items():
            if ep_name in ep_idx and tokens:
                embeddings[ep_idx[ep_name]] = self.embed_tokens(tokens)

        return embeddings


class SimplePayloadScorer:
    SUSPICIOUS = {
        "SELECT", "DROP", "UNION", "INSERT", "DELETE", "UPDATE",
        "escalate", "admin", "bearer", "bulk", "export",
        "enumerate", "'", "OR", "AND", "--", "/*", "HAVING",
        "WAITFOR", "SLEEP", "base64", "eval",
    }

    def score(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        return sum(1 for t in tokens if t in self.SUSPICIOUS) / len(tokens)
