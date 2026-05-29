"""
LLM Embedding Engine
=====================
Wraps DistilBERT-base-uncased for zero-shot payload semantic embeddings.

The model is used as a pre-trained feature extractor WITHOUT fine-tuning,
following the zero-shot semantic embedding approach. Token-level embeddings
are mean-pooled to produce a fixed 128-dimensional endpoint feature vector.

These embeddings capture:
  - Instruction patterns in API parameters
  - Sensitive data field patterns in responses
  - Anomalous semantic patterns associated with injection payloads
    that purely structural graph features miss.

In CICIDS2017 evaluation the LLM module is not exercised (no payload
content). The 2.7pp improvement from LLM embeddings is measured on the
synthetic API call graph dataset (SQL injection sub-experiment).
"""

import torch
import numpy as np
from typing import List, Optional


class DistilBERTEmbedder:
    """
    Zero-shot DistilBERT-base-uncased payload embedder.

    Produces a 128-dimensional embedding for each API endpoint
    based on the aggregated payload tokens observed in a session.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: 'distilbert-base-uncased'.
    embedding_dim : int
        Output embedding dimension (truncated from BERT's 768).
        Paper value: 128.
    device : str
        'cpu' or 'cuda'.
    """

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
        """Lazy load model on first use."""
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
        """
        Embed a list of payload tokens into a single 128-dim vector.

        Tokens are joined into a text string, tokenised, passed through
        DistilBERT, and the [CLS] token representation is returned
        (truncated to embedding_dim).

        Parameters
        ----------
        tokens : list[str]
            Payload tokens for one API endpoint.

        Returns
        -------
        np.ndarray, shape (embedding_dim,)
        """
        self._load()

        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        text = " ".join(str(t) for t in tokens[:64])  # truncate long payloads

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Use CLS token representation
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, 768)
            embedding = cls_embedding.squeeze().cpu().numpy()

        # Truncate to embedding_dim
        return embedding[:self.embedding_dim].astype(np.float32)

    def compute_session_payload_scores(
        self,
        session: dict,
        endpoints: List[str],
    ) -> np.ndarray:
        """
        Compute per-endpoint embedding scores for a session.

        Parameters
        ----------
        session : dict
            Session with 'calls' list.
        endpoints : list[str]
            Ordered endpoint names.

        Returns
        -------
        np.ndarray, shape (n_endpoints, embedding_dim)
            Mean embedding per endpoint across all calls in the session.
        """
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
    """
    Lightweight alternative to DistilBERT for environments without
    a GPU or HuggingFace. Uses token-matching suspicious vocabulary
    to produce a scalar suspicion score per endpoint.

    This is the fallback used in the main experiments when the full
    LLM module is too slow for batch training.
    """

    SUSPICIOUS = {
        "SELECT", "DROP", "UNION", "INSERT", "DELETE", "UPDATE",
        "escalate", "admin", "bearer", "bulk", "export",
        "enumerate", "'", "OR", "AND", "--", "/*", "HAVING",
        "WAITFOR", "SLEEP", "base64", "eval",
    }

    def score(self, tokens: List[str]) -> float:
        """Return fraction of tokens that are suspicious."""
        if not tokens:
            return 0.0
        return sum(1 for t in tokens if t in self.SUSPICIOUS) / len(tokens)
