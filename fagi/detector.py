from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from fagi.graph.dense import NodeFeatureScaler, sessions_to_dense
from fagi.model.dense import DenseFAGI
from fagi.training import T, adversarial_inputs, predict, train_model


class FAGIDetector:
    def __init__(
        self,
        hidden_dims: tuple = (64, 32),
        pgd_epsilon: float = 0.1,
        pgd_steps: int = 10,
        gnnguard_threshold: Optional[float] = 0.15,
        use_adversarial_training: bool = True,
        use_dp: bool = True,
        dp_delta: float = 1e-5,
        dp_sigma: float = 1.1,
        dp_clip_norm: float = 1.0,
        include_payload: bool = True,
        lr: float = 2e-3,
        batch_size: int = 32,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.hidden_dims = hidden_dims
        self.pgd_epsilon = pgd_epsilon
        self.pgd_steps = pgd_steps
        self.gnnguard_threshold = gnnguard_threshold
        self.use_adversarial_training = use_adversarial_training
        self.use_dp = use_dp
        self.dp_delta = dp_delta
        self.dp_sigma = dp_sigma
        self.dp_clip_norm = dp_clip_norm
        self.include_payload = include_payload
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.feature_dim = 6 if include_payload else 5
        self.model = DenseFAGI(self.feature_dim, hidden_dims=hidden_dims,
                               gnnguard_threshold=gnnguard_threshold, readout="mean")
        self.scaler: Optional[NodeFeatureScaler] = None
        self.train_info: dict = {}
        self._is_fitted = False

    @property
    def epsilon(self) -> Optional[float]:
        return self.train_info.get("epsilon")

    def _dense(self, sessions: List[dict]):
        X, A, M, y = sessions_to_dense(sessions, include_payload=self.include_payload)
        return X, A, M, y

    def fit(self, sessions: List[dict], epochs: int = 30, verbose: bool = True) -> "FAGIDetector":
        if not sessions:
            raise ValueError("fit() needs at least one session")
        X, A, M, y = self._dense(sessions)
        self.scaler = NodeFeatureScaler().fit(X, M)
        self.train_info = train_model(
            self.model, self.scaler.transform(X), A, y, node_mask=M, epochs=epochs,
            batch_size=self.batch_size, lr=self.lr, adv=self.use_adversarial_training, dp=self.use_dp,
            sigma=self.dp_sigma, clip=self.dp_clip_norm, delta=self.dp_delta, seed=self.seed,
            pgd_eps=self.pgd_epsilon, pgd_steps=self.pgd_steps)
        if verbose and self.use_dp:
            print(f"DP-SGD: sigma={self.dp_sigma}, q={self.train_info['sample_rate']:.4f}, "
                  f"steps={self.train_info['steps']}, epsilon={self.epsilon:.2f} (delta={self.dp_delta})")
        self._is_fitted = True
        return self

    def predict(self, sessions: List[dict], adversarial: bool = False) -> np.ndarray:
        self._check_fitted()
        X, A, M, y = self._dense(sessions)
        Xs = self.scaler.transform(X)
        if adversarial:
            Xs = adversarial_inputs(self.model, Xs, A, y, node_mask=M, eps=self.pgd_epsilon, steps=20,
                                    seed=self.seed)
        return predict(self.model, Xs, A)

    def predict_proba(self, sessions: List[dict]) -> np.ndarray:
        self._check_fitted()
        X, A, _, _ = self._dense(sessions)
        with torch.no_grad():
            self.model.eval()
            return F.softmax(self.model(T(self.scaler.transform(X)), T(A)), dim=1).numpy()

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() or load() before predicting.")

    def save(self, path: str) -> None:
        torch.save({"state": self.model.state_dict(), "scale": self.scaler.scale_ if self.scaler else None,
                    "train_info": self.train_info}, path)

    def load(self, path: str) -> "FAGIDetector":
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(blob["state"])
        self.scaler = NodeFeatureScaler()
        self.scaler.scale_ = blob["scale"]
        self.train_info = blob.get("train_info", {})
        self._is_fitted = True
        return self
