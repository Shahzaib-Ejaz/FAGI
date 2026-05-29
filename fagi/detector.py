"""
FAGIDetector — High-Level Interface
=====================================
Combines all four FAGI modules into a single sklearn-compatible
estimator for easy use.

Example
-------
>>> from fagi import FAGIDetector
>>> detector = FAGIDetector()
>>> detector.fit(train_sessions, epochs=50)
>>> preds = detector.predict(test_sessions)
>>> proba = detector.predict_proba(test_sessions)
"""

import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

from fagi.graph.constructor import APIGraphConstructor
from fagi.model.gnn import FAGIModel
from fagi.model.pgd import adversarial_loss
from fagi.privacy.mvg_dp import MVGMechanism


class FAGIDetector:
    """
    End-to-end FAGI threat detector.

    Parameters
    ----------
    hidden_dims : tuple[int, int]
        GNN hidden layer dimensions. Paper: (64, 32).
    pgd_epsilon : float
        PGD perturbation budget ε. Paper: 0.1.
    pgd_steps : int
        PGD attack steps K. Paper: 10.
    gnnguard_threshold : float or None
        GNNGuard cosine similarity threshold τ. Paper: 0.15.
        Set None to disable GNNGuard (FAGI−GNNGuard ablation).
    use_adversarial_training : bool
        Whether to use PGD adversarial training.
        Set False for FAGI−PGD ablation.
    dp_epsilon : float
        Differential privacy budget ε. Paper: 2.0 (moderate guarantee).
    dp_delta : float
        DP failure probability δ. Paper: 1e-5.
    dp_sigma : float
        MVG noise multiplier σ. Paper: 1.1.
    dp_clip_norm : float
        Gradient clipping norm C. Paper: 1.0.
    include_payload : bool
        Whether to use payload suspicion scores as node features.
        Set False for FAGI−LLM ablation.
    lr : float
        Learning rate. Default: 2e-3.
    device : str
        Compute device. Default: 'cpu'.
    """

    def __init__(
        self,
        hidden_dims: tuple = (64, 32),
        pgd_epsilon: float = 0.1,
        pgd_steps: int = 10,
        gnnguard_threshold: Optional[float] = 0.15,
        use_adversarial_training: bool = True,
        dp_epsilon: float = 2.0,
        dp_delta: float = 1e-5,
        dp_sigma: float = 1.1,
        dp_clip_norm: float = 1.0,
        include_payload: bool = True,
        lr: float = 2e-3,
        device: str = "cpu",
    ):
        self.hidden_dims = hidden_dims
        self.pgd_epsilon = pgd_epsilon
        self.pgd_steps = pgd_steps
        self.gnnguard_threshold = gnnguard_threshold
        self.use_adversarial_training = use_adversarial_training
        self.dp_sigma = dp_sigma
        self.dp_clip_norm = dp_clip_norm
        self.include_payload = include_payload
        self.lr = lr
        self.device = device

        # Initialise modules
        self.graph_constructor = APIGraphConstructor(
            include_payload=include_payload
        )
        feature_dim = 6 if include_payload else 5
        self.model = FAGIModel(
            in_dim=feature_dim,
            hidden_dims=hidden_dims,
            gnnguard_threshold=gnnguard_threshold,
        ).to(device)
        self.dp_mechanism = MVGMechanism(
            clip_norm=dp_clip_norm,
            sigma=dp_sigma,
            device=device,
        )
        self._is_fitted = False

    def fit(
        self,
        sessions: List[dict],
        epochs: int = 50,
        batch_size: int = 250,
        verbose: bool = True,
    ) -> "FAGIDetector":
        """
        Train the FAGI detector on a list of labelled API sessions.

        Parameters
        ----------
        sessions : list[dict]
            Each dict has 'calls' (list of call dicts) and 'label' (0/1).
        epochs : int
            Training epochs.
        batch_size : int
            Max graphs per gradient step.
        verbose : bool
            Print epoch losses.

        Returns
        -------
        self
        """
        graphs = self.graph_constructor.sessions_to_graphs(sessions)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

        self.model.train()
        for epoch in range(epochs):
            idx = np.random.permutation(len(graphs))[:min(batch_size, len(graphs))]
            epoch_loss = 0.0

            for i in idx:
                g = graphs[i].to(self.device)
                optimizer.zero_grad()

                if self.use_adversarial_training and epoch % 2 == 0:
                    loss = adversarial_loss(
                        self.model, g.x, g.edge_index, g.y,
                        epsilon=self.pgd_epsilon,
                        pgd_steps=self.pgd_steps,
                    )
                else:
                    out = self.model(g.x, g.edge_index)
                    loss = F.cross_entropy(out, g.y)

                loss.backward()
                self.dp_mechanism.clip_and_noise(self.model)
                optimizer.step()
                epoch_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                avg = epoch_loss / max(len(idx), 1)
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg:.4f}")

        self._is_fitted = True
        return self

    def predict(
        self,
        sessions: List[dict],
        adversarial: bool = False,
    ) -> np.ndarray:
        """
        Predict threat labels for API sessions.

        Parameters
        ----------
        sessions : list[dict]
            Sessions to classify.
        adversarial : bool
            If True, evaluate under PGD attack (adversarial accuracy).

        Returns
        -------
        np.ndarray of int
            Predicted labels: 0 = benign, 1 = malicious.
        """
        graphs = self.graph_constructor.sessions_to_graphs(sessions)
        self.model.eval()
        predictions = []

        for g in graphs:
            g = g.to(self.device)
            if adversarial:
                from fagi.model.pgd import pgd_attack
                x_adv = pgd_attack(
                    self.model, g.x, g.edge_index, g.y,
                    epsilon=self.pgd_epsilon,
                    steps=self.pgd_steps,
                )
                with torch.no_grad():
                    pred = self.model(x_adv, g.edge_index).argmax(1).item()
            else:
                with torch.no_grad():
                    pred = self.model(g.x, g.edge_index).argmax(1).item()
            predictions.append(pred)

        return np.array(predictions)

    def predict_proba(self, sessions: List[dict]) -> np.ndarray:
        """
        Return class probability estimates.

        Returns
        -------
        np.ndarray, shape (n_sessions, 2)
            Columns: [P(benign), P(malicious)]
        """
        graphs = self.graph_constructor.sessions_to_graphs(sessions)
        self.model.eval()
        probs = []

        with torch.no_grad():
            for g in graphs:
                g = g.to(self.device)
                logits = self.model(g.x, g.edge_index)
                p = F.softmax(logits, dim=1).cpu().numpy()[0]
                probs.append(p)

        return np.array(probs)

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> "FAGIDetector":
        """Load model weights from disk."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self._is_fitted = True
        return self
