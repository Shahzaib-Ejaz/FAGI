"""
FAGI — Federated Adversarial Graph Intelligence
================================================
A framework for adversarially robust, privacy-preserving detection
of threats in cloud API call sequences.

Paper: "Adversarial Machine Learning for Cloud API Threat Detection:
        A Federated Learning Framework Bridging Software Engineering,
        Data Science, and Cybersecurity"
Author: Shahzaib Ejaz, North American University
"""

from fagi.graph.constructor import APIGraphConstructor
from fagi.model.gnn import FAGIModel
from fagi.privacy.mvg_dp import MVGMechanism
from fagi.federated.server import FederatedServer
from fagi.detector import FAGIDetector

__version__ = "1.0.0"
__all__ = [
    "FAGIDetector",
    "APIGraphConstructor",
    "FAGIModel",
    "MVGMechanism",
    "FederatedServer",
]
