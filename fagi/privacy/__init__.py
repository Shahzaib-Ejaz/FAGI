"""Differential privacy for FAGI.

`DPSGD` (per-example clipping, Gaussian noise, Poisson sampling) and the RDP accountant
`compute_epsilon` are what every experiment uses. The legacy `mvg_dp.MVGMechanism`
(aggregate clipping, not differentially private) is kept only for old callers and is
deliberately not exported here.
"""
from fagi.privacy.dp_sgd import DPSGD, compute_epsilon, steps_for

__all__ = ["DPSGD", "compute_epsilon", "steps_for"]
