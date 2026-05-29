# FAGI — Federated Adversarial Graph Intelligence

**Adversarial Machine Learning for Cloud API Threat Detection: A Federated Learning Framework Bridging Software Engineering, Data Science, and Cybersecurity**

*Shahzaib Ejaz — Department of Computer Science, North American University*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format-green.svg)](paper/)

---

## Overview

FAGI is a four-module framework for adversarially robust, privacy-preserving detection of threats in cloud API call sequences. It addresses the three primary failure modes of existing approaches:

1. **Signature systems** evaded trivially by zero-day and obfuscated attacks
2. **Standalone ML detectors** achieving strong clean accuracy but collapsing under adversarial perturbation (vanilla GNN degrades 7.6pp under PGD)
3. **Centralised training** legally and operationally infeasible for cross-organisational threat intelligence sharing

FAGI solves all three by unifying:

| Module | Role | Key Method |
|--------|------|-----------|
| **API Graph Constructor** | Translates gateway logs → attributed call graphs | k-NN graph from endpoint features |
| **LLM Embedding Engine** | Semantic payload analysis | DistilBERT-base-uncased (zero-shot) |
| **Adversarial GNN Detector** | Graph-aware detection + robustness | GraphSAGE L=2 + GNNGuard + PGD training |
| **Federated Aggregation Server** | Privacy-preserving multi-org learning | FedAvg + MVG-DP (ε=2.0, δ=10⁻⁵) |

## Key Results

| Metric | FAGI | GNN Baseline | Gap |
|--------|------|-------------|-----|
| Clean Accuracy (CICIDS2017) | **75.6% ±1.7%** | 74.8% ±2.5% | +0.8pp |
| **Adversarial Accuracy (PGD ε=0.1)** | **74.4%** | 67.2% | **+7.2pp** |
| False Positive Rate | **3.1%** | 4.9% | −1.8pp |
| Membership Inference Success | **51.8%** | 78.6% | −26.8pp |
| Graph Structure Attack (+80% edges) | **74.4%** | 71.9% | +2.5pp |

> **Primary finding:** FAGI retains 74.4% detection accuracy under PGD feature-space attack versus 67.2% for the GNN baseline — a **7.2 percentage-point robustness advantage** directly attributable to adversarial training and GNNGuard edge filtering.

## Architecture

```
API Gateway Logs
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              API Graph Constructor                   │
│  Raw logs → G=(V, E, X)                             │
│  V = unique endpoints, E = call dependencies        │
│  X = [call_count, avg_dur, avg_size, privilege,     │
│        write_ratio, payload_susp_score]             │
└──────────────────────┬──────────────────────────────┘
                       │
       ┌───────────────▼───────────────┐
       │      LLM Embedding Engine     │
       │  DistilBERT → e_v ∈ ℝ^128    │
       │  x_v ← concat(h_v, e_v)      │
       └───────────────┬───────────────┘
                       │
       ┌───────────────▼───────────────┐
       │   Adversarial GNN Detector    │
       │  GraphSAGE L=2 (64/32 dims)  │
       │  + GNNGuard (τ=0.15)         │
       │  + PGD Training (ε=0.1, K=10)│
       └───────────────┬───────────────┘
                       │ (per-org gradients)
       ┌───────────────▼───────────────┐
       │  Federated Aggregation Server │
       │  FedAvg + MVG-DP              │
       │  Clip C=1.0, σ=1.1            │
       │  (ε=2.0, δ=10⁻⁵)            │
       └───────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/shahzaibejaz/FAGI.git
cd FAGI
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, PyTorch Geometric, transformers, scikit-learn

## Quick Start

```python
from fagi import FAGIDetector

# Initialise detector
detector = FAGIDetector(
    hidden_dims=(64, 32),
    pgd_epsilon=0.1,
    pgd_steps=10,
    gnnguard_threshold=0.15,
    dp_epsilon=2.0,
    dp_delta=1e-5,
    dp_sigma=1.1,
    dp_clip_norm=1.0,
)

# Train on API call graph sessions
detector.fit(train_sessions, epochs=50)

# Detect threats
predictions = detector.predict(test_sessions)
```

## Repository Structure

```
FAGI/
├── fagi/                    # Core library
│   ├── __init__.py
│   ├── graph/               # API call graph construction
│   │   ├── constructor.py   # APIGraphConstructor
│   │   └── features.py      # Node feature extraction
│   ├── model/               # GNN model + adversarial training
│   │   ├── gnn.py           # GraphSAGE with GNNGuard
│   │   ├── gnnguard.py      # Edge filtering
│   │   └── pgd.py           # PGD adversarial training
│   ├── privacy/             # Differential privacy
│   │   ├── mvg_dp.py        # MVG mechanism
│   │   └── accounting.py    # Rényi-DP accounting
│   ├── federated/           # Federated learning
│   │   ├── server.py        # FedAvg aggregation
│   │   └── client.py        # Per-org local training
│   └── utils/               # Helpers
│       ├── embeddings.py    # DistilBERT wrapper
│       └── metrics.py       # Evaluation metrics
├── experiments/             # Reproducible experiment scripts
│   ├── exp1_lateral_movement.py
│   ├── exp2_llm_contribution.py
│   ├── exp3_adversarial_robustness.py
│   ├── exp4_federated.py
│   ├── ablation.py
│   └── results/             # Stored results (JSON)
├── data/                    # Dataset utilities
│   ├── cicids2017.py        # CICIDS2017 loader
│   ├── synthetic/           # Synthetic API session generator
│   │   ├── generator.py
│   │   └── sessions_final.json
│   └── README.md
├── scripts/                 # Training + inference entry points
│   ├── train.py
│   ├── evaluate.py
│   └── generate_figures.py
├── tests/                   # Unit tests
│   ├── test_graph.py
│   ├── test_gnn.py
│   ├── test_privacy.py
│   └── test_federated.py
├── figures/                 # Reproduced paper figures
├── docs/                    # Extended documentation
│   └── architecture.md
├── requirements.txt
├── setup.py
└── README.md
```

## Reproducing Paper Results

```bash
# Experiment 1: Lateral movement detection
python experiments/exp1_lateral_movement.py

# Experiment 2: LLM payload contribution
python experiments/exp2_llm_contribution.py

# Experiment 3: Adversarial robustness (primary result)
python experiments/exp3_adversarial_robustness.py

# Experiment 4: Federated generalisation
python experiments/exp4_federated.py

# Full ablation study
python experiments/ablation.py

# Regenerate all paper figures
python scripts/generate_figures.py
```

Expected outputs are stored in `experiments/results/` for comparison.

## Datasets

**CICIDS2017** — Download from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html). Place the `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` file in `data/cicids2017/`.

**Synthetic API Session Dataset** — Generated by `data/synthetic/generator.py`. 4,000 sessions across 5 organisational nodes covering lateral movement, slow exfiltration, schema reconnaissance, and SQL injection attack types.

## Citation

```bibtex
@article{ejaz2026fagi,
  title={Adversarial Machine Learning for Cloud API Threat Detection:
         A Federated Learning Framework Bridging Software Engineering,
         Data Science, and Cybersecurity},
  author={Ejaz, Shahzaib},
  journal={IEEE Workshop on Artificial Intelligence and Security (AISec)},
  year={2026},
  institution={North American University}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgements

This work builds on [GNNGuard](https://github.com/mims-harvard/GNNGuard) (Zhang & Zitnik, NeurIPS 2020) and the MVG differential privacy mechanism (Chanyaswad, Dytso, Poor & Mittal, CCS 2018). CICIDS2017 dataset provided by the Canadian Institute for Cybersecurity, University of New Brunswick.
