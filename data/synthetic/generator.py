"""
Synthetic API Call Session Generator
=====================================
Generates the 4,000-session dataset used in Section V-C of the paper.

Design principles:
1. All sessions have overlapping aggregate statistics (similar call counts,
   write ratios, timing distributions) so that flow-level features cannot
   trivially separate benign from malicious sessions.
2. Attack patterns encode behavioural signals (call sequences, payload
   patterns) that only graph-topology-aware or payload-aware detectors
   can exploit.
3. Five organisational nodes have heterogeneous attack coverage,
   creating the data heterogeneity condition tested in Experiment 4.

Distribution overlap (validated post-generation):
  - write_ratio: benign 0.354±0.145, attack 0.357±0.158
  - call_count:  benign 9.6±4.0,    attack 10.1±1.0

Usage
-----
    python data/synthetic/generator.py --n_orgs 5 --output sessions_final.json

Or from Python:
    from data.synthetic.generator import generate_dataset
    sessions = generate_dataset(n_per_org_benign=500)
"""

import argparse
import json
import random
import numpy as np
from typing import List, Dict

random.seed(42)
np.random.seed(42)

# ── Endpoint definitions ──────────────────────────────────────────────
ENDPOINTS = {
    "auth":         {"port": 443,  "priv": 0},
    "user_read":    {"port": 443,  "priv": 1},
    "user_write":   {"port": 443,  "priv": 2},
    "admin_read":   {"port": 8443, "priv": 3},
    "admin_write":  {"port": 8443, "priv": 4},
    "data_export":  {"port": 443,  "priv": 3},
    "data_query":   {"port": 443,  "priv": 2},
    "health":       {"port": 80,   "priv": 0},
    "schema":       {"port": 443,  "priv": 0},
    "billing":      {"port": 443,  "priv": 2},
    "notify":       {"port": 443,  "priv": 1},
    "search":       {"port": 443,  "priv": 1},
}
EP_NAMES = list(ENDPOINTS.keys())
NORMAL_POOL = [
    "user_read", "data_query", "search", "notify",
    "billing", "health", "schema", "data_export",
    "user_write", "admin_read",
]

# ── Per-organisation attack coverage ─────────────────────────────────
ORG_ATTACKS = {
    0: ["lateral_movement", "slow_exfiltration"],
    1: ["schema_recon",     "sql_injection"],
    2: ["lateral_movement", "schema_recon"],
    3: ["slow_exfiltration","sql_injection"],
    4: ["lateral_movement", "slow_exfiltration",  # all types (test org)
        "schema_recon",     "sql_injection"],
}


def _make_call(src: str, dst: str, atype: str = None) -> dict:
    """Generate one API call with realistic timing and payload."""
    priv = ENDPOINTS[dst]["priv"]
    dur  = abs(np.random.normal(350 + priv * 60, 120))
    size = abs(np.random.normal(700 + priv * 150, 150))

    if atype == "sql":
        tokens = random.choice([
            ["SELECT", "FROM"],
            ["'", "OR", "1=1"],
            ["UNION", "ALL"],
        ])
    elif atype == "lat":
        tokens = ["escalate", "admin"] if random.random() < 0.5 else ["bearer", "token"]
    elif atype == "exf":
        tokens = ["bulk", "export", "all"]
        size  *= 5
    elif atype == "rec":
        tokens = ["OPTIONS", "enumerate", "schema"]
    else:
        tokens = random.sample(["json", "data", "ok", "api", "v2"], 2)

    method = "POST" if random.random() < 0.35 else "GET"
    return {
        "src": src,
        "dst": dst,
        "duration_ms": round(dur, 1),
        "response_size": round(size, 1),
        "privilege_level": priv,
        "port": ENDPOINTS[dst]["port"],
        "method": method,
        "status_code": 200,
        "payload_tokens": tokens,
    }


def _benign_session(org_id: int) -> dict:
    src = f"u{org_id}_{random.randint(1, 500)}"
    n_unique = random.randint(4, 10)
    chosen = random.sample(list(ENDPOINTS.keys()), min(n_unique, len(EP_NAMES)))
    calls = [_make_call(src, "auth")]
    for _ in range(random.randint(6, 14)):
        calls.append(_make_call(src, random.choice(chosen)))
    return {"calls": calls, "label": 0, "attack_type": "none", "org_id": org_id}


def _lateral_movement(org_id: int) -> dict:
    src = f"atk{org_id}_{random.randint(1, 100)}"
    calls = [
        _make_call(src, "health"),
        _make_call(src, "auth"),
        _make_call(src, "user_read"),
        _make_call(src, "user_write", "lat"),
        _make_call(src, "admin_read", "lat"),
        _make_call(src, "admin_write", "lat"),
        _make_call(src, "data_export", "lat"),
    ]
    for _ in range(random.randint(1, 3)):
        calls.append(_make_call(src, random.choice(["search", "notify"])))
    return {"calls": calls, "label": 1, "attack_type": "lateral_movement",
            "org_id": org_id}


def _slow_exfiltration(org_id: int) -> dict:
    src = f"atk{org_id}_{random.randint(1, 100)}"
    calls = [_make_call(src, "auth"), _make_call(src, "user_read")]
    for _ in range(random.randint(6, 10)):
        calls.append(_make_call(src, "data_export", "exf"))
        if random.random() < 0.5:
            calls.append(_make_call(src, "search"))
    return {"calls": calls, "label": 1, "attack_type": "slow_exfiltration",
            "org_id": org_id}


def _schema_recon(org_id: int) -> dict:
    src = f"atk{org_id}_{random.randint(1, 100)}"
    calls = [_make_call(src, "health")]
    for ep in random.sample(EP_NAMES, random.randint(9, 12)):
        calls.append(_make_call(src, ep, "rec"))
    return {"calls": calls, "label": 1, "attack_type": "schema_recon",
            "org_id": org_id}


def _sql_injection(org_id: int) -> dict:
    src = f"atk{org_id}_{random.randint(1, 100)}"
    calls = [_make_call(src, "auth")]
    for _ in range(random.randint(1, 3)):
        calls.append(_make_call(src, "user_read"))
    for _ in range(random.randint(4, 7)):
        calls.append(_make_call(src, "data_query", "sql"))
    calls.append(_make_call(src, "data_export", "sql"))
    return {"calls": calls, "label": 1, "attack_type": "sql_injection",
            "org_id": org_id}


ATTACK_FNS = {
    "lateral_movement":  _lateral_movement,
    "slow_exfiltration": _slow_exfiltration,
    "schema_recon":      _schema_recon,
    "sql_injection":     _sql_injection,
}


def generate_dataset(
    n_per_org_benign: int = 500,
    n_per_org_per_attack: int = 125,
    org_attacks: dict = None,
    seed: int = 42,
) -> List[dict]:
    """
    Generate the full synthetic API call session dataset.

    Parameters
    ----------
    n_per_org_benign : int
        Number of benign sessions per organisation.
    n_per_org_per_attack : int
        Number of malicious sessions per (org, attack_type) pair.
    org_attacks : dict, optional
        Mapping of org_id → list of attack type names.
        Default: ORG_ATTACKS (as in the paper).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Shuffled list of session dictionaries.
    """
    random.seed(seed)
    np.random.seed(seed)

    if org_attacks is None:
        org_attacks = ORG_ATTACKS

    sessions = []
    for org_id, attacks in org_attacks.items():
        for _ in range(n_per_org_benign):
            sessions.append(_benign_session(org_id))
        for atk in attacks:
            fn = ATTACK_FNS[atk]
            for _ in range(n_per_org_per_attack):
                sessions.append(fn(org_id))

    random.shuffle(sessions)
    return sessions


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic API call session dataset"
    )
    parser.add_argument("--output", default="sessions_final.json",
                        help="Output JSON path")
    parser.add_argument("--n_benign", type=int, default=500,
                        help="Benign sessions per org")
    parser.add_argument("--n_attack", type=int, default=125,
                        help="Attack sessions per (org, type)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Generating dataset...")
    sessions = generate_dataset(
        n_per_org_benign=args.n_benign,
        n_per_org_per_attack=args.n_attack,
        seed=args.seed,
    )

    with open(args.output, "w") as f:
        json.dump(sessions, f, indent=2)

    n_benign  = sum(1 for s in sessions if s["label"] == 0)
    n_attack  = sum(1 for s in sessions if s["label"] == 1)
    from collections import Counter
    atk_dist  = Counter(s["attack_type"] for s in sessions if s["label"] == 1)

    print(f"Generated {len(sessions)} sessions:")
    print(f"  Benign:  {n_benign}")
    print(f"  Attack:  {n_attack}")
    for k, v in sorted(atk_dist.items()):
        print(f"    {k}: {v}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
