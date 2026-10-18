import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from common import (ROOT, DenseFAGI, adversarial_inputs, predict, set_seed, train_model)
from fagi.graph.dense import NodeFeatureScaler, sessions_to_dense

SYN_CONFIG = {
    "n_folds": 5,
    "cv_seed": 42,
    "epochs": 30,
    "batch_size": 32,
    "lr": 2e-3,
    "gnnguard_tau": 0.15,
    "pgd_eps": 0.1,
    "pgd_train_steps": 10,
    "pgd_eval_steps": 20,
    "dp_sigma": 1.1,
    "dp_clip": 1.0,
    "dp_delta": 1e-5,
    "init": "reset_after_seed",
}

SYN_VARIANTS = {
    "gnn_nopayload": {"payload": False, "guard": False, "adv": False, "dp": False, "seed": 41},
    "gnn": {"payload": True, "guard": False, "adv": False, "dp": False, "seed": 42},
    "fagi": {"payload": True, "guard": True, "adv": True, "dp": True, "seed": 43},
    "fagi_no_payload": {"payload": False, "guard": True, "adv": True, "dp": True, "seed": 44},
    "fagi_no_guard": {"payload": True, "guard": False, "adv": True, "dp": True, "seed": 45},
    "fagi_no_pgd": {"payload": True, "guard": True, "adv": False, "dp": True, "seed": 46},
    "fagi_no_dp": {"payload": True, "guard": True, "adv": True, "dp": False, "seed": 47},
}


def load_sessions(path=None):
    path = path or os.path.join(ROOT, "data", "synthetic", "sessions_final.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing. Run: python data/synthetic/generator.py")
    with open(path) as f:
        return json.load(f)


class SyntheticData:
    def __init__(self, sessions):
        self.sessions = sessions
        self.attack_type = np.array([s["attack_type"] for s in sessions])
        self.org = np.array([s["org_id"] for s in sessions])
        X6, A, M, y = sessions_to_dense(sessions, include_payload=True)
        self.X = {True: X6, False: X6[:, :, :5].copy()}
        self.A, self.M, self.y = A, M, y

    def subset_folds(self, idx, seed=None):
        skf = StratifiedKFold(SYN_CONFIG["n_folds"], shuffle=True,
                              random_state=SYN_CONFIG["cv_seed"] if seed is None else seed)
        strat = self.attack_type[idx]
        return [(idx[a], idx[b]) for a, b in skf.split(np.zeros(len(idx)), strat)]

    def scaled(self, payload, tr, te):
        sc = NodeFeatureScaler().fit(self.X[payload][tr], self.M[tr])
        return sc.transform(self.X[payload][tr]), sc.transform(self.X[payload][te])

    def train_variant(self, name, tr, te, seed_offset=0, epochs=None):
        v = SYN_VARIANTS[name]
        Xtr, Xte = self.scaled(v["payload"], tr, te)
        model = DenseFAGI(Xtr.shape[2], gnnguard_threshold=SYN_CONFIG["gnnguard_tau"] if v["guard"] else None,
                          readout="mean")
        info = train_model(model, Xtr, self.A[tr], self.y[tr], node_mask=self.M[tr],
                           epochs=epochs or SYN_CONFIG["epochs"], batch_size=SYN_CONFIG["batch_size"],
                           lr=SYN_CONFIG["lr"], adv=v["adv"], dp=v["dp"], sigma=SYN_CONFIG["dp_sigma"],
                           clip=SYN_CONFIG["dp_clip"], delta=SYN_CONFIG["dp_delta"],
                           seed=v["seed"] + seed_offset, pgd_eps=SYN_CONFIG["pgd_eps"],
                           pgd_steps=SYN_CONFIG["pgd_train_steps"])
        return model, info, Xtr, Xte

    def evaluate(self, model, Xte, te, adv=True, seed=0):
        pc = predict(model, Xte, self.A[te])
        out = {"pred_clean": pc, "clean": float(np.mean(pc == self.y[te]))}
        if adv:
            xa = adversarial_inputs(model, Xte, self.A[te], self.y[te], node_mask=self.M[te],
                                    eps=SYN_CONFIG["pgd_eps"], steps=SYN_CONFIG["pgd_eval_steps"], seed=seed)
            pa = predict(model, xa, self.A[te])
            out.update({"pred_adv": pa, "adv": float(np.mean(pa == self.y[te]))})
        return out

    def rf(self, tr, te, seed=0):
        F = np.concatenate([self.X[True].reshape(len(self.y), -1), self.A.reshape(len(self.y), -1)], axis=1)
        set_seed(seed)
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed).fit(F[tr], self.y[tr])
        return clf.predict(F[te])


def inject_edges(A, ratio, seed):
    rng = np.random.RandomState(seed)
    A2 = A.copy()
    n = A.shape[1]
    for i in range(len(A2)):
        n_add = max(1, int(round(A[i].sum() * ratio)))
        src = rng.randint(0, n, n_add)
        dst = rng.randint(0, n, n_add)
        np.add.at(A2[i], (dst, src), 1.0)
    return A2
