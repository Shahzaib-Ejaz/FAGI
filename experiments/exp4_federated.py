import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.optimize import brentq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from cicids_stats import ms
from common import DenseFAGI, env_info, metrics, per_sample_loss, predict, save_json, set_seed, train_model
from fagi.federated.server import FederatedServer
from fagi.privacy.dp_sgd import compute_epsilon, steps_for
from mia import mia_report
from synthetic_common import SYN_CONFIG, SyntheticData, load_sessions, scale_features

TRAIN_ORGS = [0, 1, 2, 3]
TEST_ORG = 4
ROUNDS = 30
LOCAL_EPOCHS = 1
N_REPEATS = 3
MEMBER_FRACTION = 0.8
OVERFIT_N = 200
OVERFIT_EPOCHS = 300


def make_model(d):
    return DenseFAGI(d, gnnguard_threshold=SYN_CONFIG["gnnguard_tau"], readout="mean")


def kwargs(dp, seed, epochs, sigma=None):
    return dict(epochs=epochs, batch_size=SYN_CONFIG["batch_size"], lr=SYN_CONFIG["lr"], adv=True, dp=dp,
                sigma=sigma if sigma is not None else SYN_CONFIG["dp_sigma"], clip=SYN_CONFIG["dp_clip"],
                delta=SYN_CONFIG["dp_delta"], seed=seed, pgd_eps=SYN_CONFIG["pgd_eps"],
                pgd_steps=SYN_CONFIG["pgd_train_steps"])


def matched_sigma(target_eps, n, epochs):
    q = min(1.0, SYN_CONFIG["batch_size"] / n)
    steps = steps_for(epochs, q)
    return brentq(lambda s: compute_epsilon(s, q, steps, SYN_CONFIG["dp_delta"]) - target_eps, 0.3, 50.0)


def per_type(D, test, pred):
    return {t: float(np.mean(pred[D.attack_type[test] == t] == (0 if t == "none" else 1)))
            for t in np.unique(D.attack_type[test])}


def run_repeat(D, r):
    seed = 500 + r
    members, nonmembers = {}, {}
    for o in TRAIN_ORGS:
        idx = np.flatnonzero(D.org == o)
        a, b = train_test_split(idx, train_size=MEMBER_FRACTION, random_state=seed, stratify=D.attack_type[idx])
        members[o], nonmembers[o] = np.sort(a), np.sort(b)
    X = scale_features(D.X[True])
    test = np.flatnonzero(D.org == TEST_ORG)
    mem_all = np.concatenate([members[o] for o in TRAIN_ORGS])
    non_all = np.concatenate([nonmembers[o] for o in TRAIN_ORGS])
    d = X.shape[2]
    out = {}

    def finish(name, model, eps, extra=None):
        pred = predict(model, X[test], D.A[test])
        lm = per_sample_loss(model, X[mem_all], D.A[mem_all], D.y[mem_all])
        ln = per_sample_loss(model, X[non_all], D.A[non_all], D.y[non_all])
        out[name] = {"test": metrics(D.y[test], pred), "epsilon": eps, "per_type_detection": per_type(D, test, pred),
                     "mia": mia_report(lm, ln, D.attack_type[mem_all], D.attack_type[non_all], seed=seed)}
        if extra:
            out[name].update(extra)

    F_all = np.concatenate([D.X[True].reshape(len(D.y), -1), D.A.reshape(len(D.y), -1)], axis=1)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed).fit(F_all[mem_all], D.y[mem_all])
    pr = rf.predict(F_all[test])
    out["rf_central"] = {"test": metrics(D.y[test], pr), "epsilon": None, "per_type_detection": per_type(D, test, pr)}

    q_client = SYN_CONFIG["batch_size"] / len(members[0])
    eps_client = compute_epsilon(SYN_CONFIG["dp_sigma"], q_client, ROUNDS * steps_for(LOCAL_EPOCHS, q_client),
                                 SYN_CONFIG["dp_delta"])
    sig_match = matched_sigma(eps_client, len(mem_all), ROUNDS * LOCAL_EPOCHS)

    for name, dp, sigma in [("central_nodp", False, None), ("central_dp", True, None),
                            ("central_dp_eps_matched", True, sig_match)]:
        m = make_model(d)
        info = train_model(m, X[mem_all], D.A[mem_all], D.y[mem_all], node_mask=D.M[mem_all],
                           **kwargs(dp, seed + len(name), ROUNDS * LOCAL_EPOCHS, sigma))
        finish(name, m, info.get("epsilon"), {"sigma": sigma if sigma else (SYN_CONFIG["dp_sigma"] if dp else None)})

    for dp in (False, True):
        set_seed(seed + 10)
        server = FederatedServer(make_model(d), dp_mechanism=None, byzantine_threshold=float("inf"))
        steps = {o: 0 for o in TRAIN_ORGS}
        rates = {}
        last_locals = {}
        for rnd in range(ROUNDS):
            locals_, weights = [], []
            for o in TRAIN_ORGS:
                lm = copy.deepcopy(server.global_model)
                info = train_model(lm, X[members[o]], D.A[members[o]], D.y[members[o]], node_mask=D.M[members[o]],
                                   reinit=False, **kwargs(dp, seed * 1000 + rnd * 10 + o, LOCAL_EPOCHS))
                if dp:
                    steps[o] += info["steps"]
                    rates[o] = info["sample_rate"]
                locals_.append(lm)
                last_locals[o] = lm
                weights.append(float(len(members[o])))
            server.aggregate(locals_, weights)
        eps = (max(compute_epsilon(SYN_CONFIG["dp_sigma"], rates[o], steps[o], SYN_CONFIG["dp_delta"])
                   for o in TRAIN_ORGS) if dp else None)
        client_mia = []
        for o in TRAIN_ORGS:
            lm_ = per_sample_loss(last_locals[o], X[members[o]], D.A[members[o]], D.y[members[o]])
            ln_ = per_sample_loss(last_locals[o], X[nonmembers[o]], D.A[nonmembers[o]], D.y[nonmembers[o]])
            client_mia.append(mia_report(lm_, ln_, seed=seed + o))
        finish("fedavg_dp" if dp else "fedavg_nodp", server.global_model, eps,
               {"client_model_mia_auc_mean": float(np.mean([c["auc"] for c in client_mia])),
                "client_model_mia_tpr1_mean": float(np.mean([c["tpr_at_1pct_fpr"] for c in client_mia]))})

    for dp in (False, True):
        accs = []
        for o in TRAIN_ORGS:
            m = make_model(d)
            train_model(m, X[members[o]], D.A[members[o]], D.y[members[o]], node_mask=D.M[members[o]],
                        **kwargs(dp, seed + 20 + o + (50 if dp else 0), ROUNDS * LOCAL_EPOCHS))
            accs.append(metrics(D.y[test], predict(m, X[test], D.A[test]))["accuracy"])
        out["local_dp_mean" if dp else "local_only_mean"] = {"test": {"accuracy": float(np.mean(accs))},
                                                             "per_org": accs, "epsilon": eps_client if dp else None}

    rng = np.random.RandomState(seed)
    small = rng.choice(mem_all, OVERFIT_N, replace=False)
    m = make_model(d)
    train_model(m, X[small], D.A[small], D.y[small], node_mask=D.M[small],
                **dict(kwargs(False, seed + 99, OVERFIT_EPOCHS), adv=False))
    lm = per_sample_loss(m, X[small], D.A[small], D.y[small])
    ln = per_sample_loss(m, X[non_all], D.A[non_all], D.y[non_all])
    out["overfit_control"] = {"test": metrics(D.y[test], predict(m, X[test], D.A[test])), "epsilon": None,
                              "mia": mia_report(lm, ln, seed=seed)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_data", default=None)
    ap.add_argument("--output", default="experiments/results/exp4.json")
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    args = ap.parse_args()
    print("EXPERIMENT 4: federated generalization to an unseen organization + membership inference")
    D = SyntheticData(load_sessions(args.api_data))
    runs = []
    for r in range(args.repeats):
        res = run_repeat(D, r)
        runs.append(res)
        print(f"  repeat {r + 1}: " + "  ".join(f"{k}={v['test']['accuracy']:.3f}" for k, v in res.items()), flush=True)
    summary = {}
    for k in runs[0]:
        s = {"accuracy": ms([x[k]["test"]["accuracy"] for x in runs])}
        if "fpr" in runs[0][k]["test"]:
            s["fpr"] = ms([x[k]["test"]["fpr"] for x in runs])
        if runs[0][k].get("epsilon") is not None:
            s["epsilon"] = ms([x[k]["epsilon"] for x in runs])
        if runs[0][k].get("sigma") is not None:
            s["sigma"] = ms([x[k]["sigma"] for x in runs])
        if "mia" in runs[0][k]:
            for f in ["auc", "attack_accuracy", "tpr_at_1pct_fpr", "tpr_at_0.1pct_fpr"]:
                s[f"mia_{f}"] = ms([x[k]["mia"][f] for x in runs])
        for f in ["client_model_mia_auc_mean", "client_model_mia_tpr1_mean"]:
            if f in runs[0][k]:
                s[f] = ms([x[k][f] for x in runs])
        if "per_type_detection" in runs[0][k]:
            s["per_type_detection"] = {t: float(np.mean([x[k]["per_type_detection"][t] for x in runs]))
                                       for t in runs[0][k]["per_type_detection"]}
        summary[k] = s
    save_json({"config": SYN_CONFIG, "env": env_info(), "train_orgs": TRAIN_ORGS, "test_org": TEST_ORG,
               "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "repeats": args.repeats,
               "member_fraction": MEMBER_FRACTION, "aggregation": "FedAvg weighted by client size (no filter)",
               "normalization": "fixed per-feature scales from a public calibration sample (generator seed 7)", "summary": summary, "runs": runs}, args.output)


if __name__ == "__main__":
    main()
