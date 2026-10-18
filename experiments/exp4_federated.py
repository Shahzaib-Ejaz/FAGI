import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from common import (DenseFAGI, env_info, mean_std, membership_inference, metrics, per_sample_loss,
                    predict, set_seed, train_model, save_json)
from fagi.federated.server import FederatedServer
from fagi.graph.dense import NodeFeatureScaler
from fagi.privacy.dp_sgd import compute_epsilon
from synthetic_common import SYN_CONFIG, SyntheticData, load_sessions

TRAIN_ORGS = [0, 1, 2, 3]
TEST_ORG = 4
ROUNDS = 30
LOCAL_EPOCHS = 1
N_REPEATS = 5
MEMBER_FRACTION = 0.8


def make_model(d):
    return DenseFAGI(d, gnnguard_threshold=SYN_CONFIG["gnnguard_tau"], readout="mean")


def org_split(D, seed):
    members, nonmembers = {}, {}
    for o in TRAIN_ORGS:
        idx = np.flatnonzero(D.org == o)
        a, b = train_test_split(idx, train_size=MEMBER_FRACTION, random_state=seed,
                                stratify=D.attack_type[idx])
        members[o], nonmembers[o] = np.sort(a), np.sort(b)
    return members, nonmembers


def org_scaled(D, idx):
    sc = NodeFeatureScaler().fit(D.X[True][idx], D.M[idx])
    return sc.transform(D.X[True][idx])


def train_kwargs(dp, seed, epochs):
    return dict(epochs=epochs, batch_size=SYN_CONFIG["batch_size"], lr=SYN_CONFIG["lr"], adv=True, dp=dp,
                sigma=SYN_CONFIG["dp_sigma"], clip=SYN_CONFIG["dp_clip"], delta=SYN_CONFIG["dp_delta"],
                seed=seed, pgd_eps=SYN_CONFIG["pgd_eps"], pgd_steps=SYN_CONFIG["pgd_train_steps"])


def run_repeat(D, r):
    seed = 500 + r
    members, nonmembers = org_split(D, seed)
    Xm = {o: org_scaled(D, members[o]) for o in TRAIN_ORGS}
    Xn = {}
    for o in TRAIN_ORGS:
        sc = NodeFeatureScaler().fit(D.X[True][members[o]], D.M[members[o]])
        Xn[o] = sc.transform(D.X[True][nonmembers[o]])
    test = np.flatnonzero(D.org == TEST_ORG)
    Xtest = org_scaled(D, test)
    ytest = D.y[test]
    mem_all = np.concatenate([members[o] for o in TRAIN_ORGS])
    non_all = np.concatenate([nonmembers[o] for o in TRAIN_ORGS])
    Xmem = np.concatenate([Xm[o] for o in TRAIN_ORGS])
    Xnon = np.concatenate([Xn[o] for o in TRAIN_ORGS])
    d = Xmem.shape[2]
    out = {}

    def finish(name, model, eps):
        pred = predict(model, Xtest, D.A[test])
        lm = per_sample_loss(model, Xmem, D.A[mem_all], D.y[mem_all])
        ln = per_sample_loss(model, Xnon, D.A[non_all], D.y[non_all])
        out[name] = {"test": metrics(ytest, pred), "epsilon": eps,
                     "mia": membership_inference(lm, ln, seed=seed),
                     "per_type_detection": {t: float(np.mean(pred[D.attack_type[test] == t] == (0 if t == "none" else 1)))
                                            for t in np.unique(D.attack_type[test])}}

    F_all = np.concatenate([D.X[True].reshape(len(D.y), -1), D.A.reshape(len(D.y), -1)], axis=1)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed).fit(F_all[mem_all], D.y[mem_all])
    out["rf_central"] = {"test": metrics(ytest, rf.predict(F_all[test])), "epsilon": None,
                         "per_type_detection": {t: float(np.mean(rf.predict(F_all[test][D.attack_type[test] == t]) == (0 if t == "none" else 1)))
                                                for t in np.unique(D.attack_type[test])}}

    for dp in (False, True):
        set_seed(seed)
        m = make_model(d)
        info = train_model(m, Xmem, D.A[mem_all], D.y[mem_all], node_mask=D.M[mem_all],
                           **train_kwargs(dp, seed + (1 if dp else 0), ROUNDS * LOCAL_EPOCHS))
        finish("central_dp" if dp else "central_nodp", m, info.get("epsilon"))

    for dp in (False, True):
        set_seed(seed + 10)
        glob = make_model(d)
        server = FederatedServer(glob, dp_mechanism=None)
        steps = {o: 0 for o in TRAIN_ORGS}
        rates = {}
        for rnd in range(ROUNDS):
            locals_, weights = [], []
            for o in TRAIN_ORGS:
                lm = copy.deepcopy(server.global_model)
                info = train_model(lm, Xm[o], D.A[members[o]], D.y[members[o]], node_mask=D.M[members[o]],
                                   **train_kwargs(dp, seed * 1000 + rnd * 10 + o, LOCAL_EPOCHS), reinit=False)
                if dp:
                    steps[o] += info["steps"]
                    rates[o] = info["sample_rate"]
                locals_.append(lm)
                weights.append(float(len(members[o])))
            server.aggregate(locals_, weights)
        eps = (max(compute_epsilon(SYN_CONFIG["dp_sigma"], rates[o], steps[o], SYN_CONFIG["dp_delta"])
                   for o in TRAIN_ORGS) if dp else None)
        finish("fedavg_dp" if dp else "fedavg_nodp", server.global_model, eps)

    local_acc = []
    for o in TRAIN_ORGS:
        m = make_model(d)
        train_model(m, Xm[o], D.A[members[o]], D.y[members[o]], node_mask=D.M[members[o]],
                    **train_kwargs(False, seed + 20 + o, ROUNDS * LOCAL_EPOCHS))
        local_acc.append(metrics(ytest, predict(m, Xtest, D.A[test]))["accuracy"])
    out["local_only_mean"] = {"test": {"accuracy": float(np.mean(local_acc))}, "per_org": local_acc}
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
        print(f"  repeat {r + 1}: " + "  ".join(
            f"{k}={v['test']['accuracy']:.3f}" for k, v in res.items()), flush=True)
    keys = runs[0].keys()
    summary = {}
    for k in keys:
        s = {"accuracy": mean_std([x[k]["test"]["accuracy"] for x in runs])}
        if "fpr" in runs[0][k]["test"]:
            s["fpr"] = mean_std([x[k]["test"]["fpr"] for x in runs])
            s["detection_rate"] = mean_std([x[k]["test"]["detection_rate"] for x in runs])
        if runs[0][k].get("epsilon") is not None:
            s["epsilon"] = mean_std([x[k]["epsilon"] for x in runs])
        if "mia" in runs[0][k]:
            s["mia_auc"] = mean_std([x[k]["mia"]["auc"] for x in runs])
            s["mia_attack_accuracy"] = mean_std([x[k]["mia"]["attack_accuracy"] for x in runs])
        if "per_type_detection" in runs[0][k]:
            s["per_type_detection"] = {t: float(np.mean([x[k]["per_type_detection"][t] for x in runs]))
                                       for t in runs[0][k]["per_type_detection"]}
        summary[k] = s
    save_json({"config": SYN_CONFIG, "env": env_info(), "train_orgs": TRAIN_ORGS, "test_org": TEST_ORG,
               "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "repeats": args.repeats,
               "member_fraction": MEMBER_FRACTION, "summary": summary, "runs": runs}, args.output)


if __name__ == "__main__":
    main()
