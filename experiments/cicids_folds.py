import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors

from common import (ROOT, DenseFAGI, T, adversarial_inputs, as_graph, flat_model, per_sample_loss, predict,
                    set_seed, train_model)
from data.cicids2017 import EgoGraphBuilder, FlowPreprocessor, center_mask, load_cicids2017

CONFIG = {
    "per_class_cap": 3000,
    "sample_seed": 42,
    "n_folds": 5,
    "cv_seed": 42,
    "seeds": [0, 1, 2],
    "pool_fraction": 0.5,
    "k": 8,
    "k_inner": 3,
    "epochs": 20,
    "batch_size": 256,
    "lr": 2e-3,
    "flat_lr": 1e-3,
    "gnnguard_tau": 0.15,
    "pgd_eps": 0.1,
    "pgd_train_steps": 10,
    "pgd_eval_steps": 20,
    "eps_sweep": [0.02, 0.05, 0.1, 0.2, 0.5],
    "dp_sigma": 1.1,
    "dp_clip": 1.0,
    "dp_delta": 1e-6,
    "sigma_sweep": [0.6, 2.0, 4.0],
    "overfit_n": 1000,
    "overfit_epochs": 300,
    "attack_n": 2000,
}

GRAPH_VARIANTS = {
    "gnn": {"guard": False, "adv": False, "dp": False, "seed": 11},
    "fagi": {"guard": True, "adv": True, "dp": True, "seed": 12},
    "fagi_no_guard": {"guard": False, "adv": True, "dp": True, "seed": 13},
    "fagi_no_pgd": {"guard": True, "adv": False, "dp": True, "seed": 14},
    "fagi_no_dp": {"guard": True, "adv": True, "dp": False, "seed": 15},
}
for _s in CONFIG["sigma_sweep"]:
    GRAPH_VARIANTS[f"fagi_sigma{_s}"] = {"guard": True, "adv": True, "dp": True, "seed": 16, "sigma": _s}
GRAPH_VARIANTS["gnn_overfit"] = {"guard": False, "adv": False, "dp": False, "seed": 17, "overfit": True}
FLAT_VARIANTS = {"mlp": {"kind": "mlp", "adv": False, "seed": 21}, "mlp_at": {"kind": "mlp", "adv": True, "seed": 22},
                 "cnn": {"kind": "cnn", "adv": False, "seed": 23}}
OTHER = ["rf", "knn"]

CACHE_DIR = os.path.join(ROOT, "experiments", "results", "cache")
CODE_FILES = ["experiments/cicids_folds.py", "experiments/common.py", "data/cicids2017.py", "fagi/model/dense.py",
              "fagi/training.py", "fagi/privacy/dp_sgd.py"]


def config_hash() -> str:
    h = hashlib.sha1(json.dumps(CONFIG, sort_keys=True).encode())
    for p in CODE_FILES:
        with open(os.path.join(ROOT, p), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:10]


def inverse_flow(pp: FlowPreprocessor, Z: np.ndarray) -> np.ndarray:
    u = Z * pp.std_ + pp.mean_
    return (np.sign(u) * np.expm1(np.abs(u))).astype(np.float32)


class CICIDSFolds:
    def __init__(self, data_path: str):
        X, y, labels, feats, meta = load_cicids2017(data_path, CONFIG["per_class_cap"], CONFIG["sample_seed"])
        self.X, self.y, self.labels, self.features, self.meta = X, y, labels, feats, meta
        skf = StratifiedKFold(CONFIG["n_folds"], shuffle=True, random_state=CONFIG["cv_seed"])
        self.splits = list(skf.split(X, labels))
        self._prepared = {}
        self.hash = config_hash()

    def fold(self, f: int):
        if f in self._prepared:
            return self._prepared[f]
        tr, te = self.splits[f]
        priv, pool = train_test_split(tr, test_size=CONFIG["pool_fraction"], random_state=1000 + f,
                                      stratify=self.labels[tr])
        priv, pool = np.sort(priv), np.sort(pool)
        pp = FlowPreprocessor().fit(self.X[pool])
        Zp, Zr, Zte = pp.transform(self.X[priv]), pp.transform(self.X[pool]), pp.transform(self.X[te])
        eb = EgoGraphBuilder(CONFIG["k"], CONFIG["k_inner"]).fit(Zr)
        Gp, Ap, _ = eb.build(Zp)
        Gte, Ate, nte = eb.build(Zte)
        yr = self.y[pool]
        benign_nn = NearestNeighbors(n_neighbors=1).fit(Zr[yr == 0])
        d = {"priv": priv, "pool": pool, "te": te, "pp": pp, "Zp": Zp, "Zr": Zr, "Zte": Zte, "eb": eb,
             "Gp": Gp, "Ap": Ap, "Gte": Gte, "Ate": Ate, "nbr_te": nte, "yr": yr,
             "Mp": center_mask(len(Gp), Gp.shape[1]), "yp": self.y[priv], "yte": self.y[te],
             "labels_te": self.labels[te], "labels_p": self.labels[priv],
             "mal": np.sort(np.random.RandomState(2000 + f).choice(np.flatnonzero(self.y[te] == 1),
                                                                  CONFIG["attack_n"], replace=False)), "benign_r": Zr[yr == 0], "benign_nn": benign_nn}
        self._prepared = {f: d}
        return d

    def _path(self, name, f, s):
        return os.path.join(CACHE_DIR, f"cicids_{name}_f{f}_s{s}_{self.hash}.pt")

    def knn_predict(self, d, nbr_idx):
        votes = d["yr"][nbr_idx].mean(axis=1)
        out = (votes > 0.5).astype(np.int64)
        tie = votes == 0.5
        out[tie] = d["yr"][nbr_idx[tie, 0]]
        return out

    def graph_predict_from_centers(self, model, d, Zc):
        G, A, _ = d["eb"].build(Zc)
        return predict(model, G, A)

    def adaptive_pgd(self, model, d, Zc, y, eps, steps, seed):
        set_seed(seed)
        alpha = 2.5 * eps / steps
        delta = np.random.uniform(-eps, eps, Zc.shape).astype(np.float32)
        model.eval()
        for _ in range(steps):
            G, A, _ = d["eb"].build(Zc + delta)
            Gt = T(G).requires_grad_(True)
            loss = F.cross_entropy(model(Gt, T(A)), T(y, torch.long))
            (g,) = torch.autograd.grad(loss, Gt)
            delta = np.clip(delta + alpha * np.sign(g[:, 0, :].numpy()), -eps, eps).astype(np.float32)
        return Zc + delta

    def mimicry(self, d, Zc, eps):
        _, idx = d["benign_nn"].kneighbors(Zc)
        b = d["benign_r"][idx[:, 0]]
        return (Zc + np.clip(b - Zc, -eps, eps)).astype(np.float32)

    def _train_graph(self, name, f, s, d):
        v = GRAPH_VARIANTS[name]
        model = DenseFAGI(d["Gp"].shape[2], gnnguard_threshold=CONFIG["gnnguard_tau"] if v["guard"] else None,
                          readout="center")
        X, A, y, M = d["Gp"], d["Ap"], d["yp"], d["Mp"]
        epochs = CONFIG["epochs"]
        if v.get("overfit"):
            rng = np.random.RandomState(1000 * f + s)
            sub = rng.choice(len(y), CONFIG["overfit_n"], replace=False)
            X, A, y, M = X[sub], A[sub], y[sub], M[sub]
            epochs = CONFIG["overfit_epochs"]
            d["overfit_members"] = sub
        info = train_model(model, X, A, y, node_mask=M, epochs=epochs, batch_size=CONFIG["batch_size"],
                           lr=CONFIG["lr"], adv=v["adv"], dp=v["dp"], sigma=v.get("sigma", CONFIG["dp_sigma"]),
                           clip=CONFIG["dp_clip"], delta=CONFIG["dp_delta"], seed=1000 * f + 100 * s + v["seed"],
                           pgd_eps=CONFIG["pgd_eps"], pgd_steps=CONFIG["pgd_train_steps"])
        return model, info

    def _train_flat(self, name, f, s, d):
        v = FLAT_VARIANTS[name]
        model = flat_model(v["kind"], d["Zp"].shape[1])
        Xp, Ap = as_graph(d["Zp"])
        info = train_model(model, Xp, Ap, d["yp"], epochs=CONFIG["epochs"], batch_size=CONFIG["batch_size"],
                           lr=CONFIG["flat_lr"], adv=v["adv"], seed=1000 * f + 100 * s + v["seed"],
                           pgd_eps=CONFIG["pgd_eps"], pgd_steps=CONFIG["pgd_train_steps"])
        return model, info

    def result(self, name, f, s):
        path = self._path(name, f, s)
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
        d = self.fold(f)
        mal = d["mal"]
        Zm, ym = d["Zte"][mal], np.ones(len(mal), dtype=np.int64)
        eps_list = CONFIG["eps_sweep"]
        K = CONFIG["pgd_eval_steps"]
        attacks = {}
        info, model = {}, None
        mlp_transfer = None
        if name in ("rf", "knn"):
            mlp_blob = self.result("mlp", f, s)
            mlp_transfer = mlp_blob["adv_examples"]

        if name in GRAPH_VARIANTS:
            model, info = self._train_graph(name, f, s, d)
            pred = predict(model, d["Gte"], d["Ate"])
            Mm = center_mask(len(mal), d["Gte"].shape[1])
            for e in eps_list:
                xa = adversarial_inputs(model, d["Gte"][mal], d["Ate"][mal], ym, node_mask=Mm, eps=e, steps=K,
                                        seed=7000 + 100 * f + s)
                fixed = predict(model, xa, d["Ate"][mal])
                rer = self.graph_predict_from_centers(model, d, xa[:, 0, :])
                ada = self.graph_predict_from_centers(model, d, self.adaptive_pgd(model, d, Zm, ym, e, K, 8000 + 100 * f + s))
                mim = self.graph_predict_from_centers(model, d, self.mimicry(d, Zm, e))
                attacks[str(e)] = {"pgd_fixed_graph": fixed, "pgd_reretrieved": rer, "pgd_adaptive": ada,
                                   "mimicry": mim}
            if not GRAPH_VARIANTS[name].get("overfit"):
                xs = adversarial_inputs(model, d["Gte"][mal], d["Ate"][mal], ym, node_mask=Mm, eps=CONFIG["pgd_eps"],
                                        steps=100, seed=9000 + 100 * f + s)
                attacks["strong_k100_fixed"] = predict(model, xs, d["Ate"][mal])
            loss_m = per_sample_loss(model, d["Gp"], d["Ap"], d["yp"])
            loss_n = per_sample_loss(model, d["Gte"], d["Ate"], d["yte"])
            if GRAPH_VARIANTS[name].get("overfit"):
                loss_m = loss_m[d["overfit_members"]]
            mia = {"loss_members": loss_m, "loss_nonmembers": loss_n, "labels_members": d["labels_p"] if not GRAPH_VARIANTS[name].get("overfit") else d["labels_p"][d["overfit_members"]]}
        elif name in FLAT_VARIANTS:
            model, info = self._train_flat(name, f, s, d)
            Xte, Ate = as_graph(d["Zte"])
            pred = predict(model, Xte, Ate)
            Xm, Am = as_graph(Zm)
            adv_ex = {}
            for e in eps_list:
                xa = adversarial_inputs(model, Xm, Am, ym, eps=e, steps=K, seed=7000 + 100 * f + s)
                adv_ex[str(e)] = xa[:, 0, :]
                attacks[str(e)] = {"pgd": predict(model, xa, Am),
                                   "mimicry": predict(model, *as_graph(self.mimicry(d, Zm, e)))}
            info["adv_examples"] = None
            mia = None
        elif name == "rf":
            set_seed(1000 * f + 100 * s + 31)
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1000 * f + 100 * s + 31)
            rf.fit(self.X[d["priv"]], d["yp"])
            pred = rf.predict(self.X[d["te"]])
            for e in eps_list:
                attacks[str(e)] = {"transfer_mlp_pgd": rf.predict(inverse_flow(d["pp"], mlp_transfer[str(e)])),
                                   "mimicry": rf.predict(inverse_flow(d["pp"], self.mimicry(d, Zm, e)))}
            mia = None
        elif name == "knn":
            pred = self.knn_predict(d, d["nbr_te"])
            for e in eps_list:
                attacks[str(e)] = {"transfer_mlp_pgd": self.knn_predict(d, d["eb"].neighbours(mlp_transfer[str(e)])),
                                   "mimicry": self.knn_predict(d, d["eb"].neighbours(self.mimicry(d, Zm, e)))}
            mia = None
        else:
            raise KeyError(name)

        worst = {}
        for e in eps_list:
            a = attacks[str(e)]
            keys = [k for k in a if k != "pgd_fixed_graph"]
            detected = np.ones(len(mal), dtype=bool)
            for k in keys:
                detected &= a[k] == 1
            worst[str(e)] = detected
        blob = {"name": name, "fold": f, "seed": s, "info": {k: v for k, v in info.items() if k != "adv_examples"},
                "pred_clean": pred, "attacks": attacks, "worst_detected": worst, "mia": mia}
        if name in FLAT_VARIANTS:
            blob["adv_examples"] = adv_ex
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(blob, path)
        acc = float(np.mean(pred == d["yte"]))
        print(f"    {name:<14} fold {f} seed {s}: clean={acc:.4f}  worst-case recall@0.1={worst['0.1'].mean():.4f}"
              f"  @0.5={worst['0.5'].mean():.4f}  eps_dp={info.get('epsilon')}  ({info.get('train_seconds', 0)}s)",
              flush=True)
        return blob
