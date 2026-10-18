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
    "temporal_train_fraction": 0.7,
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
    "pgd_restarts": 2,
    "mimicry_targets": 32,
    "eps_sweep": [0.02, 0.05, 0.1, 0.2, 0.5],
    "k100_eps": [0.2, 0.5],
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
    "gnn_fullpool": {"guard": False, "adv": False, "dp": False, "seed": 18, "fullpool": True},
}
for _s in CONFIG["sigma_sweep"]:
    GRAPH_VARIANTS[f"fagi_sigma{_s}"] = {"guard": True, "adv": True, "dp": True, "seed": 16, "sigma": _s}
GRAPH_VARIANTS["gnn_overfit"] = {"guard": False, "adv": False, "dp": False, "seed": 17, "overfit": True}
FLAT_VARIANTS = {"mlp": {"kind": "mlp", "adv": False, "dp": False, "seed": 21},
                 "mlp_at": {"kind": "mlp", "adv": True, "dp": False, "seed": 22},
                 "cnn": {"kind": "cnn", "adv": False, "dp": False, "seed": 23},
                 "mlp_dp": {"kind": "mlp", "adv": True, "dp": True, "seed": 24, "all_train": True}}
TRANSFER_SOURCES = {"mlp": "pgd", "gnn": "pgd_adaptive"}

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


def _split_private_pool(tr, labels, seed):
    try:
        priv, pool = train_test_split(tr, test_size=CONFIG["pool_fraction"], random_state=seed, stratify=labels[tr])
    except ValueError:
        priv, pool = train_test_split(tr, test_size=CONFIG["pool_fraction"], random_state=seed)
    return np.sort(priv), np.sort(pool)


class CICIDSFolds:
    def __init__(self, data_path: str, protocol: str = "cv"):
        if protocol not in ("cv", "temporal"):
            raise ValueError("protocol must be 'cv' or 'temporal'")
        X, y, labels, feats, meta, source = load_cicids2017(data_path, CONFIG["per_class_cap"],
                                                            CONFIG["sample_seed"], return_source=True)
        self.X, self.y, self.labels, self.features, self.meta = X, y, labels, feats, meta
        self.protocol = protocol
        if protocol == "cv":
            skf = StratifiedKFold(CONFIG["n_folds"], shuffle=True, random_state=CONFIG["cv_seed"])
            self.splits = list(skf.split(X, labels))
        else:
            tr, te = [], []
            for src in sorted(set(source)):
                idx = np.flatnonzero(source == src)
                cut = int(round(len(idx) * CONFIG["temporal_train_fraction"]))
                tr.append(idx[:cut])
                te.append(idx[cut:])
            self.splits = [(np.sort(np.concatenate(tr)), np.sort(np.concatenate(te)))]
        self.n_splits = len(self.splits)
        self._prepared = {}
        self.hash = config_hash()

    def fold(self, f: int):
        if f in self._prepared:
            return self._prepared[f]
        tr, te = self.splits[f]
        priv, pool = _split_private_pool(tr, self.labels, 1000 + f)
        pp = FlowPreprocessor().fit(self.X[pool])
        Zp, Zr, Zte = pp.transform(self.X[priv]), pp.transform(self.X[pool]), pp.transform(self.X[te])
        eb = EgoGraphBuilder(CONFIG["k"], CONFIG["k_inner"]).fit(Zr)
        Gp, Ap, _ = eb.build(Zp)
        Gte, Ate, nte = eb.build(Zte)
        yr = self.y[pool]
        mal_all = np.flatnonzero(self.y[te] == 1)
        mal = np.sort(np.random.RandomState(2000 + f).choice(mal_all, min(CONFIG["attack_n"], len(mal_all)),
                                                             replace=False))
        benign_nn = NearestNeighbors(n_neighbors=CONFIG["mimicry_targets"], algorithm="brute", n_jobs=-1).fit(Zr[yr == 0])
        d = {"tr": tr, "priv": priv, "pool": pool, "te": te, "pp": pp, "Zp": Zp, "Zr": Zr, "Zte": Zte, "eb": eb,
             "Gp": Gp, "Ap": Ap, "Gte": Gte, "Ate": Ate, "nbr_te": nte, "yr": yr,
             "Mp": center_mask(len(Gp), Gp.shape[1]), "yp": self.y[priv], "yte": self.y[te],
             "labels_te": self.labels[te], "labels_p": self.labels[priv], "mal": mal,
             "benign_r": Zr[yr == 0], "benign_nn": benign_nn}
        self._prepared = {f: d}
        return d

    def _path(self, kind, name, f, s):
        return os.path.join(CACHE_DIR, f"cicids-{self.protocol}_{kind}_{name}_f{f}_s{s}_{self.hash}.pt")

    def knn_predict(self, d, nbr_idx):
        votes = d["yr"][nbr_idx].mean(axis=1)
        out = (votes > 0.5).astype(np.int64)
        tie = votes == 0.5
        out[tie] = d["yr"][nbr_idx[tie, 0]]
        return out

    def predictor(self, name, model, d):
        if name == "knn":
            return lambda Z: self.knn_predict(d, d["eb"].neighbours(Z))
        if name == "rf":
            return lambda Z: model.predict(inverse_flow(d["pp"], Z))
        if name in FLAT_VARIANTS:
            return lambda Z: predict(model, *as_graph(Z))
        return lambda Z: predict(model, *d["eb"].build(Z)[:2])

    def adaptive_pgd(self, model, d, Zc, y, eps, steps, restarts, seed):
        set_seed(seed)
        alpha = 2.5 * eps / steps
        best = Zc.copy()
        evaded = np.zeros(len(Zc), dtype=bool)
        model.eval()
        for _ in range(restarts):
            delta = np.random.uniform(-eps, eps, Zc.shape).astype(np.float32)
            for _ in range(steps + 1):
                G, A, _ = d["eb"].build(Zc + delta)
                Gt = T(G).requires_grad_(True)
                logits = model(Gt, T(A))
                hit = (logits.argmax(1).numpy() == 0) & ~evaded
                best[hit] = (Zc + delta)[hit]
                evaded |= hit
                loss = F.cross_entropy(logits, T(y, torch.long))
                (g,) = torch.autograd.grad(loss, Gt)
                delta = np.clip(delta + alpha * np.sign(g[:, 0, :].numpy()), -eps, eps).astype(np.float32)
            last = Zc + delta
            best[~evaded] = last[~evaded]
        return best

    def mimicry_detected(self, pred_fn, d, Zc, eps):
        _, idx = d["benign_nn"].kneighbors(Zc)
        detected = np.ones(len(Zc), dtype=bool)
        for j in range(idx.shape[1]):
            cand = (Zc + np.clip(d["benign_r"][idx[:, j]] - Zc, -eps, eps)).astype(np.float32)
            detected &= pred_fn(cand) == 1
        return detected

    def _build_model(self, name, d):
        if name in GRAPH_VARIANTS:
            v = GRAPH_VARIANTS[name]
            return DenseFAGI(d["Gp"].shape[2], gnnguard_threshold=CONFIG["gnnguard_tau"] if v["guard"] else None,
                             readout="center")
        if name in FLAT_VARIANTS:
            return flat_model(FLAT_VARIANTS[name]["kind"], d["Zp"].shape[1])
        return None

    def _train(self, name, f, s, d):
        seed = 1000 * f + 100 * s
        if name == "rf":
            set_seed(seed + 31)
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed + 31)
            return rf.fit(self.X[d["priv"]], d["yp"]), {}
        if name == "knn":
            return None, {}
        model = self._build_model(name, d)
        if name in FLAT_VARIANTS:
            v = FLAT_VARIANTS[name]
            if v.get("all_train"):
                Z = d["pp"].transform(self.X[d["tr"]])
                y = self.y[d["tr"]]
            else:
                Z, y = d["Zp"], d["yp"]
            Xg, Ag = as_graph(Z)
            info = train_model(model, Xg, Ag, y, epochs=CONFIG["epochs"], batch_size=CONFIG["batch_size"],
                               lr=CONFIG["flat_lr"], adv=v["adv"], dp=v["dp"], sigma=CONFIG["dp_sigma"],
                               clip=CONFIG["dp_clip"], delta=CONFIG["dp_delta"], seed=seed + v["seed"],
                               pgd_eps=CONFIG["pgd_eps"], pgd_steps=CONFIG["pgd_train_steps"])
            return model, info
        v = GRAPH_VARIANTS[name]
        X, A, y, M = d["Gp"], d["Ap"], d["yp"], d["Mp"]
        epochs = CONFIG["epochs"]
        if v.get("overfit"):
            sub = np.random.RandomState(seed + 7).choice(len(y), CONFIG["overfit_n"], replace=False)
            X, A, y, M = X[sub], A[sub], y[sub], M[sub]
            epochs = CONFIG["overfit_epochs"]
            d["overfit_members"] = sub
        if v.get("fullpool"):
            Zall = d["pp"].transform(self.X[d["tr"]])
            ebf = EgoGraphBuilder(CONFIG["k"], CONFIG["k_inner"]).fit(Zall)
            X, A, _ = ebf.build(Zall, self_index=np.arange(len(Zall)))
            y = self.y[d["tr"]]
            M = center_mask(len(X), X.shape[1])
            d["eb_full"] = ebf
        info = train_model(model, X, A, y, node_mask=M, epochs=epochs, batch_size=CONFIG["batch_size"],
                           lr=CONFIG["lr"], adv=v["adv"], dp=v["dp"], sigma=v.get("sigma", CONFIG["dp_sigma"]),
                           clip=CONFIG["dp_clip"], delta=CONFIG["dp_delta"], seed=seed + v["seed"],
                           pgd_eps=CONFIG["pgd_eps"], pgd_steps=CONFIG["pgd_train_steps"])
        return model, info

    def load_model(self, name, f, s):
        blob = self.result(name, f, s)
        if name == "knn":
            return None
        if name == "rf":
            return blob["model"]
        model = self._build_model(name, self.fold(f))
        model.load_state_dict(blob["state"])
        model.eval()
        return model

    def result(self, name, f, s, k100=False):
        path = self._path("main", name, f, s)
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
        d = self.fold(f)
        model, info = self._train(name, f, s, d)
        v = GRAPH_VARIANTS.get(name, {})
        if v.get("fullpool"):
            Gt, At, _ = d["eb_full"].build(d["Zte"])
            blob = {"name": name, "fold": f, "seed": s, "info": info, "state": model.state_dict(),
                    "pred_clean": predict(model, Gt, At), "attacks": {}, "adv_examples": {}, "mia": None}
            return self._save(path, blob, d)
        pred_fn = self.predictor(name, model, d)
        pred = pred_fn(d["Zte"])
        mal = d["mal"]
        Zm, ym = d["Zte"][mal], np.ones(len(mal), dtype=np.int64)
        attacks, adv_ex = {}, {}
        light = v.get("overfit", False)
        for e in CONFIG["eps_sweep"]:
            a = {}
            if not light:
                a["mimicry32"] = self.mimicry_detected(pred_fn, d, Zm, e)
            if name in GRAPH_VARIANTS:
                Mm = center_mask(len(mal), d["Gte"].shape[1])
                xa = adversarial_inputs(model, d["Gte"][mal], d["Ate"][mal], ym, node_mask=Mm, eps=e,
                                        steps=CONFIG["pgd_eval_steps"], seed=7000 + 100 * f + s)
                a["pgd_fixed_graph"] = predict(model, xa, d["Ate"][mal]) == 1
                a["pgd_reretrieved"] = pred_fn(xa[:, 0, :]) == 1
                za = self.adaptive_pgd(model, d, Zm, ym, e, CONFIG["pgd_eval_steps"], CONFIG["pgd_restarts"],
                                       8000 + 100 * f + s)
                a["pgd_adaptive"] = pred_fn(za) == 1
                adv_ex[str(e)] = za
            elif name in FLAT_VARIANTS:
                xa = adversarial_inputs(model, *as_graph(Zm), ym, eps=e, steps=CONFIG["pgd_eval_steps"],
                                        seed=7000 + 100 * f + s)
                a["pgd"] = pred_fn(xa[:, 0, :]) == 1
                adv_ex[str(e)] = xa[:, 0, :]
            attacks[str(e)] = a
        if k100 and name in GRAPH_VARIANTS and not light:
            for e in CONFIG["k100_eps"]:
                za = self.adaptive_pgd(model, d, Zm, ym, e, 100, 1, 9000 + 100 * f + s)
                attacks[str(e)]["pgd_adaptive_k100_check"] = pred_fn(za) == 1
        mia = None
        if name in GRAPH_VARIANTS:
            lm = per_sample_loss(model, d["Gp"], d["Ap"], d["yp"])
            ln = per_sample_loss(model, d["Gte"], d["Ate"], d["yte"])
            labs = d["labels_p"]
            if v.get("overfit"):
                lm, labs = lm[d["overfit_members"]], labs[d["overfit_members"]]
            mia = {"loss_members": lm, "loss_nonmembers": ln, "labels_members": labs}
        blob = {"name": name, "fold": f, "seed": s, "info": info, "pred_clean": pred, "attacks": attacks,
                "adv_examples": adv_ex, "mia": mia}
        if name == "rf":
            blob["model"] = model
        elif model is not None:
            blob["state"] = model.state_dict()
        return self._save(path, blob, d)

    def _save(self, path, blob, d):
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(blob, path)
        acc = float(np.mean(blob["pred_clean"] == d["yte"]))
        msg = f"    [{self.protocol}] {blob['name']:<14} fold {blob['fold']} seed {blob['seed']}: clean={acc:.4f}"
        if "0.5" in blob["attacks"]:
            det = np.ones(len(d["mal"]), dtype=bool)
            for k_, v_ in blob["attacks"]["0.5"].items():
                if k_ != "pgd_fixed_graph" and not k_.endswith("_check"):
                    det &= v_
            msg += f"  own-attack recall@0.5={det.mean():.4f}"
        print(msg + f"  eps_dp={blob['info'].get('epsilon')}  ({blob['info'].get('train_seconds', 0)}s)", flush=True)
        return blob

    def transfer(self, name, f, s):
        path = self._path("transfer", name, f, s)
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
        d = self.fold(f)
        pred_fn = self.predictor(name, self.load_model(name, f, s), d)
        out = {}
        for src in TRANSFER_SOURCES:
            if src == name:
                continue
            ex = self.result(src, f, s)["adv_examples"]
            out[src] = {e: pred_fn(ex[e]) == 1 for e in ex}
        torch.save(out, path)
        return out

    def worst_detected(self, name, f, s):
        main = self.result(name, f, s)
        tr = self.transfer(name, f, s)
        out = {}
        for e, a in main["attacks"].items():
            det = np.ones(len(self.fold(f)["mal"]), dtype=bool)
            for k_, v_ in a.items():
                if k_ == "pgd_fixed_graph" or k_.endswith("_check"):
                    continue
                det &= v_
            for src, by_e in tr.items():
                det &= by_e[e]
            out[e] = det
        return out
