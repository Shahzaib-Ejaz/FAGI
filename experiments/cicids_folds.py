import hashlib
import json
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from common import (ROOT, DenseFAGI, adversarial_inputs, as_graph, flat_model, predict,
                    per_sample_loss, set_seed, train_model)
from data.cicids2017 import EgoGraphBuilder, FlowPreprocessor, center_mask, load_cicids2017

CONFIG = {
    "per_class_cap": 3000,
    "sample_seed": 42,
    "n_folds": 5,
    "cv_seed": 42,
    "k": 8,
    "k_inner": 3,
    "epochs": 20,
    "batch_size": 256,
    "lr": 2e-3,
    "gnnguard_tau": 0.15,
    "pgd_eps": 0.1,
    "pgd_train_steps": 10,
    "pgd_eval_steps": 20,
    "dp_sigma": 1.1,
    "dp_clip": 1.0,
    "dp_delta": 1e-5,
}

GRAPH_VARIANTS = {
    "gnn": {"guard": False, "adv": False, "dp": False, "seed": 11},
    "fagi": {"guard": True, "adv": True, "dp": True, "seed": 12},
    "fagi_no_guard": {"guard": False, "adv": True, "dp": True, "seed": 13},
    "fagi_no_pgd": {"guard": True, "adv": False, "dp": True, "seed": 14},
    "fagi_no_dp": {"guard": True, "adv": True, "dp": False, "seed": 15},
}
FLAT_VARIANTS = {"mlp": 21, "cnn": 22, "lstm": 23}

CACHE_DIR = os.path.join(ROOT, "experiments", "results", "cache")


def config_hash() -> str:
    return hashlib.sha1(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:10]


class CICIDSFolds:
    def __init__(self, data_path: str):
        X, y, labels, feats, meta = load_cicids2017(
            data_path, CONFIG["per_class_cap"], CONFIG["sample_seed"])
        self.X, self.y, self.labels, self.features, self.meta = X, y, labels, feats, meta
        skf = StratifiedKFold(CONFIG["n_folds"], shuffle=True, random_state=CONFIG["cv_seed"])
        self.splits = list(skf.split(X, labels))
        self._prepared = {}

    def fold(self, f: int):
        if f in self._prepared:
            return self._prepared[f]
        tr, te = self.splits[f]
        pp = FlowPreprocessor().fit(self.X[tr])
        Ztr, Zte = pp.transform(self.X[tr]), pp.transform(self.X[te])
        eb = EgoGraphBuilder(CONFIG["k"], CONFIG["k_inner"]).fit(Ztr)
        Gtr, Atr, _ = eb.build(Ztr, self_index=np.arange(len(Ztr)))
        Gte, Ate, _ = eb.build(Zte)
        d = {"tr": tr, "te": te, "Ztr": Ztr, "Zte": Zte, "builder": eb,
             "Gtr": Gtr, "Atr": Atr, "Gte": Gte, "Ate": Ate,
             "Mtr": center_mask(len(Gtr), Gtr.shape[1]), "Mte": center_mask(len(Gte), Gte.shape[1]),
             "ytr": self.y[tr], "yte": self.y[te], "labels_te": self.labels[te]}
        self._prepared = {f: d}
        return d

    def _cache_path(self, name: str, f: int) -> str:
        return os.path.join(CACHE_DIR, f"cicids_{name}_f{f}_{config_hash()}.pt")

    def graph_model(self, name: str, f: int):
        v = GRAPH_VARIANTS[name]
        d = self.fold(f)
        model = DenseFAGI(d["Gtr"].shape[2], gnnguard_threshold=CONFIG["gnnguard_tau"] if v["guard"] else None,
                          readout="center")
        path = self._cache_path(name, f)
        if os.path.exists(path):
            blob = torch.load(path, weights_only=False)
            model.load_state_dict(blob["state"])
            model.eval()
            return model, blob
        info = train_model(model, d["Gtr"], d["Atr"], d["ytr"], node_mask=d["Mtr"],
                           epochs=CONFIG["epochs"], batch_size=CONFIG["batch_size"], lr=CONFIG["lr"],
                           adv=v["adv"], dp=v["dp"], sigma=CONFIG["dp_sigma"], clip=CONFIG["dp_clip"],
                           delta=CONFIG["dp_delta"], seed=1000 * f + v["seed"],
                           pgd_eps=CONFIG["pgd_eps"], pgd_steps=CONFIG["pgd_train_steps"])
        pc = predict(model, d["Gte"], d["Ate"])
        xa = adversarial_inputs(model, d["Gte"], d["Ate"], d["yte"], node_mask=d["Mte"],
                                eps=CONFIG["pgd_eps"], steps=CONFIG["pgd_eval_steps"], seed=7 + f)
        pa = predict(model, xa, d["Ate"])
        blob = {"state": model.state_dict(), "info": info, "pred_clean": pc, "pred_adv": pa}
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(blob, path)
        print(f"    trained {name} fold {f}: {info['train_seconds']}s  clean={np.mean(pc == d['yte']):.4f}"
              f"  adv={np.mean(pa == d['yte']):.4f}  eps={info.get('epsilon')}", flush=True)
        return model, blob

    def flat_model(self, name: str, f: int):
        d = self.fold(f)
        path = self._cache_path(name, f)
        if name == "rf":
            if os.path.exists(path):
                return None, torch.load(path, weights_only=False)
            set_seed(1000 * f + 31)
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1000 * f + 31)
            rf.fit(self.X[d["tr"]], d["ytr"])
            blob = {"pred_clean": rf.predict(self.X[d["te"]]), "pred_adv": None, "info": {}}
            os.makedirs(CACHE_DIR, exist_ok=True)
            torch.save(blob, path)
            return None, blob
        model = flat_model(name, d["Ztr"].shape[1])
        Xtr, Atr = as_graph(d["Ztr"])
        Xte, Ate = as_graph(d["Zte"])
        if os.path.exists(path):
            blob = torch.load(path, weights_only=False)
            model.load_state_dict(blob["state"])
            model.eval()
            return model, blob
        info = train_model(model, Xtr, Atr, d["ytr"], epochs=CONFIG["epochs"], batch_size=CONFIG["batch_size"],
                           lr=1e-3, seed=1000 * f + FLAT_VARIANTS[name])
        pc = predict(model, Xte, Ate)
        xa = adversarial_inputs(model, Xte, Ate, d["yte"], eps=CONFIG["pgd_eps"],
                                steps=CONFIG["pgd_eval_steps"], seed=7 + f)
        pa = predict(model, xa, Ate)
        blob = {"state": model.state_dict(), "info": info, "pred_clean": pc, "pred_adv": pa}
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(blob, path)
        print(f"    trained {name} fold {f}: {info['train_seconds']}s  clean={np.mean(pc == d['yte']):.4f}"
              f"  adv={np.mean(pa == d['yte']):.4f}", flush=True)
        return model, blob

    def losses(self, model, f: int):
        d = self.fold(f)
        return (per_sample_loss(model, d["Gtr"], d["Atr"], d["ytr"]),
                per_sample_loss(model, d["Gte"], d["Ate"], d["yte"]))
