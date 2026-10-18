import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, NullLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

C_BASE = "#7a7f87"
C_FAGI = "#2a78d6"
INK = "#222222"
GRID = "#e3e3e3"
SERIES = {
    "fagi": ("FAGI (DP-SGD)", "#2a78d6", "o", "-"),
    "gnn": ("GNN baseline", "#eb6834", "s", "--"),
    "fagi_no_dp": ("FAGI − DP-SGD", "#1baf7a", "^", ":"),
    "knn": ("k-NN vote", "#eda100", "D", "-."),
    "mlp_at": ("MLP + PGD training", "#6250d6", "v", (0, (5, 1, 1, 1))),
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 6.5, "savefig.bbox": "tight", "hatch.linewidth": 0.6,
})


def pct(x):
    return None if x is None else 100.0 * x


def save(fig, out, name, dpi):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=dpi)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def fig_benchmark(r, out, dpi):
    rows = r["exp1"]["models"]
    short = {"rf": "RF", "knn": "k-NN", "mlp": "MLP", "mlp_at": "MLP-AT", "cnn": "CNN", "gnn": "GNN", "fagi": "FAGI"}
    names = [short[x["key"]] for x in rows]
    clean = [pct(x["clean"]["mean"]) for x in rows]
    w01 = [pct(x["worst_recall_eps0.1"]["mean"]) for x in rows]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    b1 = ax.bar(x - w / 2, clean, w, color=C_BASE, label="Clean accuracy")
    b2 = ax.bar(x + w / 2, w01, w, color=C_FAGI, hatch="////", edgecolor="white", linewidth=0,
                label="Attack recall, worst case, ε=0.1")
    for bars, vals in ((b1, clean), (b2, w01)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=5.2,
                    rotation=90, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.8)
    ax.set_ylabel("%")
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig1_cicids_benchmark.png", dpi)


def fig_sweep(r, out, dpi):
    sw = r["exp3"]["cicids2017"]["eps_sweep_worst_recall"]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for key, (name, c, mk, ls) in SERIES.items():
        eps = sorted(sw[key], key=float)
        xs = [float(e) for e in eps]
        ys = [pct(sw[key][e]["mean"]) for e in eps]
        es = [pct(sw[key][e]["std"]) for e in eps]
        ax.errorbar(xs, ys, yerr=es, color=c, marker=mk, linestyle=ls, linewidth=1.3, markersize=3.8, capsize=1.5,
                    elinewidth=0.6, label=name)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
    ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
    ax.set_xlabel("Attack budget $\\epsilon_{adv}$ (L∞, standardized units)")
    ax.set_ylabel("Attack recall, worst case (%)")
    ax.set_ylim(-3, 103)
    ax.legend(loc="lower left", ncol=1)
    save(fig, out, "fig2_eps_sweep.png", dpi)


def fig_privacy(r, out, dpi):
    sw = r["ablation"]["cicids2017"]["sigma_sweep_seed0"]
    dp = [x for x in sw if x["sigma"] is not None]
    nodp = [x for x in sw if x["sigma"] is None][0]
    dp.sort(key=lambda x: x["epsilon"]["mean"])
    xs = [x["epsilon"]["mean"] for x in dp]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for field, name, c, mk, ls in [("clean", "Clean accuracy", C_BASE, "s", "-"),
                                   ("worst_recall_eps0.1", "Attack recall, $\\epsilon_{adv}$=0.1", "#1baf7a", "^", ":"),
                                   ("worst_recall_eps0.5", "Attack recall, $\\epsilon_{adv}$=0.5", C_FAGI, "o", "--")]:
        ys = [pct(x[field]["mean"]) for x in dp]
        es = [pct(x[field]["std"]) for x in dp]
        ax.errorbar(xs, ys, yerr=es, color=c, marker=mk, linestyle=ls, linewidth=1.3, markersize=3.8, capsize=1.5,
                    elinewidth=0.6, label=name)
        ax.axhline(pct(nodp[field]["mean"]), color=c, linewidth=0.7, alpha=0.7, linestyle=(0, (1, 2)))
    for x in dp:
        ax.annotate(f"σ={x['sigma']}", (x["epsilon"]["mean"], pct(x["worst_recall_eps0.5"]["mean"])),
                    textcoords="offset points", xytext=(0, -14), ha="center", fontsize=5.6, color="#555555")
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ticks = [0.5, 1, 2, 5, 10]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(0.38, 20)
    ax.set_ylim(40, 103)
    ax.set_xlabel("Privacy budget $\\epsilon_{DP}$ (δ = 10$^{-6}$)")
    ax.set_ylabel("%")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    save(fig, out, "fig3_privacy_utility.png", dpi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "experiments", "results", "paper_results.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    with open(args.results) as f:
        r = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    fig_benchmark(r, args.out, args.dpi)
    fig_sweep(r, args.out, args.dpi)
    fig_privacy(r, args.out, args.dpi)


if __name__ == "__main__":
    main()
