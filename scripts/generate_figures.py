import argparse
import json
import os
import sys
from functools import reduce

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

C_BASE = "#7a7f87"
C_ALT = "#c9d3df"
C_FAGI = "#2a78d6"
C_S2 = "#eb6834"
C_S3 = "#1baf7a"
INK = "#222222"
GRID = "#e3e3e3"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 6.8, "savefig.bbox": "tight",
    "hatch.linewidth": 0.6,
})


def pct(x):
    return None if x is None else 100.0 * x


def get(d, path):
    return reduce(lambda acc, k: acc[int(k)] if isinstance(acc, list) else acc[k], path.split("."), d)


def check_claims(results, claims_path):
    if not os.path.exists(claims_path):
        print(f"(no {os.path.relpath(claims_path, ROOT)}; skipping paper-number check)")
        return
    with open(claims_path) as f:
        claims = json.load(f)
    bad = 0
    print("Paper numbers vs paper_results.json")
    for c in claims:
        got = get(results, c["path"]) * c.get("scale", 100.0)
        if "minus" in c:
            got -= get(results, c["minus"]) * c.get("scale", 100.0)
        ok = abs(round(got, c.get("decimals", 1)) - c["paper"]) <= c.get("tol", 0.051)
        bad += not ok
        print(f"  [{'ok' if ok else 'MISMATCH'}] {c['label']:<46} json={got:8.3f}  paper={c['paper']}")
    if bad:
        sys.exit(f"{bad} paper number(s) do not match the results JSON.")
    print(f"All {len(claims)} paper numbers match.\n")


def save(fig, out, name, dpi):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=dpi)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def label_bars(ax, bars, vals, dy=1.0, fs=5.6):
    for b, v in zip(bars, vals):
        if v is None or np.isnan(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, v + dy, f"{v:.1f}", ha="center", va="bottom",
                fontsize=fs, color=INK)


def fig_cicids(r, out, dpi):
    rows = r["cicids2017"]["baselines"]
    short = {"Random Forest": "RF", "GNN Baseline": "GNN"}
    names = [short.get(x["name"], x["name"]) for x in rows]
    clean = [pct(x["clean"]) for x in rows]
    adv = [pct(x["adv"]) if x["adv"] is not None else np.nan for x in rows]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    b1 = ax.bar(x - w / 2, clean, w, color=C_BASE, label="Clean")
    b2 = ax.bar(x + w / 2, adv, w, color=C_FAGI, hatch="////", edgecolor="white", linewidth=0,
                label="PGD, $\\epsilon$=0.1, K=20")
    label_bars(ax, b1, clean)
    label_bars(ax, b2, adv)
    for xi, v in zip(x, adv):
        if np.isnan(v):
            ax.text(xi + w / 2, 2, "n/a", ha="center", fontsize=5.6, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("Detection accuracy (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig3a_cicids_benchmark.png", dpi)


def fig_api(r, out, dpi):
    keys = [("rf", "RF"), ("gnn_nopayload", "GNN\n(5 feat.)"), ("gnn", "GNN\n(6 feat.)"),
            ("fagi_no_payload", "FAGI\n−payload"), ("fagi", "FAGI")]
    colors = [C_BASE, C_BASE, C_BASE, C_ALT, C_FAGI]
    panels = [("original", "(a) Original generator"), ("hard", "(b) Hard generator")]
    panels = [(k, t) for k, t in panels if r["api_native"].get(k)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5, 2.3), sharey=True, squeeze=False)
    for ax, (k, title) in zip(axes[0], panels):
        t = r["api_native"][k]["all_attacks"]
        vals = [pct(t[m]["clean"]) for m, _ in keys]
        errs = [pct(t[m]["clean_std"]) for m, _ in keys]
        bars = ax.bar(range(len(keys)), vals, color=colors, yerr=errs,
                      error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": INK})
        label_bars(ax, bars, vals, dy=1.5, fs=5.2)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([n for _, n in keys], fontsize=5.6)
        ax.set_ylim(0, 112)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Detection accuracy (%)")
    save(fig, out, "fig3b_api_native.png", dpi)


def fig_adversarial(r, out, dpi):
    sweep = r["cicids2017"]["eps_sweep"]
    series = [("gnn", "GNN baseline", C_S2, "s", "--"),
              ("fagi_no_dp", "FAGI − DP-SGD", C_S3, "^", ":"),
              ("fagi", "FAGI (DP-SGD, $\\sigma$=1.1)", C_FAGI, "o", "-")]
    gsa = r["graph_structure_attack"]["original"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 2.2), gridspec_kw={"width_ratios": [3, 2]})
    for key, name, c, mk, ls in series:
        eps = sorted(sweep[key], key=float)
        a.plot([float(e) for e in eps], [pct(sweep[key][e]) for e in eps], color=c, marker=mk,
               linestyle=ls, linewidth=1.4, markersize=4, label=name)
    a.set_xscale("log")
    a.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
    a.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
    a.set_xlabel("PGD budget $\\epsilon$ (std. units)")
    a.set_ylabel("Accuracy under attack (%)")
    a.set_title("(a) CICIDS2017, K=20")
    a.legend(loc="lower left")
    conds = [("clean", 0), ("edge_40pct", 40), ("edge_80pct", 80)]
    allv = []
    for key, name, c, mk, ls in [s_ for s_ in series if s_[0] in gsa]:
        v = [pct(gsa[key][cond]) for cond, _ in conds]
        allv += v
        b.plot([x_ for _, x_ in conds], v, color=c, marker=mk, linestyle=ls, linewidth=1.4, markersize=4)
    lo = min(allv)
    b.set_ylim(max(0.0, np.floor(lo) - 2), 100.5)
    b.set_xticks([0, 40, 80])
    b.set_xticklabels(["0%", "+40%", "+80%"])
    b.set_xlabel("Injected edges")
    b.set_title("(b) Synthetic, edge injection")
    save(fig, out, "fig5_adversarial.png", dpi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "experiments", "results", "paper_results.json"))
    ap.add_argument("--claims", default=os.path.join(ROOT, "experiments", "results", "paper_claims.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    with open(args.results) as f:
        r = json.load(f)
    check_claims(r, args.claims)
    os.makedirs(args.out, exist_ok=True)
    fig_cicids(r, args.out, args.dpi)
    fig_api(r, args.out, args.dpi)
    fig_adversarial(r, args.out, args.dpi)


if __name__ == "__main__":
    main()
