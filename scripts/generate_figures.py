import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_BASE = "#7a7f87"
C_FAGI = "#1f5fa8"
C_ALT = "#c9d3df"
INK = "#222222"
GRID = "#d9d9d9"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "savefig.bbox": "tight",
})


def pct(x):
    return 100.0 * x


def near(a, b, tol=0.051):
    return abs(a - b) <= tol


def check_claims(r):
    c = r["cicids2017"]["adversarial_robustness"]
    api = r["api_native"]
    fed = r["federated"]
    abl = {v["name"]: v for v in r["ablation"]["variants"]}
    gsa = api["graph_structure_attack"]
    lm = api["lateral_movement"]
    llm = api["llm_contribution"]
    claims = [
        ("Fig5a FAGI adv 74.4", pct(c["fagi_adv"]), 74.4),
        ("Fig5a GNN adv 67.2", pct(c["gnn_adv"]), 67.2),
        ("Fig5a gap +7.2pp", pct(c["fagi_adv"] - c["gnn_adv"]), 7.2),
        ("GNN collapse 74.8->67.2 = 7.6pp", pct(c["gnn_clean"] - c["gnn_adv"]), 7.6),
        ("FAGI clean - GNN clean = 0.8pp", pct(c["fagi_clean"] - c["gnn_clean"]), 0.8),
        ("Fig3b RF lateral 99.0", pct(lm["rf"]), 99.0),
        ("Fig3b GNN lateral 87.0", pct(lm["gnn"]), 87.0),
        ("Fig3b FAGI lateral 88.7", pct(lm["fagi"]), 88.7),
        ("Fig3b LLM gain +2.7pp", pct(llm["with_payload"] - llm["without_payload"]), 2.7),
        ("Edge-80 FAGI 74.4", pct(gsa["fagi"]["edge_80"]), 74.4),
        ("Edge-80 GNN 71.9", pct(gsa["gnn"]["edge_80"]), 71.9),
        ("Edge-80 gap +2.5pp", pct(gsa["fagi"]["edge_80"] - gsa["gnn"]["edge_80"]), 2.5),
        ("Federated FAGI 49.9", pct(fed["fagi_federated"]), 49.9),
        ("Central FAGI 62.8", pct(fed["fagi_central"]), 62.8),
        ("MIA FAGI 51.8", pct(fed["membership_inference"]["fagi"]), 51.8),
        ("MIA centralized 78.6", pct(fed["membership_inference"]["centralised"]), 78.6),
        ("Ablation -GNNGuard adv 68.9", pct(abl["FAGI − GNNGuard"]["adv"]), 68.9),
        ("Ablation -PGD adv 67.2", pct(abl["FAGI − PGD"]["adv"]), 67.2),
        ("Ablation -LLM clean 74.1", pct(abl["FAGI − LLM"]["clean"]), 74.1),
        ("Ablation -Federated clean 76.1", pct(abl["FAGI − Federated"]["clean"]), 76.1),
    ]
    bad = 0
    print("Caption/text claims vs results JSON")
    print("-" * 60)
    for label, got, want in claims:
        ok = near(got, want)
        bad += not ok
        print(f"  [{'ok' if ok else 'MISMATCH'}] {label:<38} json={got:6.2f}  paper={want}")
    print("-" * 60)
    if bad:
        sys.exit(f"{bad} claim(s) do not match the results JSON. Fix the paper or the data.")
    print("All claims match.\n")


def save(fig, out, name, dpi):
    png = os.path.join(out, name)
    fig.savefig(png, dpi=dpi)
    fig.savefig(os.path.splitext(png)[0] + ".pdf")
    plt.close(fig)
    print("wrote", png)


def label_bars(ax, bars, fmt="{:.1f}", dy=0.6):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=5.6, color=INK)


def fig3a(r, out, dpi):
    rows = r["cicids2017"]["baselines"]
    short = {"Signature IDS": "Signature", "Random Forest": "RF", "LSTM Network": "LSTM",
             "CNN on Features": "CNN", "GNN Baseline": "GNN", "FAGI": "FAGI"}
    names = [short.get(x["name"], x["name"]) for x in rows]
    clean = [pct(x["clean"]) for x in rows]
    adv = [pct(x["adv"]) if x["adv"] is not None else np.nan for x in rows]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    b1 = ax.bar(x - w / 2, clean, w, color=C_BASE, label="Clean")
    b2 = ax.bar(x + w / 2, adv, w, color=C_FAGI, hatch="////", edgecolor="white",
                linewidth=0, label="Under PGD (ε=0.1)")
    label_bars(ax, b1)
    label_bars(ax, [b for b, v in zip(b2, adv) if not np.isnan(v)])
    for xi, v in zip(x, adv):
        if np.isnan(v):
            ax.text(xi + w / 2, 1.5, "n/a", ha="center", fontsize=6, color="#666666")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("Detection accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig3a_cicids_benchmark.png", dpi)


def fig3b(r, out, dpi):
    lm = r["api_native"]["lateral_movement"]
    llm = r["api_native"]["llm_contribution"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 2.4), gridspec_kw={"width_ratios": [3, 2]})
    names = ["RF", "GNN", "FAGI"]
    vals = [pct(lm["rf"]), pct(lm["gnn"]), pct(lm["fagi"])]
    errs = [pct(lm["rf_std"]), pct(lm["gnn_std"]), pct(lm["fagi_std"])]
    bars = a.bar(names, vals, color=[C_BASE, C_BASE, C_FAGI], yerr=errs,
                 error_kw={"elinewidth": 0.7, "capsize": 2, "ecolor": INK})
    bars[1].set_hatch("////")
    bars[1].set_edgecolor("white")
    label_bars(a, bars, dy=2.0)
    a.set_ylim(0, 112)
    a.set_ylabel("Detection accuracy (%)")
    a.set_title("(a) Lateral movement", fontsize=7.5)
    a.grid(axis="x", visible=False)

    nb = ["Without", "With"]
    nv = [pct(llm["without_payload"]), pct(llm["with_payload"])]
    bars2 = b.bar(nb, nv, color=[C_ALT, C_FAGI])
    label_bars(b, bars2, dy=1.5)
    b.set_ylim(0, 112)
    b.set_title("(b) SQL injection:\npayload embeddings", fontsize=7.5)
    b.tick_params(axis="x", labelsize=7)
    b.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "fig3b_api_native.png", dpi)


def fig5(r, out, dpi):
    c = r["cicids2017"]["adversarial_robustness"]
    g = r["api_native"]["graph_structure_attack"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 2.4))
    x = np.arange(2)
    w = 0.36
    gb = a.bar(x - w / 2, [pct(c["gnn_clean"]), pct(c["gnn_adv"])], w, color=C_BASE, label="GNN")
    fb = a.bar(x + w / 2, [pct(c["fagi_clean"]), pct(c["fagi_adv"])], w, color=C_FAGI,
               hatch="////", edgecolor="white", linewidth=0, label="FAGI")
    label_bars(a, gb, dy=1.0)
    label_bars(a, fb, dy=1.0)
    a.set_xticks(x)
    a.set_xticklabels(["Clean", "PGD"])
    a.set_ylim(0, 100)
    a.set_ylabel("Detection accuracy (%)")
    a.set_title("(a) Feature-space PGD", fontsize=7.5)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    a.grid(axis="x", visible=False)

    xs = [0, 40, 80]
    gy = [pct(g["gnn"]["clean"]), pct(g["gnn"]["edge_40"]), pct(g["gnn"]["edge_80"])]
    fy = [pct(g["fagi"]["clean"]), pct(g["fagi"]["edge_40"]), pct(g["fagi"]["edge_80"])]
    b.plot(xs, gy, color=C_BASE, marker="o", markersize=4, linewidth=1.5, label="GNN")
    b.plot(xs, fy, color=C_FAGI, marker="s", markersize=4, linewidth=1.5, label="FAGI")
    b.set_xticks(xs)
    b.set_xlabel("Injected edges (%)")
    b.set_ylim(70, 76)
    b.set_title("(b) Edge injection", fontsize=7.5)
    b.legend(loc="center right")
    b.set_ylabel("Detection accuracy (%)")
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "fig5_adversarial.png", dpi)


def fig4(r, out, dpi):
    q = r.get("qualitative_scores")
    if not q:
        print("skipped fig4_radar.png: no \"qualitative_scores\" in the JSON. "
              "The radar axes other than privacy are extrapolated, not measured, "
              "so no values are invented here. Add them (with their provenance) "
              "or replace the radar with a table of measured values.")
        return
    axes = q["axes"]
    n = len(axes)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(3.2, 3.2))
    ax = fig.add_subplot(111, polar=True)
    for key, color, ls in (("federated", C_FAGI, "-"), ("centralized", C_BASE, "--")):
        v = q[key] + q[key][:1]
        ax.plot(ang, v, color=color, linewidth=1.5, linestyle=ls,
                label=key.capitalize())
        ax.fill(ang, v, color=color, alpha=0.12)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(axes, fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05))
    save(fig, out, "fig4_radar.png", dpi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="experiments/results/paper_results.json")
    p.add_argument("--out", default="figures")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    with open(a.results, encoding="utf-8") as f:
        r = json.load(f)
    check_claims(r)
    if a.check_only:
        return
    os.makedirs(a.out, exist_ok=True)
    fig3a(r, a.out, a.dpi)
    fig3b(r, a.out, a.dpi)
    fig5(r, a.out, a.dpi)
    fig4(r, a.out, a.dpi)


if __name__ == "__main__":
    main()
