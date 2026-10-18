import html
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results", "paper_results.json")
OUT = os.path.join(ROOT, "dashboard.html")


def p(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}"


def ms(x, d=2):
    return "n/a" if x is None else f"{100 * x['mean']:.{d}f} ± {100 * x['std']:.{d}f}"


def table(head, rows):
    h = "".join(f"<th>{html.escape(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def main():
    with open(RES) as f:
        R = json.load(f)
    e1, e3, ab, abh = R["exp1"], R["exp3"]["cicids2017"], R["ablation"]["cicids2017"], R["ablation_hard"]["synthetic"]
    e2, e2h, e4, e4h = R["exp2"]["tasks"]["all_attacks"]["models"], R["exp2_hard"]["tasks"]["all_attacks"]["models"], \
        R["exp4"]["summary"], R["exp4_hard"]["summary"]
    ds = e1["dataset"]
    sec = []
    sec.append(("CICIDS2017 benchmark",
                f"{ds['clean_rows']:,} cleaned flows; stratified sample of {ds['sample_rows']:,}; 5-fold CV × {len(e1['config']['seeds'])} seeds. "
                "Attack recall is the share of attacked malicious flows still detected under the worst of all attacks run against that model.",
                table(["Model", "Clean acc. %", "FPR %", "Attack recall ε=0.1 %", "Attack recall ε=0.5 %", "ε_DP"],
                      [[r["name"], ms(r["clean"]), ms(r["fpr"]), ms(r["worst_recall_eps0.1"]), ms(r["worst_recall_eps0.5"]),
                        f"{r['epsilon']['mean']:.2f}" if r["epsilon"] else "—"] for r in e1["models"]])))
    sw = e3["eps_sweep_worst_recall"]
    eps = sorted(next(iter(sw.values())).keys(), key=float)
    sec.append(("Attack budget sweep (worst-case attack recall %)", "",
                table(["Model"] + [f"ε={e}" for e in eps], [[m] + [p(sw[m][e]["mean"]) for e in eps] for m in sw])))
    sec.append(("Ablation and privacy (CICIDS2017, seed 0)", "",
                table(["Variant", "Clean acc. %", "Attack recall ε=0.1 %", "Attack recall ε=0.5 %", "ε_DP"],
                      [[r["name"], ms(r["clean"]), ms(r["worst_recall_eps0.1"]), ms(r["worst_recall_eps0.5"]),
                        f"{r['epsilon']['mean']:.2f}" if r["epsilon"] else "—"] for r in ab["rows_seed0"]])
                + table(["σ", "ε_DP", "Clean acc. %", "Attack recall ε=0.5 %"],
                        [[x["sigma"] if x["sigma"] else "no DP", f"{x['epsilon']['mean']:.2f}" if x["epsilon"] else "∞",
                          ms(x["clean"]), ms(x["worst_recall_eps0.5"])] for x in ab["sigma_sweep_seed0"]])
                + table(["Model", "MIA AUC", "TPR at 1% FPR"],
                        [[k, f"{v['auc']['mean']:.3f}", p(v["tpr_at_1pct_fpr"]["mean"])] for k, v in ab["mia_seed0"].items()])))
    keys = [("rf", "Random forest"), ("gnn_nopayload", "GNN (5 feat.)"), ("gnn", "GNN (6 feat.)"),
            ("fagi_no_payload", "FAGI − payload"), ("fagi", "FAGI")]
    sec.append(("Synthetic API sessions (all attacks, accuracy %)", "",
                table(["Model", "Original generator", "Hard generator"], [[n, p(e2[k]["clean"]["mean"]), p(e2h[k]["clean"]["mean"])] for k, n in keys])))
    fk = [("rf_central", "RF, centralized"), ("central_nodp", "FAGI central, no DP"), ("central_dp", "FAGI central, DP"),
          ("central_dp_eps_matched", "FAGI central, DP, ε matched"), ("fedavg_nodp", "FedAvg, no DP"),
          ("fedavg_dp", "FedAvg, client DP-SGD"), ("local_only_mean", "Local only, no DP"), ("local_dp_mean", "Local only, DP")]
    sec.append(("Federated: accuracy on the held-out organization %", "",
                table(["Setting", "Original", "Hard", "ε_DP"],
                      [[n, p(e4[k]["accuracy"]["mean"]), p(e4h[k]["accuracy"]["mean"]),
                        f"{e4[k]['epsilon']['mean']:.2f}" if "epsilon" in e4[k] else "—"] for k, n in fk])))
    body = "".join(f"<section><h2>{html.escape(t)}</h2>{f'<p>{html.escape(d)}</p>' if d else ''}{tb}</section>" for t, d, tb in sec)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FAGI results</title><style>
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--line:#dcdbd6;--head:#f1f0ec}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a37;--head:#252523}}}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:960px;margin:0 auto}}h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:17px;margin:28px 0 8px}}
p{{color:var(--muted);margin:4px 0 10px}}.scroll{{overflow-x:auto;margin-bottom:12px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th{{background:var(--head);font-weight:600}}th:first-child,td:first-child{{text-align:left}}
</style></head><body><main><h1>FAGI results</h1>
<p>Generated by scripts/build_dashboard.py from experiments/results/paper_results.json (config {html.escape(e1['config_hash'])}). Values are mean ± std over folds and seeds.</p>
{body}</main></body></html>"""
    with open(OUT, "w") as f:
        f.write(page)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
