"""Build dashboard.html from experiments/results/paper_results.json.

The page layout, styles and chart script live in scripts/dashboard_template.html.
This script fills the @@MARKERS@@ in that template with tables and text computed
from the results file, so no number on the page is typed in by hand.
"""
import html
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results", "paper_results.json")
TEMPLATE = os.path.join(ROOT, "scripts", "dashboard_template.html")
OUT = os.path.join(ROOT, "dashboard.html")

NAMES = {
    "rf": "Random Forest", "knn": "k-NN vote", "mlp": "MLP", "mlp_at": "MLP + PGD training",
    "mlp_dp": "MLP + PGD + DP-SGD", "cnn": "1D-CNN", "gnn": "GNN Baseline", "fagi_no_dp": "FAGI − DP-SGD",
    "fagi": "FAGI", "gnn_overfit": "GNN, deliberately overfit",
}
ORDER = ["rf", "knn", "mlp", "mlp_at", "mlp_dp", "cnn", "gnn", "fagi_no_dp", "fagi"]
BUDGETS = ["0.02", "0.05", "0.1", "0.2", "0.5"]
E = html.escape


def M(o):
    return o["mean"]


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def ms(o, d=2):
    return f'{100 * o["mean"]:.{d}f} <span class="uncertainty">± {100 * o["std"]:.{d}f}</span>'


def dash():
    return "—"


def th(label, small=None, scope="col"):
    s = f"<small>{small}</small>" if small else ""
    return f'<th scope="{scope}">{label}{s}</th>'


def row(label, cells, focus=False, data_model=None, small=None):
    cls = ' class="focus-row"' if focus else ""
    dm = f' data-model="{data_model}"' if data_model else ""
    sm = f"<small>{small}</small>" if small else ""
    return f"<tr{cls}{dm}><th scope=\"row\">{label}{sm}</th>{''.join(cells)}</tr>"


def td(text, best=False, extra=""):
    return f'<td{" class=\"best\"" if best else ""}{extra}>{text}</td>'


def head_body(head_cells, rows):
    return f"<thead><tr>{''.join(head_cells)}</tr></thead>\n<tbody>\n{chr(10).join(rows)}\n</tbody>"


def eps_cell(o):
    return f"{M(o):.2f}" if o else "—"


def main():
    with open(RES, encoding="utf8") as f:
        R = json.load(f)
    e1, cfg, ds = R["exp1"], R["exp1"]["config"], R["exp1"]["dataset"]
    B = {m["key"]: m for m in e1["models"]}
    sw = R["exp3"]["cicids2017"]["eps_sweep_worst_recall"]
    rc = R["robust_checks"]
    FX, FX0, FXT = rc["fixed_fpr"]["models"], rc["fixed_fpr_seed0"]["models"], rc["fixed_fpr_temporal"]["models"]
    grid, target = rc["fixed_fpr"]["fpr_grid"], rc["fixed_fpr"]["fpr_target"]
    T = f"{target:g}"
    fpr_txt = f"{100 * target:g}%"
    ab = R["ablation"]["cicids2017"]
    can = R["mia_canary"]["summary"]
    tp = R["exp5_temporal"]
    TPB = {m["key"]: m for m in tp["models"]}
    h = e1["config_hash"]
    kk = cfg["k"]
    NAMES["knn"] = f"k-NN vote (k={kk})"
    nf, ns = cfg["n_folds"], len(cfg["seeds"])

    def fx(table, key, f, eps):
        return table[key][f"fixed_{f}_{eps}"]

    # ---------- study line and summary ----------
    study = (f"<span>CICIDS2017</span><span>{ds['clean_rows']:,} cleaned flows</span>"
             f"<span>{ds['sample_rows']:,} sampled flows</span>")
    fa = B["fagi"]
    matched_fagi = fx(FX, "fagi", T, "0.5")
    summary = (
        f'<div><dt>Clean accuracy</dt><dd>{pct(M(fa["clean"]), 2)}<span class="unit">%</span>'
        f'<small>± {pct(fa["clean"]["std"], 2)} standard deviation</small></dd></div>'
        f'<div class="featured"><dt>Attack recall · ε = 0.5 · {fpr_txt} FPR</dt><dd>{pct(M(matched_fagi), 2)}<span class="unit">%</span>'
        f'<small>± {pct(matched_fagi["std"], 2)} standard deviation. At FAGI\'s own threshold '
        f'({pct(M(fa["fpr"]), 2)}% FPR): {pct(M(fa["worst_recall_eps0.5"]), 2)}%.</small></dd></div>'
        f'<div><dt>Privacy budget · ε<sub>DP</sub></dt><dd>{M(fa["epsilon"]):.2f}<small>Computed by the Rényi-DP accountant</small></dd></div>'
        f'<div><dt>Evaluation protocol</dt><dd>{nf}<span class="times">×</span>{ns}<small>Cross-validation folds × seeds</small></dd></div>')

    # ---------- Table 1 ----------
    keys1 = [m["key"] for m in e1["models"]]

    def best_of(vals, hi=True):
        vals = {k: v for k, v in vals.items() if v is not None}
        t = max(vals.values()) if hi else min(vals.values())
        return {k for k, v in vals.items() if v == t}

    b_clean = best_of({k: M(B[k]["clean"]) for k in keys1})
    b_fpr = best_of({k: M(B[k]["fpr"]) for k in keys1}, hi=False)
    b_r1 = best_of({k: M(B[k]["worst_recall_eps0.1"]) for k in keys1})
    b_r5 = best_of({k: M(B[k]["worst_recall_eps0.5"]) for k in keys1})
    b_m5 = best_of({k: M(fx(FX, k, T, "0.5")) if k in FX else None for k in keys1})
    rows = []
    for k in keys1:
        m = B[k]
        cells = [td(ms(m["clean"]), k in b_clean), td(ms(m["fpr"]), k in b_fpr),
                 td(ms(m["worst_recall_eps0.1"]), k in b_r1), td(ms(m["worst_recall_eps0.5"]), k in b_r5),
                 td(ms(fx(FX, k, T, "0.5")) if k in FX else "—", k in b_m5), td(eps_cell(m["epsilon"]))]
        rows.append(row(NAMES[k], cells, focus=(k == "fagi"), small="All training data private" if k == "mlp_dp" else None))
    t1 = head_body([th("Model"), th("Clean accuracy", "% · higher is better"), th("FPR", "% · lower is better"),
                    th("Attack recall", "ε = 0.1 · default thr. · %"), th("Attack recall", "ε = 0.5 · default thr. · %"),
                    th("Attack recall", f"ε = 0.5 · matched {fpr_txt} FPR · %"), th("ε<sub>DP</sub>")], rows)
    bench_desc = (f"A stratified sample evaluated with {nf}-fold cross-validation across {ns} seeds. Attack recall is the share of attacked "
                  "malicious flows still detected under the worst of all attacks run against each model. The first two recall columns use each "
                  f"model's own default threshold, so models flag different amounts of benign traffic (see FPR). The last recall column moves "
                  f"every model to a {fpr_txt} false-positive rate on benign test flows.")
    thr_lo = min(M(FX[k][f"threshold_{T}"]) for k in FX if k not in ("rf",))
    t1_note = ("Matched-threshold recall re-scores the stored attacks at the new threshold; it does not re-optimize them, so it is an upper bound "
               "for models whose matched threshold is above 0.5.")
    if M(FX["rf"][f"threshold_{T}"]) < 0.5:
        t1_note += f" The random forest's matched threshold is {M(FX['rf'][f'threshold_{T}']):.2f}, below 0.5, which may favor it."

    # ---------- Tables 2 and 3 ----------
    def sweep_rows(getter):
        out = []
        for k in ORDER:
            vals = getter(k)
            out.append(row(NAMES[k], [td(pct(v)) for v in vals], focus=(k == "fagi"), data_model=k))
        return out

    hdr = [th("Model")] + [th(f"ε = {e}") for e in BUDGETS]
    t2 = head_body(hdr, sweep_rows(lambda k: [M(fx(FX, k, T, e)) for e in BUDGETS]))
    t3 = head_body(hdr, sweep_rows(lambda k: [M(sw[k][e]) for e in BUDGETS]))
    t2_note = (f"Every model's threshold is set so that {fpr_txt} of benign test flows are flagged (the same oracle choice for every model; ties are "
               "broken at random). A second calibration on benign reference-pool flows, each excluded from its own neighborhood, is stored under "
               "<code>pool_*</code> in <code>robust_checks.json</code>; the private MLP is excluded from it because it trains on the pool.")

    # ---------- Table 4 ----------
    fk = [f"{g:g}" for g in grid]
    rows = []
    for k in ORDER:
        cells = [td(pct(M(FX[k]["fpr_default"]), 2))] if "fpr_default" in FX[k] else [td("—")]
        cells.append(td(f"{M(FX[k][f'threshold_{T}']):.3f}"))
        cells += [td(ms(fx(FX, k, f, "0.5"), 1)) for f in fk]
        rows.append(row(NAMES[k], cells, focus=(k == "fagi")))
    t4 = head_body([th("Model"), th("Default-threshold FPR", "%"), th(f"Threshold at {fpr_txt}", "score"),
                    *[th(f"Recall at {100 * g:g}% FPR", "ε = 0.5 · %") for g in grid]], rows)
    npn = [M(FX[k][f"threshold_{T}"]) for k in ("mlp", "mlp_at", "cnn", "gnn", "fagi_no_dp")]
    t4_note = (f"Thresholds are score cut-offs on a 0–1 scale: FAGI {M(FX['fagi'][f'threshold_{T}']):.3f}, MLP + PGD + DP-SGD {M(FX['mlp_dp'][f'threshold_{T}']):.3f}, "
               f"and {min(npn):.2f}–{max(npn):.2f} for the non-private neural models.")
    if M(FX["fagi"][f"threshold_{T}"]) > 0.99 and M(FX["mlp_dp"][f"threshold_{T}"]) > 0.99 and max(npn) < 0.7:
        t4_note += " Private models need cut-offs near 1, which is why their recall depends strongly on the operating point."

    # ---------- Table 5 ----------
    cur = rc["restart_curve"]["curve"]
    rs = sorted(next(iter(cur.values())).keys(), key=int)
    rows = [row(NAMES[k], [td(pct(M(cur[k][r]))) for r in rs], focus=(k == "fagi")) for k in ("gnn", "fagi_no_dp", "fagi") if k in cur]
    t5 = head_body([th("Model")] + [th(f"{r} restart{'s' if r != '1' else ''}") for r in rs], rows)
    t5_note = (f"{rc['restart_curve']['steps']}-step adaptive PGD at ε = {rc['restart_curve']['eps_adv']}, default thresholds, seed 0. "
               "Recall that keeps falling as restarts increase means the attack has not converged on that model.")

    # ---------- Table 6 ----------
    rows = []
    for r in ab["rows_seed0"]:
        k = r["key"]
        mt = FX0.get(k)
        cells = [td(ms(r["clean"])), td(ms(r["worst_recall_eps0.1"])), td(ms(r["worst_recall_eps0.5"])),
                 td(ms(mt[f"fixed_{T}_0.5"]) if mt else "—"), td(eps_cell(r["epsilon"]))]
        rows.append(row(E(r["name"]), cells, focus=(k == "fagi")))
    t6 = head_body([th("Variant"), th("Clean accuracy", "%"), th("Attack recall", "ε = 0.1 · default thr. · %"),
                    th("Attack recall", "ε = 0.5 · default thr. · %"), th("Attack recall", f"ε = 0.5 · matched {fpr_txt} FPR · %"), th("ε<sub>DP</sub>")], rows)
    TA = ab["tests_seed0"]
    t6_note = ("Mean ± standard deviation over the five folds of seed 0. Default-threshold differences between variants partly reflect different false-positive rates; "
               "the matched column removes that effect.")

    # ---------- Table 7 ----------
    rows = []
    for x in ab["sigma_sweep_seed0"]:
        k = x["key"]
        mt = FX0.get(k)
        label = "No DP" if x["sigma"] is None else f"{x['sigma']:g}"
        cells = [td(eps_cell(x["epsilon"]) if x["epsilon"] else "∞"), td(ms(x["clean"])), td(ms(x["worst_recall_eps0.5"])),
                 td(ms(mt[f"fixed_{T}_0.5"]) if mt else "—")]
        rows.append(row(label, cells, focus=(x["sigma"] == cfg["dp_sigma"])))
    t7 = head_body([th("σ"), th("ε<sub>DP</sub>"), th("Clean accuracy", "%"), th("Attack recall", "ε = 0.5 · default thr. · %"),
                    th("Attack recall", f"ε = 0.5 · matched {fpr_txt} FPR · %")], rows)
    dp_rows = [x for x in ab["sigma_sweep_seed0"] if x["sigma"] is not None and x["key"] in FX0]
    dp_rows.sort(key=lambda x: x["sigma"])
    d_def = M(dp_rows[-1]["worst_recall_eps0.5"]) - M(dp_rows[0]["worst_recall_eps0.5"])
    d_mat = M(FX0[dp_rows[-1]["key"]][f"fixed_{T}_0.5"]) - M(FX0[dp_rows[0]["key"]][f"fixed_{T}_0.5"])
    if d_def * d_mat < 0:
        t7_note = ("From the smallest to the largest σ, default-threshold recall "
                   f"{'rises' if d_def > 0 else 'falls'} while matched-threshold recall {'rises' if d_mat > 0 else 'falls'}; "
                   "the default-threshold trend reflects changing false-positive rates.")
    else:
        t7_note = "Default and matched columns move in the same direction across σ."

    # ---------- Table 8 ----------
    mia = ab["mia_seed0"]
    rows = []
    for k in ("gnn_overfit", "gnn", "fagi_no_dp", "fagi"):
        v = mia[k]
        c = can.get(k)
        if c:
            lo, hi = c["auc_pooled_ci95"]
            spl = c["eps_lower_bound_over_splits"]["values"]
            cells = [td(f"{M(v['auc']):.3f}"), td(f"{c['auc_pooled']:.3f} <span class=\"uncertainty\">[{lo:.3f}, {hi:.3f}]</span>"),
                     td(f"{c['empirical_eps_lower_bound']:.2f}"), td(f"{min(spl):.2f}–{max(spl):.2f}"),
                     td(f"{c['epsilon']:.2f}" if c.get("epsilon") else "—")]
        else:
            cells = [td(f"{M(v['auc']):.3f}"), td("—"), td("—"), td("—"), td("—")]
        rows.append(row(f"<code>{k}</code>", cells, focus=(k == "fagi")))
    t8 = head_body([th("Model"), th("Loss-MIA AUC", "seed 0"), th("Canary AUC", "pooled · 95% CI"), th("Empirical ε lower bound"),
                    th("Range over splits", f"{len(next(iter(can.values()))['eps_lower_bound_over_splits']['values'])} splits"), th("ε<sub>DP</sub>", "audited model")], rows)
    t8_note = (f"Loss-based membership inference uses a positive control (<code>gnn_overfit</code>). The canary audit plants {R['mia_canary']['n_canary_per_fold']} "
               "label-flipped private flows per fold, half of them trained on. The ε lower bound is a split-sample Clopper–Pearson bound clamped at 0; "
               "it treats the canaries as independent and cannot exceed the accountant's ε. The reported bound uses one split fixed in advance; the range column shows "
               "how much it moves across splits and cannot be combined across them.")

    # ---------- Table 9 ----------
    tkeys = [m["key"] for m in tp["models"]]
    b_c = best_of({k: M(TPB[k]["clean"]) for k in tkeys})
    b_f = best_of({k: M(TPB[k]["fpr"]) for k in tkeys}, hi=False)
    b_m1 = best_of({k: M(FXT[k][f"fixed_{T}_0.1"]) for k in tkeys})
    b_m5 = best_of({k: M(FXT[k][f"fixed_{T}_0.5"]) for k in tkeys})
    rows = []
    for k in tkeys:
        m = TPB[k]
        cells = [td(ms(m["clean"]), k in b_c), td(ms(m["fpr"]), k in b_f), td(ms(m["worst_recall_eps0.1"])), td(ms(m["worst_recall_eps0.5"])),
                 td(ms(FXT[k][f"fixed_{T}_0.1"], 1), k in b_m1), td(ms(FXT[k][f"fixed_{T}_0.5"], 1), k in b_m5), td(eps_cell(m["epsilon"]))]
        rows.append(row(NAMES[k], cells, focus=(k == "fagi")))
    t9 = head_body([th("Model"), th("Clean accuracy", "%"), th("FPR", "%"), th("Recall", "ε = 0.1 · default · %"), th("Recall", "ε = 0.5 · default · %"),
                    th("Recall", f"ε = 0.1 · matched {fpr_txt} · %"), th("Recall", f"ε = 0.5 · matched {fpr_txt} · %"), th("ε<sub>DP</sub>")], rows)
    drops = [100 * (M(B[k]["clean"]) - M(TPB[k]["clean"])) for k in tkeys if k in B]
    sp = tp["split"]
    temporal_desc = (f"Within each day file the earlier {int(round(100 * cfg['temporal_train_fraction']))}% of rows train and the later "
                     f"{int(round(100 * (1 - cfg['temporal_train_fraction'])))}% test ({sp['private_train']:,} private training flows, {sp['reference_pool']:,} pool flows, "
                     f"{sp['test']:,} test flows). Every model is retrained on the earlier period. MachineLearningCSV has no timestamps, so row order is assumed to follow flow order.")
    t9_note = (f"Compared with random cross-validation, clean accuracy changes by {-max(drops):.2f} to {-min(drops):.2f} points across the models present in both settings "
               f"(a negative number is a drop). Matched-threshold recall at ε = 0.5 is at most {max(M(FXT[k][f'fixed_{T}_0.5']) for k in tkeys) * 100:.1f}% for every model. "
               "Matched thresholds are calibrated on the same test flows (oracle); a threshold calibrated on the training-period pool flags a different share of later benign flows.")
    if all(d > 0 for d in drops):
        t9_note = (f"Clean accuracy falls for every model compared with random cross-validation; the smallest drop among the models in both settings is {min(drops):.2f} points. "
                   + t9_note.split("(a negative number is a drop). ", 1)[1])

    # ---------- Table 10 ----------
    e2 = R["exp2"]["tasks"]["all_attacks"]["models"]
    e2h = R["exp2_hard"]["tasks"]["all_attacks"]["models"]
    syn = [("rf", "Random Forest"), ("gnn_nopayload", "GNN (5 feat.)"), ("gnn", "GNN (6 feat.)"),
           ("fagi_no_payload", "FAGI − payload"), ("fagi", "FAGI")]
    rows = []
    for k, n in syn:
        a, b = 100 * M(e2[k]["clean"]), 100 * M(e2h[k]["clean"])
        rows.append(row(n, [f'<td class="data-bar" style="--value:{a:.1f}%">{a:.1f}</td>',
                            f'<td class="data-bar" style="--value:{b:.1f}%">{b:.1f}</td>'], focus=(k == "fagi")))
    t10 = head_body([th("Model"), th("Original generator", "Accuracy, %"), th("Hard generator", "Accuracy, %")], rows)

    # ---------- Table 11 ----------
    e4, e4h = R["exp4"]["summary"], R["exp4_hard"]["summary"]
    fed = [("rf_central", "RF, centralized"), ("central_nodp", "FAGI central, no DP"), ("central_dp", "FAGI central, DP"),
           ("central_dp_eps_matched", "FAGI central, DP, ε matched"), ("fedavg_nodp", "FedAvg, no DP"),
           ("fedavg_dp", "FedAvg, client DP-SGD"), ("fedavg_dp_reset_optimizer", "FedAvg, client DP-SGD, optimizer reset"),
           ("local_only_mean", "Local only, no DP"), ("local_dp_mean", "Local only, DP")]
    rows = []
    for k, n in fed:
        a, b = 100 * M(e4[k]["accuracy"]), 100 * M(e4h[k]["accuracy"])
        eps = f"{M(e4[k]['epsilon']):.2f}" if "epsilon" in e4[k] else "—"
        rows.append(row(n, [f'<td class="data-bar" style="--value:{a:.1f}%">{a:.1f}</td>',
                            f'<td class="data-bar" style="--value:{b:.1f}%">{b:.1f}</td>', td(eps)]))
    t11 = head_body([th("Setting"), th("Original", "Accuracy, %"), th("Hard", "Accuracy, %"), th("ε<sub>DP</sub>")], rows)
    t11_note = (f"Organizations {', '.join(str(o) for o in R['exp4']['train_orgs'])} train and organization {R['exp4']['test_org']} is held out; "
                f"{R['exp4']['rounds']} rounds, {R['exp4']['repeats']} repeats, plain FedAvg. Privacy budgets are shown as reported for each setting. "
                "Underlines show accuracy on a common 0–100% scale.")
    drop_o = M(e4["fedavg_nodp"]["accuracy"]) - M(e4["fedavg_dp"]["accuracy"])
    drop_h = M(e4h["fedavg_nodp"]["accuracy"]) - M(e4h["fedavg_dp"]["accuracy"])
    same_o = abs(M(e4["fedavg_dp"]["accuracy"]) - M(e4["fedavg_dp_reset_optimizer"]["accuracy"])) < 0.05
    same_h = abs(M(e4h["fedavg_dp"]["accuracy"]) - M(e4h["fedavg_dp_reset_optimizer"]["accuracy"])) < 0.05
    if drop_o > 0.1 and drop_h > 0.1 and same_o and same_h:
        t11_note += (" Client-side DP-SGD lowers held-out accuracy sharply, and resetting the optimizer each round does not explain it; "
                     "the cause has not been identified.")

    # ---------- methods ----------
    split = e1["split"]
    methods = (f"CICIDS2017 contains {ds['clean_rows']:,} cleaned flows; the benchmark uses a stratified sample of {ds['sample_rows']:,}, {nf}-fold CV × {ns} seeds. "
               f"Each training fold is split into a private set ({split['private_train_per_fold']:,} flows, protected by DP-SGD) and a public reference pool "
               f"({split['reference_pool_per_fold']:,} flows) that supplies graph neighbors; {split['test_per_fold']:,} flows test each fold. "
               f"{split['attacked_malicious_per_fold']:,} malicious test flows per fold are attacked; benign flows are unperturbed. The worst case counts a flow as detected only if it survives every attack. "
               "Ablation and privacy results are reported for seed 0. Synthetic and federated experiments retain their separately reported settings.")

    marks = {
        "HASH": E(h), "FPR": fpr_txt, "STUDY": study, "SUMMARY": summary, "BENCH_DESC": E(bench_desc),
        "T1": t1, "T1_NOTE": E(t1_note), "T2": t2, "T2_NOTE": t2_note, "T3": t3, "T4": t4, "T4_NOTE": E(t4_note),
        "T5": t5, "T5_NOTE": E(t5_note), "T6": t6, "T6_NOTE": E(t6_note), "T7": t7, "T7_NOTE": E(t7_note),
        "T8": t8, "T8_NOTE": t8_note, "TEMPORAL_DESC": E(temporal_desc), "T9": t9, "T9_NOTE": E(t9_note),
        "T10": t10, "T11": t11, "T11_NOTE": E(t11_note), "METHODS_PROTOCOL": E(methods),
    }
    with open(TEMPLATE, encoding="utf8") as f:
        page = f.read()
    for k in sorted(marks, key=len, reverse=True):
        page = page.replace(f"@@{k}@@", marks[k])
    assert "@@" not in page, "unfilled marker in template"
    with open(OUT, "w", encoding="utf8") as f:
        f.write(page)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
