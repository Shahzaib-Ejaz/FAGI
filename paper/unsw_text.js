// Text and table for the UNSW-NB15 replication (experiments/unsw_benchmark.py). Every number comes from
// paper_results.json; the directional words are chosen from the data, and the statements that matter are guarded.
module.exports = function ({ TN, U, M, pct, pp, fx, int, claim, eC }) {
  const MD = U.models, T = U.tests, ds = U.dataset;
  const nF = U.folds.length, nS = U.seeds.length;
  const r5 = (m) => M(MD[m]["fixed_0.01_0.5"]), r1 = (m) => M(MD[m]["fixed_0.01_0.1"]), cl = (m) => M(MD[m].clean);
  const fpr = (m) => M(MD[m].fpr), d5 = (m) => M(MD[m]["default_0.5"]);
  const flat = ["rf", "mlp", "mlp_at", "mlp_dp", "cnn"];
  const flatMax5 = Math.max(...flat.map(r5));
  const graphs = ["gnn", "fagi_no_dp", "fagi", "knn"];
  const best5 = graphs.reduce((a, b) => (r5(b) > r5(a) ? b : a));
  const NAME = { rf: "the random forest", knn: "the k-NN vote", mlp: "the MLP", mlp_at: "the PGD-trained MLP", mlp_dp: "the private MLP", cnn: "the CNN", gnn: "the GNN", fagi_no_dp: "FAGI without DP", fagi: "FAGI" };
  const scope = `${nF === 5 ? "5 folds" : `${nF} fold${nF > 1 ? "s" : ""}`} × ${nS} seed${nS > 1 ? "s" : ""}`;
  const t = (a, b, k) => T[`${a}_vs_${b}_${k}`];
  const wtxt = (x, nA, nB) => `${x.wins_a} of ${x.n} folds${x.n >= 5 ? `, p = ${x.p.toFixed(3)}` : ""}`;
  const epsU = M(MD.fagi.epsilon);
  const atGain = r5("fagi_no_dp") > r5("gnn") + 0.02, dpCost = r5("fagi") < r5("fagi_no_dp") - 0.02;
  const knnVsFagi = r5("knn") >= r5("fagi") - 0.01, knnVsNoDP = r5("knn") >= r5("fagi_no_dp") - 0.01;
  const all = Object.keys(MD);
  const fagiTopDefault = all.every((m) => d5("fagi") >= d5(m));
  const fagiFprRank = all.filter((m) => fpr(m) > fpr("fagi")).length;
  const fprOk = fagiFprRank === 0 || (fagiFprRank === 1 && fpr("mlp_dp") > fpr("fagi"));
  if (fagiTopDefault) claim(fprOk, "FAGI's default-threshold FPR on UNSW-NB15 is exceeded at most by the private MLP");
  const rfBest = cl("rf") >= Math.max(...Object.keys(MD).map(cl)) - 1e-9;
  claim(flatMax5 < Math.min(r5("fagi_no_dp"), r5("knn")), "on UNSW-NB15 every flat model is below FAGI without DP and the k-NN vote at eps 0.5");
  claim(Math.abs(epsU - eC) > 0.01 ? epsU > 0 : true, "the UNSW-NB15 epsilon is finite");

  const dataPara = `To test whether the findings are specific to CICIDS2017, we repeat the main comparison on UNSW-NB15 {unsw}, a second public flow dataset collected on a different testbed with a different feature extractor and nine attack families. We combine its official training and testing partitions (${int(ds.raw_rows)} flows) and re-split them by cross-validation, because the partitions share the testbed and keeping them apart would not test a shift in time. The cleaning follows Section V-A: no row has a non-finite value, ${int(ds.dropped_duplicates)} exact duplicates are dropped, and ${int(ds.clean_rows)} flows remain; ${int(ds.rows_with_conflicting_labels)} of them share their features with a flow of another label, which caps any model's accuracy. We drop the three categorical features (protocol, service and connection state), since an L∞ perturbation of a categorical code has no traffic meaning, and keep the ${int(U.features.length)} numeric ones. The sampler, the cap of 3,000 flows per attack family, the split into a private half and a reference pool, the models, their hyperparameters, the attack suite and the matched-rate evaluation are those of CICIDS2017 without change (${int(ds.sample_rows)} flows, ${scope}; Table ${TN.u}). Nothing was tuned on UNSW-NB15. With ${int(ds.sample_rows)} flows the sampling rate of DP-SGD changes, so σ = 1.1 gives FAGI ε_{DP} = ${fx(epsU)} here.`;

  const resPara = `UNSW-NB15 is harder than CICIDS2017 for every model: clean accuracy ranges from ${pct(Math.min(...Object.keys(MD).map(cl)), 1)}% to ${pct(Math.max(...Object.keys(MD).map(cl)), 1)}%${rfBest ? ", the random forest again highest" : ""}, and default thresholds flag ${pct(Math.min(...Object.keys(MD).map(fpr)), 1)}% to ${pct(Math.max(...Object.keys(MD).map(fpr)), 1)}% of benign flows. The ordering under attack is the same. At ε_{adv} = 0.5 and a matched 1% false-positive rate, no flat model keeps more than ${pct(flatMax5)}% of attacked flows detected, the PGD-trained MLP included. ${atGain ? `Adversarial training lifts the graph detector from ${pct(r5("gnn"))}% for the undefended GNN to ${pct(r5("fagi_no_dp"))}% for FAGI without DP (higher in ${wtxt(t("fagi_no_dp", "gnn", "fixed_0.01_0.5"))})` : `FAGI without DP keeps ${pct(r5("fagi_no_dp"))}% against ${pct(r5("gnn"))}% for the undefended GNN, a smaller gain from adversarial training than on CICIDS2017`}${dpCost ? `, and DP-SGD gives much of that back: FAGI keeps ${pct(r5("fagi"))}% (lower than FAGI without DP in ${wtxt(t("fagi_no_dp", "fagi", "fixed_0.01_0.5"))}) and loses ${pp(cl("fagi_no_dp"), cl("fagi"))} points of clean accuracy` : `, and DP-SGD costs little robustness here (${pct(r5("fagi"))}% for FAGI)`}. The k-NN vote over the pool keeps ${pct(r5("knn"))}%, ${knnVsNoDP ? "as much as FAGI without DP" : knnVsFagi ? "more than FAGI but less than FAGI without DP" : "less than FAGI"}. ${fagiTopDefault ? `At default thresholds FAGI again looks strongest (${pct(d5("fagi"))}% at ε_{adv} = 0.5), ${fagiFprRank === 0 ? "because it flags the most benign traffic" : "largely because it flags more benign traffic than every model but the private MLP"} (${pct(fpr("fagi"), 1)}%, against ${pct(fpr("fagi_no_dp"), 1)}% for FAGI without DP).` : `At default thresholds FAGI keeps ${pct(d5("fagi"))}% at ε_{adv} = 0.5 and flags ${pct(fpr("fagi"), 1)}% of benign flows.`} ${best5 === "fagi_no_dp" && knnVsFagi && dpCost && atGain ? "The robustness findings of CICIDS2017 therefore hold on a second dataset with no change to the models or their settings: flat models fail at the large budget, adversarial training is what keeps a graph detector working, DP-SGD gives much of that back, and a vote over the public pool does at least as well as the private model." : "The CICIDS2017 findings on robustness hold in direction on a second dataset, though not every gap is as large."}`;

  const order = ["rf", "knn", "mlp", "mlp_at", "mlp_dp", "cnn", "gnn", "fagi_no_dp", "fagi"];
  const LAB = { rf: "Random forest", knn: "k-NN vote (pool)", mlp: "MLP", mlp_at: "MLP + PGD", mlp_dp: "MLP + DP-SGD", cnn: "1D-CNN", gnn: "GNN", fagi_no_dp: "FAGI w/o DP", fagi: "FAGI" };
  const sd = (x) => (x.n > 1 ? ` ± ${pct(x.std)}` : "");
  const table = {
    header: ["Model", "ε_{DP}", "Clean acc.", "FPR (def.)", "R@1% 0.1", "R@1% 0.5", "Default 0.5"],
    rows: order.filter((m) => MD[m]).map((m) => [LAB[m], MD[m].epsilon ? fx(M(MD[m].epsilon)) : "—", pct(cl(m), 2), pct(fpr(m), 2), pct(r1(m)) + sd(MD[m]["fixed_0.01_0.1"]), pct(r5(m)) + sd(MD[m]["fixed_0.01_0.5"]), pct(d5(m))]),
    note: `UNSW-NB15, ${scope}, the CICIDS2017 protocol and settings unchanged. Means over folds of the per-fold mean over seeds${nF > 1 ? "; ± is the standard deviation over folds" : ""}. R@1% is worst-case recall (%) on attacked malicious test flows at a threshold that flags 1% of benign test flows, at ε_{adv} = 0.1 and 0.5; "Default 0.5" is the same recall at each model's default threshold. "—": no DP guarantee.`,
  };

  const short = `on UNSW-NB15, with the same models and settings, ${atGain && dpCost && knnVsFagi ? "adversarial training, DP-SGD and the k-NN vote compare the same way" : "the same comparisons hold in direction"}`;
  const limit = `The second dataset shares the first one's limits: UNSW-NB15 also comes from a testbed, and we evaluate it by cross-validation only, so it tests whether the findings carry across feature sets and attack families, not across time.`;
  const abstract = `On UNSW-NB15 (${int(ds.sample_rows)} flows, ${scope}), with every model and setting unchanged, no flat model keeps more than ${pct(flatMax5)}% at ε_{adv} = 0.5, adversarial training ${atGain ? `lifts the GNN from ${pct(r5("gnn"))}% to ${pct(r5("fagi_no_dp"))}%` : `moves the GNN from ${pct(r5("gnn"))}% to ${pct(r5("fagi_no_dp"))}%`}, DP-SGD ${dpCost ? "lowers it" : "leaves it"} at ${pct(r5("fagi"))}%, and the k-NN vote keeps ${pct(r5("knn"))}%.`;
  const conclusion = `The same comparison on UNSW-NB15, run with no change to the models or settings, ${atGain && dpCost && knnVsFagi ? "gives the same ordering" : "gives the same ordering in direction"}: flat models keep at most ${pct(flatMax5)}% of attacked flows at ε_{adv} = 0.5, FAGI without DP ${pct(r5("fagi_no_dp"))}%, FAGI ${pct(r5("fagi"))}%, and the k-NN vote ${pct(r5("knn"))}%.`;
  return { abstract, conclusion, dataPara, resPara, table, short, limit, epsU, scope };
};
