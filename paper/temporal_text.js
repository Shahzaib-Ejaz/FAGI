module.exports = function ({ TP, TPB, B, AB, M, pct, int, claim, w, FXT }) {
  const mt = (k, e) => M(FXT[k][`fixed_0.01_${e}`]);
  const mtKeys = Object.keys(FXT);
  const keys = TP.models.map((m) => m.key);
  const c = (k) => M(TPB[k].clean);
  const r = (k, e) => M(TPB[k][`worst_recall_eps${e}`]);
  const best = keys.reduce((x, k) => (c(k) > c(x) ? k : x), keys[0]);
  const max05 = Math.max(...keys.map((k) => r(k, "0.5")));
  const lc = TP.label_counts;
  const absent = Object.entries(lc).filter(([l, v]) => l !== "BENIGN" && v.test === 0).map(([l]) => l);
  const bot = (k) => TPB[k].per_class_seed0.Bot.recall;
  const graphLike = ["knn", "gnn", "fagi_no_dp", "fagi", "mlp_at"];
  const flatLow = ["rf", "mlp"];

  const drop0 = (k) => M((B[k] || AB[k]).clean) - c(k);
  const minDrop = Math.min(...keys.map(drop0));
  const atLeast = Math.floor(100 * minDrop);
  claim(atLeast >= 5, "every model loses at least five points of clean accuracy on the temporal split");
  claim(best === "mlp", "the plain MLP has the best clean accuracy on the temporal split");
  claim(c("fagi") === Math.min(...keys.map(c)), "FAGI has the lowest clean accuracy on the temporal split");
  claim(max05 < 0.1, "every model detects under 10% at eps_adv=0.5 on the temporal split");
  claim(["gnn", "fagi_no_dp"].every((k) => mt(k, "0.1") > 0.7) && mt("fagi", "0.1") < Math.min(mt("gnn", "0.1"), mt("fagi_no_dp", "0.1"), mt("knn", "0.1")) - 0.05, "temporal, matched FPR: GNN and FAGI-DP keep >70% at 0.1, FAGI clearly below");
  claim(mtKeys.every((k) => mt(k, "0.5") < 0.06), "temporal, matched FPR: every model under 6% at 0.5");
  claim(["mlp", "mlp_at", "gnn", "fagi_no_dp", "fagi"].every((k) => M(FXT[k].pool_realized_test_fpr) > 0.015), "temporal: pool-calibrated thresholds overshoot 1.5% FPR on later traffic for the neural models");
  claim(keys.every((k) => bot(k) < 0.05), "Bot recall under 5% for every model on the temporal split");
  const drop = (k) => M((B[k] || AB[k]).clean) - c(k);
  claim(keys.every((k) => k === "fagi" || drop(k) < drop("fagi")), "FAGI loses the most clean accuracy on the temporal split");
  claim(c("fagi_no_dp") - c("fagi") > M(AB.fagi_no_dp.clean) - M(AB.fagi.clean), "DP costs more on the temporal split than under CV");
  claim(TP.labels_only_in_test.length === 2, "two labels appear only in the temporal test set");

  const intro = `Random cross-validation lets a model see flows from the same attack burst it is tested on {arp}. As a stricter check, we keep the stratified sample's flows from each day file in their original row order, train on the earlier 70% of them and test on the later 30%, and keep everything else unchanged. MachineLearningCSV carries no timestamps, so this assumes that row order follows flow order within each file, which we could not verify against the timestamped release. The training rows are split in half into private set and pool (${int(TP.split.private_train)} flows each), and ${int(TP.split.attacked_malicious)} malicious test flows are attacked. The split is harder in ways that matter. ${TP.labels_only_in_test.join(" and ").replace("Sql", "SQL")} occur only in the test set; ${absent.length} attack labels (${absent.join(", ").replace("Sql", "SQL")}) occur only in training; and DoS GoldenEye has ${int(lc["DoS GoldenEye"].train)} training flows against ${int(lc["DoS GoldenEye"].test)} test flows. Table VI reports one split with ${TPB.fagi.clean.n} seeds.`;

  const body = `Every model loses at least ${atLeast} points of clean accuracy. The plain MLP is now the most accurate (${pct(c("mlp"), 2)}%), ahead of the random forest (${pct(c("rf"), 2)}%), and the k-NN vote falls to ${pct(c("knn"), 2)}%, which supports the reading of its cross-validation result as partly a duplicate effect. FAGI is the least accurate model at ${pct(c("fagi"), 2)}%, with a ${pct(M(TPB.fagi.fpr), 2)}% false-positive rate: DP's cost grows when the test traffic differs from the training traffic. Detection rates of ${pct(Math.min(...keys.map((k) => M(TPB[k].detection_rate))))}–${pct(Math.max(...keys.map((k) => M(TPB[k].detection_rate))))}% hide a common failure, since every model detects fewer than 5% of the later Bot flows (FAGI ${pct(bot("fagi"))}%). At a matched 1% false-positive rate and ε_{adv} = 0.1, the GNN and FAGI without DP keep ${pct(mt("gnn", "0.1"))}% and ${pct(mt("fagi_no_dp", "0.1"))}%, the PGD-trained MLP ${pct(mt("mlp_at", "0.1"))}%, the k-NN vote ${pct(mt("knn", "0.1"))}%, FAGI ${pct(mt("fagi", "0.1"))}%, and the random forest and undefended MLP ${pct(mt("rf", "0.1"))}% and ${pct(mt("mlp", "0.1"))}%. At ε_{adv} = 0.5 no model detects more than ${pct(Math.max(...mtKeys.map((k) => mt(k, "0.5"))))}% (at the default threshold, ${pct(max05)}%). Thresholds calibrated on the training period's pool do not carry over either: on the later traffic they flag ${pct(Math.min(...["mlp", "mlp_at", "gnn", "fagi_no_dp", "fagi"].map((k) => M(FXT[k].pool_realized_test_fpr))))}–${pct(Math.max(...["mlp", "mlp_at", "gnn", "fagi_no_dp", "fagi"].map((k) => M(FXT[k].pool_realized_test_fpr))))}% of benign flows instead of 1%, and FAGI's default threshold flags ${pct(M(TPB.fagi.fpr), 2)}%, too many for deployment. FAGI's large-budget robustness in Table I depends on test flows that resemble training flows, and it does not carry over to later traffic.`;

  return {
    intro, body,
    abstract: `On a temporal split that tests on later traffic, every model loses at least ${atLeast} points of clean accuracy, FAGI the most, and no model detects more than ${Math.ceil(100 * max05)}% of attacked flows at ε_{adv} = 0.5.`,
    discussion: `The temporal split qualifies all of this: on later traffic, clean accuracy drops for every model, DP costs more, and no model is robust at ε_{adv} = 0.5.`,
  };
};
