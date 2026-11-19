module.exports = function ({ TP, TPB, B, AB, M, pct, pp, int, claim, w, FXT, FXCV, AC }) {
  const cv05 = (k) => M(FXCV[k]["fixed_0.01_0.5"]);
  const mt = (k, e) => M(FXT[k][`fixed_0.01_${e}`]);
  const mtKeys = Object.keys(FXT);
  const poolT = mtKeys.filter((k) => k !== "mlp_dp");
  const keys = TP.models.map((m) => m.key);
  const c = (k) => M(TPB[k].clean);
  const r = (k, e) => M(TPB[k][`worst_recall_eps${e}`]);
  const best = keys.reduce((x, k) => (c(k) > c(x) ? k : x), keys[0]);
  const max05 = Math.max(...keys.map((k) => r(k, "0.5")));
  const lc = TP.label_counts;
  const allBotSame = keys.every((k) => Math.abs(TPB[k].per_class_seed0.Bot.recall - TPB.fagi.per_class_seed0.Bot.recall) < 1e-9);
  const absent = Object.entries(lc).filter(([l, v]) => l !== "BENIGN" && v.test === 0).map(([l]) => l);
  const bot = (k) => TPB[k].per_class_seed0.Bot.recall;
  const graphLike = ["knn", "gnn", "fagi_no_dp", "fagi", "mlp_at"];
  const flatLow = ["rf", "mlp"];

  const drop0 = (k) => M((B[k] || AB[k]).clean) - c(k);
  const minDrop = Math.min(...keys.map(drop0));
  const atLeast = Math.floor(100 * minDrop);
  claim(atLeast >= 5, "every model loses at least five points of clean accuracy on the temporal split");
  claim(best === "mlp", "the plain MLP has the best clean accuracy on the temporal split");
  claim(c("fagi") === Math.min(...keys.map(c)), "FAGI has the lowest clean accuracy on the file-order split");
  claim(max05 < 0.1, "every model detects under 10% at eps_adv=0.5 on the temporal split");
  claim(["gnn", "fagi_no_dp"].every((k) => mt(k, "0.1") > 0.7) && mt("fagi", "0.1") < Math.min(mt("gnn", "0.1"), mt("fagi_no_dp", "0.1"), mt("knn", "0.1")) - 0.05, "temporal, matched FPR: GNN and FAGI-DP keep >70% at 0.1, FAGI clearly below");
  claim(mtKeys.every((k) => mt(k, "0.5") < 0.06), "temporal, matched FPR: every model under 6% at 0.5");
  claim(poolT.every((k) => M(FXT[k].pool_realized_test_fpr) > 0.015), "temporal: pool-calibrated thresholds overshoot 1.5% FPR on later traffic for the neural models");
  claim(keys.every((k) => bot(k) < 0.05), "Bot recall under 5% for every model on the temporal split");
  const drop = (k) => M((B[k] || AB[k]).clean) - c(k);
  claim(keys.every((k) => k === "fagi" || drop(k) < drop("fagi")), "FAGI loses the most clean accuracy on the temporal split");
  claim(c("fagi_no_dp") - c("fagi") > M(AB.fagi_no_dp.clean) - M(AB.fagi.clean), "DP costs more on the temporal split than under CV");
  claim(TP.labels_only_in_test.length === 2, "two labels appear only in the temporal test set");

  const intro = `Random cross-validation lets a model see flows from the same attack burst it is tested on {arp}. As a stricter check, we keep the stratified sample's flows from each of the eight day files in their original row order, train on the first 70% of them and test on the last 30%, and keep everything else unchanged. MachineLearningCSV carries no timestamps, and row order is only roughly chronological: in the Wednesday file, about half of the DoS GoldenEye rows come after the Heartbleed rows, although the Heartbleed attack happened hours after GoldenEye ended. This is therefore a file-order split, which keeps attack bursts largely apart, not a strict split by time. The training rows are split in half into private set and pool (${int(TP.split.private_train)} flows each), and ${int(TP.split.attacked_malicious)} malicious test flows are attacked. The split is harder in ways that matter. ${TP.labels_only_in_test.join(" and ").replace("Sql", "SQL")} occur only in the test set; ${absent.length} attack labels (${absent.join(", ").replace("Sql", "SQL")}) occur only in training; and DoS GoldenEye has ${int(lc["DoS GoldenEye"].train)} training flows against ${int(lc["DoS GoldenEye"].test)} test flows. Table VI reports one split with ${TPB.fagi.clean.n} seeds.`;

  const body = `Every model loses at least ${atLeast} points of clean accuracy. The plain MLP is now the most accurate (${pct(c("mlp"), 2)}%), ahead of the random forest (${pct(c("rf"), 2)}%), and the k-NN vote falls to ${pct(c("knn"), 2)}%, which supports the reading of its cross-validation result as partly a near-duplicate effect. FAGI is the least accurate model at ${pct(c("fagi"), 2)}%, with a ${pct(M(TPB.fagi.fpr), 2)}% false-positive rate. Its gap to FAGI without DP grows from ${pp(M(AB.fagi_no_dp.clean), M(AB.fagi.clean))} points under cross-validation to ${pp(c("fagi_no_dp"), c("fagi"))} here, although the private MLP is ${c("mlp_dp") > c("mlp_at") ? "slightly more" : "less"} accurate than the non-private PGD-trained MLP on this split (${pct(c("mlp_dp"), 2)}% against ${pct(c("mlp_at"), 2)}%). Detection rates of ${pct(Math.min(...keys.map((k) => M(TPB[k].detection_rate))))}–${pct(Math.max(...keys.map((k) => M(TPB[k].detection_rate))))}% hide a common failure: on seed 0, every model detects ${allBotSame ? `exactly the same ${pct(bot("fagi"))}%` : `fewer than 5%`} of the Bot flows in the test portion. At a matched 1% false-positive rate and ε_{adv} = 0.1, the GNN and FAGI without DP keep ${pct(mt("gnn", "0.1"))}% and ${pct(mt("fagi_no_dp", "0.1"))}%, the PGD-trained MLP ${pct(mt("mlp_at", "0.1"))}%, the k-NN vote ${pct(mt("knn", "0.1"))}%, FAGI ${pct(mt("fagi", "0.1"))}%, and the random forest and undefended MLP ${pct(mt("rf", "0.1"))}% and ${pct(mt("mlp", "0.1"))}%. At ε_{adv} = 0.5 no model detects more than ${pct(Math.max(...mtKeys.map((k) => mt(k, "0.5"))))}% (at the default threshold, ${pct(max05)}%). Thresholds calibrated on the pool, which comes from the first 70% of each file, do not carry over either: on the last 30% they flag ${pct(Math.min(...poolT.map((k) => M(FXT[k].pool_realized_test_fpr))))}–${pct(Math.max(...poolT.map((k) => M(FXT[k].pool_realized_test_fpr))))}% of benign flows instead of 1% (the private MLP, which trains on its pool, is left out), and FAGI's default threshold flags ${pct(M(TPB.fagi.fpr), 2)}%, too many for deployment. At ε_{adv} = 0.5 and a matched 1% false-positive rate, FAGI without DP falls from ${pct(cv05('fagi_no_dp'))}% under cross-validation to ${pct(mt('fagi_no_dp', '0.5'))}% on the held-out end of each file, and FAGI from ${pct(cv05('fagi'))}% to ${pct(mt('fagi', '0.5'))}%. ${AC && AC.restricted_to_file_order_test_labels ? `Part of this drop is which attacks are tested: the three DoS types that carry most of the cross-validation robustness do not occur in the test portion of this split. Restricted to the labels that do occur, cross-validation recall is already ${pct(M(AC.restricted_to_file_order_test_labels.fagi_no_dp))}% for FAGI without DP and ${pct(M(AC.restricted_to_file_order_test_labels.fagi))}% for FAGI, so the shift itself accounts for the rest of the drop.` : ''} With one split and three seeds, Table VI is descriptive; we do not rank models that differ by a point or two.`;

  return {
    intro, body,
    short: `On a file-order split that tests on the last rows of each day file, no model detects more than ${Math.ceil(100 * max05)}% of attacked flows at ε_{adv} = 0.5.`,
    abstract: `On a file-order split that tests on the last rows of each day file, every model we run on it loses at least ${atLeast} points of clean accuracy, FAGI ends as the least accurate model (${pct(c("fagi"), 2)}%, with a ${pct(M(TPB.fagi.fpr), 2)}% false-positive rate), and no model detects more than ${Math.ceil(100 * max05)}% of attacked flows at ε_{adv} = 0.5.`,
    discussion: `The file-order split qualifies all of this: on the held-out end of each day file, clean accuracy drops for every model, DP costs FAGI more, and no model is robust at ε_{adv} = 0.5.`,
  };
};
