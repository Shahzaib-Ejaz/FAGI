module.exports = function ({ TP, TPB, B, AB, M, pct, int, claim, w }) {
  const keys = TP.models.map((m) => m.key);
  const c = (k) => M(TPB[k].clean);
  const r = (k, e) => M(TPB[k][`worst_recall_eps${e}`]);
  const best = keys.reduce((x, k) => (c(k) > c(x) ? k : x), keys[0]);
  const max05 = Math.max(...keys.map((k) => r(k, "0.5")));
  const lc = TP.label_counts;
  const absent = Object.entries(lc).filter(([l, v]) => l !== "BENIGN" && v.test === 0).map(([l]) => l);
  const bot = (k) => TPB[k].per_class_seed0.Bot.recall;
  const graphLike = ["knn", "gnn", "fagi_no_dp", "fagi"];
  const flatLow = ["rf", "mlp"];

  claim(keys.every((k) => c(k) < M((B[k] || AB[k]).clean) - 0.04), "every model loses more than 4 points of clean accuracy on the temporal split");
  claim(best === "mlp", "the plain MLP has the best clean accuracy on the temporal split");
  claim(c("fagi") === Math.min(...keys.map(c)), "FAGI has the lowest clean accuracy on the temporal split");
  claim(max05 < 0.1, "every model detects under 10% at eps_adv=0.5 on the temporal split");
  claim(graphLike.every((k) => r(k, "0.1") > 0.7) && flatLow.every((k) => r(k, "0.1") < 0.4), "neighborhood models keep >70% at 0.1, RF and MLP <40%");
  claim(keys.every((k) => bot(k) < 0.05), "Bot recall under 5% for every model on the temporal split");
  const drop = (k) => M((B[k] || AB[k]).clean) - c(k);
  claim(keys.every((k) => k === "fagi" || drop(k) < drop("fagi")), "FAGI loses the most clean accuracy on the temporal split");
  claim(c("fagi_no_dp") - c("fagi") > M(AB.fagi_no_dp.clean) - M(AB.fagi.clean), "DP costs more on the temporal split than under CV");
  claim(TP.labels_only_in_test.length === 2, "two labels appear only in the temporal test set");

  const intro = `Random cross-validation lets a model see flows from the same attack burst it is tested on {arp}. As a stricter check, we keep each day file in its original row order, train on the earlier 70% of rows and test on the later 30%, and keep everything else unchanged: the training rows are split in half into private set and pool (${int(TP.split.private_train)} flows each), and ${int(TP.split.attacked_malicious)} malicious test flows are attacked. The split is harder in ways that matter. ${TP.labels_only_in_test.join(" and ")} occur only in the test set; ${absent.length} attack labels (${absent.join(", ")}) occur only in training; and DoS GoldenEye has ${int(lc["DoS GoldenEye"].train)} training flows against ${int(lc["DoS GoldenEye"].test)} test flows. Table V reports one split with ${TPB.fagi.clean.n} seeds.`;

  const body = `Every model loses at least four points of clean accuracy. The plain MLP is now the most accurate (${pct(c("mlp"), 2)}%), ahead of the random forest (${pct(c("rf"), 2)}%), and the k-NN vote falls to ${pct(c("knn"), 2)}%, which supports the reading of its cross-validation result as partly a duplicate effect. FAGI is the least accurate model at ${pct(c("fagi"), 2)}%, with a ${pct(M(TPB.fagi.fpr), 2)}% false-positive rate: DP's cost grows when the test traffic differs from the training traffic. Detection rates of ${pct(Math.min(...keys.map((k) => M(TPB[k].detection_rate))), 0)}–${pct(Math.max(...keys.map((k) => M(TPB[k].detection_rate))), 0)}% hide a common failure, since every model detects fewer than 5% of the later Bot flows (FAGI ${pct(bot("fagi"))}%). Under attack the pattern of Section V-C survives at ε_{adv} = 0.1, where the neighborhood models keep ${pct(Math.min(...graphLike.map((k) => r(k, "0.1"))))}–${pct(Math.max(...graphLike.map((k) => r(k, "0.1"))))}% and the random forest and MLP keep ${pct(r("rf", "0.1"))}% and ${pct(r("mlp", "0.1"))}%. It does not survive at ε_{adv} = 0.5, where no model detects more than ${pct(max05)}% (FAGI ${pct(r("fagi", "0.5"))}%, k-NN ${pct(r("knn", "0.5"))}%, GNN ${pct(r("gnn", "0.5"))}%). FAGI's large-budget robustness in Table I depends on test flows that resemble training flows, and it does not carry over to later traffic.`;

  return {
    intro, body,
    abstract: `On a temporal split that tests on later traffic, every model loses at least four points of clean accuracy, FAGI the most, and no model detects more than ${Math.ceil(100 * max05)}% of attacked flows at ε_{adv} = 0.5.`,
    discussion: `The temporal split qualifies all of this: on later traffic, clean accuracy drops for every model, DP costs more, and no model is robust at ε_{adv} = 0.5.`,
  };
};
