module.exports = function ({ F0, FH, M, pct, fx, claim }) {
  const rows = [["rf_central", "Random forest, central"], ["central_nodp", "Central, no DP"], ["central_dp", "Central, DP-SGD"],
    ["central_dp_eps_matched", "Central, DP, ε matched"], ["fedavg_nodp", "FedAvg, no DP"], ["fedavg_dp", "FedAvg, client DP-SGD"],
    ["fedavg_dp_reset_optimizer", "FedAvg, DP, Adam reset"], ["local_only_mean", "Local only, no DP"], ["local_dp_mean", "Local only, DP-SGD"]];
  const acc = (F, k) => M(F[k].accuracy);
  const both = (f) => [F0, FH].every(f);
  const mia = ["central_nodp", "central_dp", "fedavg_nodp", "fedavg_dp"];

  claim(both((F) => acc(F, "fedavg_nodp") > acc(F, "local_only_mean")), "FedAvg without DP beats local training on both generators");
  claim(acc(F0, "fedavg_dp") >= acc(F0, "local_dp_mean") && acc(F0, "fedavg_dp") - acc(F0, "local_dp_mean") < 0.05, "original: FedAvg-DP level with or slightly above local DP");
  claim(acc(FH, "fedavg_dp") < acc(FH, "local_dp_mean") - 0.03, "hard: FedAvg-DP below local DP");
  claim(both((F) => Math.abs(acc(F, "fedavg_dp") - acc(F, "fedavg_dp_reset_optimizer")) < 0.03), "persistent vs reset Adam within 3 points on both generators");
  claim(both((F) => acc(F, "central_dp_eps_matched") > acc(F, "fedavg_dp") + 0.1), "centralized DP at the same epsilon beats FedAvg-DP by >10 points");
  claim(both((F) => mia.every((k) => Math.abs(M(F[k].mia_auc) - 0.5) < 0.05)), "federated MIA near chance on both generators");
  claim(both((F) => ["fedavg_nodp", "fedavg_dp"].every((k) => Math.abs(M(F[k].client_model_mia_auc_mean) - 0.5) < 0.05)), "client-model MIA near chance");

  const lo = (k) => `${pct(acc(F0, k))}% and ${pct(acc(FH, k))}%`;
  const fpH = FH.fedavg_dp.fpr ? M(FH.fedavg_dp.fpr) : null;
  const ptH = FH.fedavg_dp.per_type_detection || {};
  const attackTypes = Object.keys(ptH).filter((t) => t !== "none");
  const meanDet = attackTypes.length ? attackTypes.reduce((s, t) => s + ptH[t], 0) / attackTypes.length : null;

  const body = `Without DP, federated averaging reaches ${lo("fedavg_nodp")} on the original and hard generators, above the ${lo("local_only_mean")} an organization reaches alone, so federation does help an organization whose data it never saw. With client DP-SGD the picture splits by generator. On the original data FedAvg reaches ${pct(acc(F0, "fedavg_dp"))}% against ${pct(acc(F0, "local_dp_mean"))}% for private local training, about level; on the hard data it reaches ${pct(acc(FH, "fedavg_dp"))}% against ${pct(acc(FH, "local_dp_mean"))}%${meanDet !== null ? `, and it detects on average ${pct(meanDet)}% of attack sessions per type with a ${pct(fpH)}% false-positive rate` : ""}. Keeping each client's Adam state across rounds, which costs no privacy because Adam only post-processes noisy gradients, changes little: resetting it every round gives ${lo("fedavg_dp_reset_optimizer")}. The optimizer reset is therefore not the cause, but we have not identified what is. Centralized training at the same ε_{DP} reaches ${lo("central_dp_eps_matched")}, and on the hard data the federated model collapses to calling almost every session benign. Averaging four clients' noisy updates should, on its own, give each global step about the noise of a larger central batch, so noise level alone is an unlikely explanation; client drift and noisy Adam steps at our client learning rate are candidates we did not separate (by rounds, local epochs, client optimizer, or learning rate). Membership inference on the final global and local models is near chance in every setting, and the overfit control reaches AUC ${fx(M(F0.overfit_control.mia_auc), 3)} and ${fx(M(FH.overfit_control.mia_auc), 3)}.`;

  return {
    rows: rows.filter(([k]) => F0[k] && FH[k]),
    abstract: `On synthetic API sessions, federated averaging helps an unseen organization without DP; with client DP-SGD it is level with private local training on one generator and below it on the harder one, and centralized training at the same ε_{DP} is far better than both.`,
    body,
  };
};
