module.exports = function ({ F0, FH, M, pct, fx, claim }) {
  const rows = [["rf_central", "Random forest, central"], ["central_nodp", "Central, no DP"], ["central_dp", "Central, DP-SGD"],
    ["central_dp_eps_matched", "Central, DP, ε matched"], ["fedavg_nodp", "FedAvg, no DP"], ["fedavg_dp", "FedAvg, client DP-SGD"],
    ["fedavg_dp_reset_optimizer", "FedAvg, DP, Adam reset"], ["local_only_mean", "Local only, no DP"], ["local_dp_mean", "Local only, DP-SGD"]];
  return { rows: rows.filter(([k]) => F0[k] && FH[k]), abstract: "PLACEHOLDER", body: "PLACEHOLDER" };
};
