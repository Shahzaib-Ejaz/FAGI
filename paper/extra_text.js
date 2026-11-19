// Text and tables for the controls of Section VI-C (experiments/extra_controls.py) and the accounting table of
// Section VI-D. Every number comes from paper_results.json; directional words are chosen from the data and the
// statements that matter are guarded by claim().
module.exports = function ({ TN, XC, M, pct, pp, fx, int, claim, cfg, nP, nR, eC, eM, eS, SIG, CAN, TPS, TPB, F0, nSyn, nFedMembers, exp4, synCfg }) {
  const DC = XC.dp_controls.models, PO = XC.pool_only.models, P10 = XC.pool10.models, P10o = XC.pool10.pool_only;
  const poolFolds = PO.rf.per_fold.map((r) => r.fold);
  const onFolds = (m, key) => {
    const v = m.per_fold.filter((r) => poolFolds.includes(r.fold)).map((r) => r[key]);
    return v.reduce((a, b) => a + b, 0) / v.length;
  };
  const r5 = (m) => M(m["fixed_0.01_0.5"]), r1 = (m) => M(m["fixed_0.01_0.1"]), cl = (m) => M(m.clean);
  const wins = (a, b, key) => a.per_fold.filter((r, i) => r[key] > b.per_fold[i][key]).length;
  const nF = DC.fagi.per_fold.length, nPF = poolFolds.length;
  const nSd = (XC.seeds || [0]).length, sdTxt = nSd > 1 ? `${nSd} seeds` : "seed 0";
  const scopeTxt = nSd > 1 ? `${nSd} seeds, ${nPF === 5 && nF === 5 ? "5 folds" : `${nPF} to ${nF} folds`}` : "seed 0";
  const foldTxt = nPF === 5 ? "all five folds" : `folds ${poolFolds.join(", ").replace(/, (\d)$/, " and $1")}`;

  // ---- noise vs clipping vs configuration
  const clipShareClean = (cl(DC.fagi_no_dp) - cl(DC.fagi_clip_only)) / (cl(DC.fagi_no_dp) - cl(DC.fagi));
  const clipShareRob = (r5(DC.fagi_no_dp) - r5(DC.fagi_clip_only)) / (r5(DC.fagi_no_dp) - r5(DC.fagi));
  const tunedSigma = M(DC.fagi_dp_tuned.sigma), tunedEps = M(DC.fagi_dp_tuned.epsilon);
  const tunedBetter = r5(DC.fagi_dp_tuned) > r5(DC.fagi) + 0.02 && 2 * wins(DC.fagi_dp_tuned, DC.fagi, "fixed_0.01_0.5") > nF;
  const tdiff = DC.fagi.per_fold.map((r, i) => r["fixed_0.01_0.5"] - DC.fagi_dp_tuned.per_fold[i]["fixed_0.01_0.5"]);
  const oneFold = Math.max(...tdiff) > 0.5 * tdiff.reduce((a, b) => a + b, 0) && tdiff.reduce((a, b) => a + b, 0) > 0;
  const tunedWorse = r5(DC.fagi_dp_tuned) < r5(DC.fagi) - 0.02 && 2 * wins(DC.fagi_dp_tuned, DC.fagi, "fixed_0.01_0.5") < nF;
  claim(Math.abs(tunedEps - eC) < 0.01, "the tuned DP configuration has the same epsilon as FAGI");
  claim(clipShareRob < 0.5 === (r5(DC.fagi_clip_only) > (r5(DC.fagi) + r5(DC.fagi_no_dp)) / 2), "clip-only share of the robustness cost is consistent");
  const noiseMain = clipShareRob < 0.5;

  const dpPara = `To separate DP-SGD's noise from its other parts, we train FAGI with DP-SGD's per-example clipping and Poisson sampling but no noise (σ = 0, so no guarantee), and FAGI with a configuration chosen for private training instead of the non-private one: an expected batch of ${int(XC.tuned.batch_size)}, ${int(XC.tuned.epochs)} epochs, learning rate ${XC.tuned.lr}, and σ = ${fx(tunedSigma)}, which gives the same ε_{DP} = ${fx(tunedEps)} at the new sampling rate and step count (${sdTxt}, ${nF} folds; Table ${TN.xc}).${nSd > 1 ? "" : " All controls use seed 0, so FAGI's numbers here differ from the three-seed means of Tables I and III."} Without noise, clipping and sampling cost ${pp(cl(DC.fagi_no_dp), cl(DC.fagi_clip_only))} points of clean accuracy (${pct(cl(DC.fagi_clip_only), 2)}% against ${pct(cl(DC.fagi_no_dp), 2)}% without DP-SGD) and leave ${pct(r5(DC.fagi_clip_only))}% of attacked flows detected at ε_{adv} = 0.5 and a matched 1% false-positive rate, against ${pct(r5(DC.fagi_no_dp))}% without DP-SGD and ${pct(r5(DC.fagi))}% for FAGI. Clipping and sampling therefore account for ${pct(Math.max(0, clipShareClean), 0)}% of DP-SGD's cost in clean accuracy and ${pct(Math.max(0, clipShareRob), 0)}% of its cost in robustness${XC.dp_controls.tests["fagi_no_dp_vs_fagi_clip_only_fixed_0.01_0.5"].p >= 0.05 ? " (a gap to no DP that is within fold-to-fold variation)" : ""}; ${noiseMain ? "most of the robustness cost is the noise" : "much of the robustness cost comes from clipping, not noise"}. The configuration chosen for private training reaches ${pct(cl(DC.fagi_dp_tuned), 2)}% clean accuracy against ${pct(cl(DC.fagi), 2)}% for FAGI (higher in ${wins(DC.fagi_dp_tuned, DC.fagi, "clean")} of ${nF} folds) and keeps ${pct(r1(DC.fagi_dp_tuned))}% of attacked flows at ε_{adv} = 0.1 against ${pct(r1(DC.fagi))}%, but ${pct(r5(DC.fagi_dp_tuned))}% at 0.5 against ${pct(r5(DC.fagi))}% (ahead of FAGI in ${wins(DC.fagi_dp_tuned, DC.fagi, "fixed_0.01_0.5")} of ${nF} folds). ${tunedBetter ? "Part of what DP costs in our main configuration therefore comes from hyperparameters tuned without privacy, and the FAGI numbers elsewhere in this paper overstate the robustness cost of DP-SGD by about " + pp(r5(DC.fagi_dp_tuned), r5(DC.fagi)) + " points at ε_{adv} = 0.5. We keep the main configuration so that FAGI and its ablations share one training setup." : tunedWorse ? "Retuning for private training recovers some clean accuracy and some robustness at the small budget, but not at the large one, so we keep the main configuration." : "Retuning for private training therefore recovers some clean accuracy and some robustness at the small budget, but it does not change robustness at the large one, where the two are level fold by fold" + (oneFold ? " and the gap in means comes mostly from one fold" : "") + ". We keep the main configuration so that FAGI and its ablations share one training setup; retuning would gain " + pp(cl(DC.fagi_dp_tuned), cl(DC.fagi), 2) + " points of clean accuracy."}`;

  // ---- models with epsilon = 0 with respect to the private set
  const fagiF = { clean: onFolds(DC.fagi, "clean"), r1: onFolds(DC.fagi, "fixed_0.01_0.1"), r5: onFolds(DC.fagi, "fixed_0.01_0.5") };
  const bestPool = Math.max(r5(PO.fagi_no_dp), r5(PO.gnn));
  const poolBeatsHalf = cl(PO.rf) > fagiF.clean && bestPool >= fagiF.r5 - 0.01;
  const f10 = P10.fagi, best10 = Math.max(r5(P10o.fagi_no_dp), r5(P10o.gnn), r5(P10.knn));
  const poolBeats10 = best10 >= r5(f10) - 0.01 && Math.max(cl(P10o.rf), cl(P10.knn)) >= cl(f10);
  const eps10 = M(f10.epsilon);
  claim(cl(PO.rf) > fagiF.clean, "the pool-only random forest is more accurate than FAGI");

  const poolPara = `The pool is labeled and published, so a model trained on it alone uses no private record: its ε with respect to the private set is 0. The k-NN vote is one such model. We also train the random forest, the GNN, and FAGI without DP on the pool alone, excluding each pool flow from its own neighborhood (${sdTxt}, ${foldTxt}). The random forest reaches ${pct(cl(PO.rf), 2)}% clean accuracy and keeps ${pct(r1(PO.rf))}% of attacked flows at ε_{adv} = 0.1 and a matched 1% false-positive rate; FAGI without DP trained on the pool reaches ${pct(cl(PO.fagi_no_dp), 2)}% and keeps ${pct(r5(PO.fagi_no_dp))}% at ε_{adv} = 0.5, and the GNN ${pct(cl(PO.gnn), 2)}% and ${pct(r5(PO.gnn))}%. On the same folds FAGI, trained with DP-SGD on the private half, reaches ${pct(fagiF.clean, 2)}% and ${pct(fagiF.r5)}%. ${poolBeatsHalf ? "With a pool that is a random half of the training data, DP-SGD on the private half buys nothing over these models." : "With a pool that is a random half of the training data, the ε = 0 models do not match FAGI on every measure."} A smaller pool is the case where private training should pay. With 10% of each training fold as the pool and 90% private, FAGI trains on ${int(M(f10.n_train))} private flows (ε_{DP} = ${fx(eps10)}) and reaches ${pct(cl(f10), 2)}% clean accuracy and ${pct(r5(f10))}% matched recall at 0.5. Trained on the 10% pool alone, FAGI without DP reaches ${pct(cl(P10o.fagi_no_dp), 2)}% and ${pct(r5(P10o.fagi_no_dp))}%, the random forest ${pct(cl(P10o.rf), 2)}% clean, and the k-NN vote over that pool ${pct(cl(P10.knn), 2)}% and ${pct(r5(P10.knn))}%. ${poolBeats10 ? "Even with a pool a ninth the size of the private set, models that use no private data match or beat FAGI (the random forest in clean accuracy, FAGI without DP and the k-NN vote in matched recall), so in CICIDS2017 the private records add little that the public pool does not already provide." : "With the smaller pool, DP-SGD on the private set does buy something over the models that use no private data."}`;

  const foldsOf = (m) => { const f = m.per_fold.map((r) => r.fold); return f.every((x, i) => x === f[0] + i) ? `${f[0]}–${f[f.length - 1]}` : f.join(","); };
  const row = (setting, name, data, m, eps) => [setting, name, data, foldsOf(m), eps, pct(cl(m), 2), pct(M(m.fpr), 2), pct(r1(m)), pct(r5(m))];
  const sub = (m) => {
    const o = { per_fold: m.per_fold.filter((r) => poolFolds.includes(r.fold)) };
    for (const k of ["clean", "fpr", "fixed_0.01_0.1", "fixed_0.01_0.5"]) o[k] = { mean: onFolds(m, k) };
    return o;
  };
  const table = {
    header: ["Setting", "Model", "Trained on", "Folds", "ε_{DP}", "Clean acc.", "FPR (def.)", "R@1% 0.1", "R@1% 0.5"],
    rows: [
      row("50% pool", "FAGI", "private half", DC.fagi, fx(eC)),
      ...(nPF < nF ? [row("50% pool", "FAGI", "private half", sub(DC.fagi), fx(eC))] : []),
      row("50% pool", "FAGI w/o DP", "private half", DC.fagi_no_dp, "—"),
      row("50% pool", "FAGI, clip only (σ = 0)", "private half", DC.fagi_clip_only, "—"),
      row("50% pool", "FAGI, DP-tuned", "private half", DC.fagi_dp_tuned, fx(tunedEps)),
      row("50% pool", "Random forest", "pool only", PO.rf, "0"),
      row("50% pool", "GNN", "pool only", PO.gnn, "0"),
      row("50% pool", "FAGI w/o DP", "pool only", PO.fagi_no_dp, "0"),
      row("10% pool", "FAGI", "private 90%", f10, fx(eps10)),
      row("10% pool", "FAGI w/o DP", "private 90%", P10.fagi_no_dp, "—"),
      row("10% pool", "k-NN vote", "pool only", P10.knn, "0"),
      row("10% pool", "Random forest", "pool only", P10o.rf, "0"),
      row("10% pool", "FAGI w/o DP", "pool only", P10o.fagi_no_dp, "0"),
    ],
    note: `${nSd > 1 ? `Means over the folds listed of the per-fold mean over ${nSd} seeds` : "Seed 0; means over the folds listed"}, standard deviations omitted.${nPF < nF ? ` FAGI is shown on all five folds and on ${foldTxt}, the folds of the pool-only rows.` : ""}${nSd > 1 ? "" : " Its seed-0 numbers differ from the three-seed means of Tables I and III."} Clean accuracy and FPR are at the default threshold; R@1% 0.1 and 0.5: worst-case attack recall at a matched 1% false-positive rate under the full attack suite, at ε_{adv} = 0.1 and 0.5. ε_{DP} = 0: the model uses no private record. "—": no guarantee (no DP-SGD, or clipping with σ = 0).`,
  };

  // ---- accounting table
  const acct = (name, n, batch, epochs, sigma, eps, steps) => {
    const q = batch / n;
    const T = steps || Math.ceil(epochs / q);
    return [name, int(n), int(batch), fx(q, 4), int(T), fx(sigma, 2), fx(eps)];
  };
  const fedMembersPerOrg = nFedMembers / 4;
  const qFed = synCfg.batch_size / fedMembersPerOrg;
  const sigOf = (s) => SIG.find((x) => Math.abs(x.sigma - s) < 1e-9);
  const accounting = {
    header: ["Model", "|D|", "Expected batch", "q", "Steps T", "σ", "ε_{DP}"],
    rows: [
      acct("FAGI, CICIDS2017", nP, cfg.batch_size, cfg.epochs, cfg.dp_sigma, eC),
      ...[0.6, 2.0, 4.0].map((s) => acct(`FAGI, σ sweep`, nP, cfg.batch_size, cfg.epochs, s, M(sigOf(s).epsilon))),
      acct("FAGI, DP-tuned configuration", nP, XC.tuned.batch_size, XC.tuned.epochs, tunedSigma, tunedEps),
      acct("FAGI, 10% pool", M(f10.n_train), cfg.batch_size, cfg.epochs, cfg.dp_sigma, eps10),
      acct("FAGI, canary audit", nP - 100, cfg.batch_size, cfg.epochs, cfg.dp_sigma, CAN.fagi.epsilon),
      acct("Private MLP", nP + nR, cfg.batch_size, cfg.epochs, cfg.dp_sigma, eM),
      acct("FAGI, file-order split", TPS.private_train, cfg.batch_size, cfg.epochs, cfg.dp_sigma, M(TPB.fagi.epsilon)),
      acct("FAGI, API sessions (central)", nSyn, synCfg.batch_size, synCfg.epochs, synCfg.dp_sigma, eS),
      acct("Central DP, federated data", nFedMembers, synCfg.batch_size, synCfg.epochs, synCfg.dp_sigma, M(F0.central_dp.epsilon)),
      acct("Central DP, ε matched", nFedMembers, synCfg.batch_size, synCfg.epochs, M(F0.central_dp_eps_matched.sigma), M(F0.central_dp_eps_matched.epsilon)),
      acct("FedAvg client (all rounds)", fedMembersPerOrg, synCfg.batch_size, null, synCfg.dp_sigma, M(F0.fedavg_dp.epsilon), exp4.rounds * Math.ceil(exp4.local_epochs / qFed)),
      acct("Local DP (one organization)", fedMembersPerOrg, synCfg.batch_size, synCfg.epochs, synCfg.dp_sigma, M(F0.local_dp_mean.epsilon)),
    ],
    note: `Poisson sampling with rate q = expected batch / |D|; T = ⌈epochs / q⌉ (for FedAvg, rounds × ⌈local epochs / q⌉). Clipping norm C = ${fx(cfg.dp_clip, 1)}, δ = 10^{-6}, RDP accountant (Opacus, orders 1.1–1024) with the conversion of Balle et al. The API-session models use epochs = ${int(synCfg.epochs)}.`,
  };

  return {
    scopeTxt,
    dpPara, poolPara, table, accounting,
    short: poolBeats10 && poolBeatsHalf
      ? "models trained on the pool alone match or beat FAGI even when the pool is a ninth the size of the private set"
      : poolBeatsHalf ? "models trained on the pool alone match FAGI when the pool is half the training data, but not when it is small"
        : "models trained on the pool alone come close to FAGI",
    abstract: poolBeats10 && poolBeatsHalf
      ? `Models trained only on the public reference pool, which use no private record, match or beat FAGI${nSd > 1 && nPF === 5 ? "" : ` (${scopeTxt}${nPF < 5 ? `, ${nPF} folds` : ""})`}: a random forest in clean accuracy, and FAGI without DP or the k-NN vote in matched recall at ε_{adv} = 0.5, even when the pool is a ninth the size of the private set. `
      : poolBeatsHalf
        ? "Models trained only on the public reference pool, which use no private record, match or beat FAGI when the pool is half the training data but not when it is small. "
        : "Models trained only on the public reference pool, which use no private record, come close to FAGI but do not match it on every measure. ",
    discussion: poolBeats10 && poolBeatsHalf
      ? `Models trained on the pool alone do as well${nSd > 1 ? "" : " on seed 0"}, even with a small pool, so in this setting the private records add little that the public pool does not provide.`
      : poolBeatsHalf
        ? "Models trained on the pool alone do as well when the pool is large, but not when it is small, which is where DP-SGD on private data pays."
        : "Models trained on the pool alone come close but do not match FAGI on every measure, so the private records do add something.",
    conclusion: poolBeats10 && poolBeatsHalf
      ? `but the pool itself is published unprotected, and in our setting models trained on the pool alone, with no private data, match or beat the private model${nSd > 1 ? "" : " on seed 0"} even when the pool is small.`
      : poolBeatsHalf
        ? "but the pool itself is published unprotected, and models trained on the pool alone match the private model when the pool is large; DP-SGD pays only when the pool is small."
        : "but the pool itself is published unprotected, and models trained on the pool alone come close to the private model.",
    noiseMain,
  };
};
