const fs = require("fs");
const path = require("path");
const L = require("./lib");
const REFS = require("./refs");

const REPO = process.env.FAGI_REPO || path.resolve(__dirname, "..");
const R = JSON.parse(fs.readFileSync(path.join(REPO, "experiments/results/paper_results.json"), "utf8"));
const OUT = process.argv[2] || path.join(__dirname, "out", "FAGI.docx");
fs.mkdirSync(path.dirname(OUT), { recursive: true });

const pct = (x, d = 1) => (100 * x).toFixed(d);
const pm = (o, d = 1) => `${(100 * o.mean).toFixed(d)} ± ${(100 * o.std).toFixed(d)}`;
const M = (o) => o.mean;
const pp = (a, b, d = 1) => (100 * (a - b)).toFixed(d);
const int = (n) => Number(n).toLocaleString("en-US");
const fx = (x, d = 2) => Number(x).toFixed(d);
const pv = (p) => (p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`);
const checks = [];
const claim = (cond, msg) => { checks.push(msg); if (!cond) throw new Error("claim no longer holds: " + msg); };

// ---------- data ----------
const E1 = R.exp1, ds = E1.dataset, cfg = E1.config, split = E1.split;
const B = Object.fromEntries(E1.models.map((m) => [m.key, m]));
const T1 = E1.tests;
const X3 = R.exp3.cicids2017, SW = X3.eps_sweep_worst_recall, BA = X3.by_attack_recall, T3 = X3.tests;
const K100 = X3.adaptive_k100_vs_k20_seed0;
const GS = R.exp3.graph_structure_attack;
const AB = Object.fromEntries(R.ablation.cicids2017.rows_seed0.map((r) => [r.key, r]));
const SIG = R.ablation.cicids2017.sigma_sweep_seed0;
const MIA = R.ablation.cicids2017.mia_seed0;
const TA = R.ablation.cicids2017.tests_seed0;
const CAN = R.mia_canary.summary;
const TP = R.exp5_temporal, TPB = Object.fromEntries(TP.models.map((m) => [m.key, m]));
const FP = E1.gnn_fullpool_seed0;
const ABH = Object.fromEntries(R.ablation_hard.synthetic.map((r) => [r.key, r]));
const A0 = R.exp2.tasks.all_attacks.models, AH = R.exp2_hard.tasks.all_attacks.models;
const SQ0 = R.exp2.tasks.sql_injection;
const LMH = R.exp2_hard.tasks.lateral_movement.models;
const F0 = R.exp4.summary, FH = R.exp4_hard.summary;
const eC = M(B.fagi.epsilon), eS = M(A0.fagi.epsilon), eF = M(F0.fedavg_dp.epsilon), eM = M(B.mlp_dp.epsilon);
const nP = split.private_train_per_fold, nR = split.reference_pool_per_fold;
const q = cfg.batch_size / nP, steps = Math.ceil(cfg.epochs / q);
const nSeeds = cfg.seeds.length;
const w = (k, e) => M(SW[k][e]);
const a = (k, atk, e) => M(BA[k][atk][e]);
const rare = ["Infiltration", "Heartbleed", "Bot", "Web Attack - Sql Injection", "Web Attack - XSS"];
const PC = (k) => B[k].per_class_seed0;
const flat = ["rf", "mlp", "cnn"];
const held = ["knn", "gnn", "fagi", "mlp_at", "mlp_dp"];
const best05 = Object.keys(SW).reduce((x, k) => (w(k, "0.5") > w(x, "0.5") ? k : x), "fagi");
const atkNames = { pgd_reretrieved: "PGD followed by re-retrieval", pgd_adaptive: "adaptive PGD", mimicry32: "benign mimicry",
  pgd: "white-box PGD", transfer_from_mlp: "transfer from the MLP", transfer_from_gnn: "transfer from the GNN" };
const inWorst = (k) => Object.entries(BA[k]).filter(([x]) => x !== "pgd_fixed_graph");
const strongest = (k, e) => inWorst(k).reduce((acc, [x, v]) => (M(v[e]) < acc[1] ? [x, M(v[e])] : acc), ["", 2])[0];
const sigOrdered = SIG.filter((s) => s.sigma !== null).sort((x, y) => x.sigma - y.sigma);
const noDP = SIG.find((s) => s.sigma === null);
const sigR = sigOrdered.map((s) => M(s["worst_recall_eps0.5"])), sigC = sigOrdered.map((s) => M(s.clean));
const syn = JSON.parse(fs.readFileSync(path.join(REPO, "data/synthetic/sessions_final.json"), "utf8"));
const nSynTrain = Math.round(syn.length * (1 - 1 / scfgFolds()));
function scfgFolds() { return R.exp2.config.n_folds; }
const nFedMembers = [0, 1, 2, 3].map((o) => Math.floor(R.exp4.member_fraction * syn.filter((s) => s.org_id === o).length)).reduce((x, y) => x + y, 0);
const TPfagiClean = M(TPB.fagi.clean), TPrfClean = M(TPB.rf.clean);
const tpOrder = TP.models.map((m) => m.key);
const tpBest05 = tpOrder.reduce((x, k) => (M(TPB[k]["worst_recall_eps0.5"]) > M(TPB[x]["worst_recall_eps0.5"]) ? k : x), "fagi");

// ---------- claims checked against the results ----------
claim(M(B.rf.clean) === Math.max(...E1.models.map((m) => M(m.clean))), "RF has the best clean accuracy");
claim(E1.models.every((m) => M(m.clean) > 0.95), "every detector exceeds 95% clean accuracy");
claim(flat.every((k) => w(k, "0.1") < 0.6) && w("rf", "0.1") > 0.4 && w("mlp", "0.1") > 0.4, "RF and MLP keep 40-60% at 0.1; CNN keeps less");
claim(held.every((k) => w(k, "0.1") > 0.94), "k-NN, graph models and PGD-trained MLPs keep >94% at 0.1");
claim(w("mlp_at", "0.1") > 0.9 && w("mlp_at", "0.5") < 0.05, "PGD-trained MLP holds at 0.1 and collapses at 0.5");
claim(w("mlp_dp", "0.5") < 0.05 && T1.fagi_vs_mlp_dp_w05.p < 0.01, "pool-free private MLP collapses at 0.5, FAGI far ahead");
claim(best05 === "fagi", "FAGI has the highest mean worst-case recall at 0.5");
claim(Math.abs(w("fagi", "0.5") - w("knn", "0.5")) < 0.05 && T1.fagi_vs_knn_w05.p > 0.05, "FAGI and k-NN level at 0.5, not significantly different");
claim(w("fagi", "0.5") - w("gnn", "0.5") > 0.2 && T1.fagi_vs_gnn_w05.p < 0.01, "FAGI far ahead of the GNN at 0.5");
claim(Math.abs(w("fagi", "0.1") - w("gnn", "0.1")) < 0.02 && T1.fagi_vs_gnn_w01.p > 0.05, "FAGI and GNN level at 0.1");
claim(M(AB.fagi_no_pgd["worst_recall_eps0.5"]) < M(AB.fagi["worst_recall_eps0.5"]) - 0.15, "removing PGD costs >15 points at 0.5");
claim(TA.fagi_vs_no_pgd_w05.wins_a === TA.fagi_vs_no_pgd_w05.n, "PGD helps in every fold");
claim(Math.max(...sigR) - Math.min(...sigR) < 0.1 && sigR[sigR.length - 1] >= sigR[0] && Math.max(...sigC) - Math.min(...sigC) > 0.015,
  "across sigma, robustness stays within 10 points and does not fall with more noise, while clean accuracy moves");
claim(M(AB.fagi_no_guard["worst_recall_eps0.5"]) > M(AB.fagi["worst_recall_eps0.5"]) && TA.fagi_vs_no_guard_w05.p > 0.05, "GNNGuard does not help; removing it raises recall, not significantly");
claim(M(AB.fagi_no_dp.clean) - M(AB.fagi.clean) > 0.005 && M(AB.fagi_no_dp.clean) - M(AB.fagi.clean) < 0.025, "DP costs one to two points");
claim(TA.fagi_vs_no_dp_clean.wins_b === TA.fagi_vs_no_dp_clean.n, "DP lowers clean accuracy in every fold");
claim(M(FP.clean) > M(AB.gnn.clean) && M(FP.clean) - M(AB.gnn.clean) < 0.01, "the pool split costs the GNN under one point");
claim(strongest("gnn", "0.5") === "pgd_adaptive", "adaptive PGD is the strongest single attack on the undefended GNN at 0.5");
claim(["fagi", "fagi_no_dp"].every((m) => strongest(m, "0.5") === "mimicry32"), "mimicry is the strongest single attack on the PGD-trained graph models at 0.5");
claim(["fagi", "fagi_no_dp"].every((m) => a(m, "pgd_adaptive", "0.5") < a(m, "pgd_reretrieved", "0.5") && a(m, "pgd_reretrieved", "0.5") < a(m, "pgd_fixed_graph", "0.5")), "adaptive < re-retrieved < fixed-graph PGD for PGD-trained graph models");
claim(T3["fagi_vs_fagi_no_dp_eps0.5"].mean_diff > 0 && T3["fagi_vs_fagi_no_dp_eps0.5"].p < 0.05 && T3["fagi_vs_fagi_no_dp_eps0.1"].mean_diff < 0, "DP raises recall slightly at 0.5 and lowers it at 0.1");
claim(a("fagi", "pgd_fixed_graph", "0.5") - w("fagi", "0.5") > 0.08, "fixed-graph evaluation overstates FAGI robustness by >8 points at 0.5");
claim(w("fagi", "0.5") < Math.min(...inWorst("fagi").map(([, v]) => M(v["0.5"]))), "worst case over attacks is below every single attack");
claim(a("knn", "mimicry32", "0.5") < Math.min(a("knn", "transfer_from_mlp", "0.5"), a("knn", "transfer_from_gnn", "0.5")), "mimicry is the strongest attack on k-NN");
claim(["gnn", "fagi", "fagi_no_dp"].every((m) => M(K100[m]["0.5"].k20_restarts) <= M(K100[m]["0.5"].k100) + 0.005), "20 steps with 2 restarts is at least as strong as 100 steps with 1");
claim(["gnn", "fagi", "fagi_no_dp"].every((m) => Math.abs(M(K100[m]["0.5"].k20_restarts) - M(K100[m]["0.5"].k100)) < 0.05), "K=100 within 5 points of K=20");
claim(Object.values(MIA).every((v) => Math.abs(M(v.auc) - 0.5) < 0.03), "loss-based MIA at chance, positive control included");
claim(CAN.gnn.auc_pooled > CAN.fagi.auc_pooled && CAN.fagi_no_dp.auc_pooled > CAN.fagi.auc_pooled, "DP lowers canary-audit AUC");
claim(CAN.gnn.auc_pooled_ci95[0] > 0.5 && CAN.fagi_no_dp.auc_pooled_ci95[0] > 0.5, "canary AUC CIs of non-private models exclude 0.5");
claim(CAN.fagi.auc_pooled_ci95[0] < 0.5 && CAN.fagi.auc_pooled_ci95[1] > 0.5, "canary AUC CI of FAGI includes 0.5");
claim(CAN.fagi.empirical_eps_lower_bound < eC, "empirical epsilon lower bound is below the accountant's epsilon");
claim(rare.some((r) => PC("gnn")[r] && PC("fagi")[r] && PC("fagi")[r].recall < PC("gnn")[r].recall - 0.1), "DP hurts at least one rare class by >10pp");
claim(M(A0.fagi.clean) > 0.99 && M(A0.rf.clean) > 0.99, "original generator saturates");
claim(M(AH.fagi_no_payload.clean) < M(AH.fagi.clean) - 0.05, "payload feature matters on hard data");
claim(LMH.fagi.per_type_detection.lateral_movement < 0.1 && M(LMH.fagi.clean) > 0.8, "on the hard lateral-movement task FAGI has high accuracy but near-zero recall");
claim(TPrfClean < M(B.rf.clean) && TPfagiClean < M(B.fagi.clean), "the temporal split lowers clean accuracy");

const order = [];
function cite(text) {
  return text.replace(/(?<![_^])\{([a-z0-9_,]+)\}/g, (_, keys) => keys.split(",").map((key) => {
    if (!REFS[key]) throw new Error("unknown ref " + key);
    if (!order.includes(key)) order.push(key);
    return `[${order.indexOf(key) + 1}]`;
  }).join(""));
}
const P = (t, o) => L.body(cite(t), o);
const k = [];

const TITLE = "FAGI: Measuring the Costs of Adversarial Training, Edge Filtering, and Differential Privacy in Graph-Based Threat Detection";
const title = [
  new L.Paragraph({ alignment: L.AlignmentType.CENTER, spacing: { after: 160 }, children: [new L.TextRun({ text: TITLE, font: L.FONT, size: 44 })] }),
  new L.Paragraph({ alignment: L.AlignmentType.CENTER, children: [new L.TextRun({ text: "Shahzaib Ejaz", font: L.FONT, size: 22 })] }),
  new L.Paragraph({ alignment: L.AlignmentType.CENTER, children: [new L.TextRun({ text: "Department of Computer Science, North American University, Houston, TX, USA", font: L.FONT, size: 18, italics: true })] }),
  new L.Paragraph({ alignment: L.AlignmentType.CENTER, spacing: { after: 240 }, children: [new L.TextRun({ text: "sejaz@student.na.edu", font: L.FONT, size: 18, italics: true })] }),
];

const FED_TEXT = require("./fed_text")({ F0, FH, M, pct, fx, claim });
const TEMP_TEXT = require("./temporal_text")({ TP, TPB, B, AB, M, pct, int, claim, w });

const abstract = `Learned detectors for network and API threats face two pressures that are usually studied apart: attackers who perturb their traffic to evade the model, and privacy rules that keep organizations from pooling logs. We present FAGI, which combines a GraphSAGE detector over k-nearest-neighbor flow neighborhoods, GNNGuard edge filtering, PGD adversarial training, and DP-SGD inside federated averaging, and we measure what each component buys and costs. On all eight days of CICIDS2017 (${int(ds.clean_rows)} flows after cleaning; a stratified sample of ${int(ds.sample_rows)} under 5-fold cross-validation with ${nSeeds} seeds), neighbors come from a reference pool kept apart from the private training set, so DP-SGD gives a valid per-flow guarantee (ε_{DP} = ${fx(eC)}, δ = 10^{-6}). Every model exceeds ${Math.floor(100 * Math.min(...E1.models.map((m) => M(m.clean))))}% clean accuracy and a random forest is best. We attack only malicious flows, in feature space without protocol constraints and through each model's own inference path, with one suite for every model: best-of-32 benign mimicry, transferred examples, and white-box or retrieval-aware adaptive PGD. At an attack budget of ε_{adv} = 0.1 the random forest still detects ${pct(w("rf", "0.1"))}% of attacked flows and an undefended MLP ${pct(w("mlp", "0.1"))}%, while the graph models, a k-NN vote, and PGD-trained MLPs detect more than ${Math.floor(100 * Math.min(...held.map((x) => w(x, "0.1"))))}%. At ε_{adv} = 0.5 FAGI detects ${pct(w("fagi", "0.5"))}%, level with the k-NN vote over the same pool (${pct(w("knn", "0.5"))}%) and well above an undefended GNN (${pct(w("gnn", "0.5"))}%) and every flat model. Adversarial training supplies this robustness; GNNGuard adds none; DP costs ${pp(M(AB.fagi_no_dp.clean), M(AB.fagi.clean))} points of clean accuracy, concentrated on rare attacks. ${TEMP_TEXT.abstract} Loss-based membership inference finds nothing even against an overfit control, while a canary audit separates the non-private models from FAGI. ${FED_TEXT.abstract}`;

k.push(new L.Paragraph({ alignment: L.AlignmentType.JUSTIFIED, spacing: { after: 80 }, children: [
  new L.TextRun({ text: "Abstract", font: L.FONT, size: 18, bold: true, italics: true }), new L.TextRun({ text: "—", font: L.FONT, size: 18, bold: true }),
  ...L.runs(abstract, { size: 18, bold: true })] }));
k.push(new L.Paragraph({ alignment: L.AlignmentType.JUSTIFIED, spacing: { after: 120 }, children: [
  new L.TextRun({ text: "Index Terms", font: L.FONT, size: 18, bold: true, italics: true }),
  new L.TextRun({ text: "—adversarial machine learning, differential privacy, federated learning, graph neural networks, intrusion detection, API security.", font: L.FONT, size: 18, bold: true })] }));

// ---------- I ----------
k.push(L.section("I. Introduction"));
k.push(P(`APIs are the main interface between cloud services, and the OWASP API Security Top 10 {owasp} catalogues how that interface is abused. Salt Security reported a 400% increase in unique API attackers over six months in its Q1 2023 report {salt}, Verizon's 2024 DBIR found that exploitation of vulnerabilities as an initial breach vector rose by 180% in a year {dbir}, and CISA's advisory AA24-038A documented state-sponsored actors holding persistent access to U.S. critical infrastructure for years {cisa}. Machine-learning detectors are a natural response, but two problems follow them into deployment. Attackers adapt their traffic to evade a learned model {szegedy,grosse,apruzzese22}, and organizations cannot pool the logs a detector would learn from {mothukuri}.`));
k.push(P(`Defenses exist for each problem. Adversarial training hardens a model against bounded perturbations {madry}; GNNGuard prunes suspicious edges in graph models {gnnguard}; DP-SGD bounds what a model reveals about any one training record {abadi}; federated averaging trains without moving data {fedavg}. Combining them is attractive, and each combination claims to get the benefits of all its parts. What it costs is rarely measured.`));
k.push(P(`This paper measures it. FAGI builds attributed graphs from network flows or API sessions, classifies them with a two-layer GraphSAGE {sage}, and adds GNNGuard, PGD training, and DP-SGD with federated averaging. We evaluate on all of CICIDS2017 {cicids} and on a synthetic API-session dataset, and for each component ask what it buys under a stated attacker and what it costs.`));
k.push(P(`The contributions are: (1) a construction in which graph neighbors come from a reference pool that is disjoint from the private training set, so that DP-SGD gives a valid per-record guarantee for a neighborhood-based detector, together with a measurement of what the pool costs; (2) an evaluation that attacks malicious flows only, through each model's own inference path, with the same mimicry, transfer, and gradient attacks for every model, and reports the worst case; (3) paired tests over folds and seeds for the main comparisons, and a component ablation and privacy-utility sweep reported as fold win counts; (4) an audit of the privacy claim with both a membership-inference attack and planted canaries; (5) a temporal split that trains on earlier traffic and tests on later traffic; and (6) a federated experiment that tests generalization to an organization whose data never enters training, with DP and federation separated.`));
k.push(P(`The results do not crown a single design. A random forest is the most accurate clean detector but loses about half of the attacked flows at a small budget. Adversarial training is the component that keeps FAGI working as the budget grows; GNNGuard contributes nothing measurable; DP lowers accuracy, mostly on rare attacks. A k-NN vote over the reference pool, with no training at all, is as robust as FAGI at the largest budget we test. Section II reviews related work, Section III states the threat model, Section IV describes FAGI, Section V reports the evaluation, Section VI the ablation, and Section VII discusses limitations.`));

// ---------- II ----------
k.push(L.section("II. Related Work"));
k.push(L.subsection("A. API Security and Evaluation Practice"));
k.push(P(`Ejaz catalogued cloud API management anti-patterns such as excessive trust delegation and missing schema validation {ejaz}, and supply-chain compromises add structural risk {ohm,ladisa}. Sommer and Paxson explained why detectors trained on one traffic distribution often fail on another {sommer}, and Arp et al. list evaluation pitfalls in security ML, including data snooping and unrealistic splits {arp}. CICIDS2017 {cicids} is the most used benchmark of its kind and has documented labeling problems {engelen}.`));
k.push(L.subsection("B. Graph Neural Networks for Intrusion Detection"));
k.push(P(`Graph neural networks learn from structure and attributes through message passing {sage}. E-GraphSAGE applies GraphSAGE to flow graphs for intrusion detection {egraphsage}, and graph learning has worked for related tasks such as security patch detection {graphspd}. Wang et al. re-evaluated graph NIDS and found that published results overstate what they achieve under careful evaluation {wang25}. Neural detectors over flow features include autoencoders {kitsune} and CNNs {li_cnn}, and CNNs over raw bytes have been used for malware {raff}.`));
k.push(L.subsection("C. Adversarial Machine Learning in Security"));
k.push(P(`Small, targeted input changes flip deep classifiers {szegedy}, in malware {grosse} and in network detection {apruzzese22,sheatsley}. Gradient attacks in feature space are not the attacks practitioners face {apruzzese23}, and attacks that respect problem-space constraints are a stricter test {pierazzi,zhan25}. Madry et al. formalized PGD adversarial training {madry}. Defenses with non-differentiable parts can mask gradients {athalye}, and evaluations should use adaptive attacks designed against the specific defense {tramer} and strong parameter-free attacks {autoattack}.`));
k.push(L.subsection("D. Payload Semantics"));
k.push(P(`Language models pre-trained on security text {securebert} and learned API-sequence embeddings {api2vec} have been used to represent payloads and call sequences, and Kim et al. document the latency cost of LLM features in payload-based detection {kim25}. Our payload signal is a deliberately simple keyword count, so its effect can be isolated.`));
k.push(L.subsection("E. Differential Privacy and Federated Learning"));
k.push(P(`DP-SGD clips each example's gradient and adds Gaussian noise {abadi}; Rényi DP gives tight composition {mironov}, converted to (ε, δ) with the bound of Balle et al. {balle} as implemented in Opacus {opacus}. Tuning hyperparameters on private data consumes privacy budget {papernot_tuning}. For graphs, one node can appear in many neighborhoods, and node-level guarantees need bounded degree or aggregation perturbation {daigavane,gap}; our reference-pool construction avoids the issue by never placing a private record in another record's neighborhood. DP-SGD costs more accuracy on under-represented classes {bagdasaryan}. Membership inference measures leakage empirically, with shadow models {shokri}, loss thresholds {yeom}, or per-example calibration {carlini}; planting canaries audits DP training directly and yields an empirical lower bound on ε {jagielski}. Federated averaging trains without moving data {fedavg} but gives no formal privacy on its own {mothukuri}, and robust aggregation against poisoning is an active area {sanjalawe}.`));

// ---------- III ----------
k.push(L.section("III. Threat Model"));
k.push(L.subsection("A. Adversaries"));
k.push(P(`We consider three adversaries {papernot}. The evasion adversary controls its own malicious traffic and changes the features of its flows or sessions so that the detector labels them benign. It cannot change other users' traffic or the reference pool. We give it white-box access to the model and to the retrieval procedure, which is stronger than the limited-knowledge attacker of most deployments. The inference adversary sees the trained model, as a federation member or the aggregation server would, and tries to learn whether a given record was in an organization's private training data. The poisoning adversary controls federation members and sends malicious updates; we do not defend against or evaluate it.`));
k.push(L.subsection("B. Privacy Unit"));
k.push(P(`The protected unit is one record of an organization's private training set: one flow for CICIDS2017, one session for API data. Every training example is the ego graph of exactly one private record, with neighbors drawn from a reference pool that contains no private records, so adding or removing a private record changes exactly one clipped per-example gradient and DP-SGD's guarantee applies per record. The reference pool is not protected. It plays the role of data an organization already shares or has consent to use, and in our experiments it is a held-out half of each training fold. Feature standardization for CICIDS2017 is fit on the pool; API features are divided by fixed per-feature scales computed once from a public calibration sample, a separate run of the original generator that is never evaluated (the hard generator uses the same scales). Two things fall outside the guarantee: the reference pool itself, and the size of each organization's dataset, which federated weighting reveals. The guarantee is per record, not per host or user.`));

// ---------- IV ----------
k.push(L.section("IV. FAGI Architecture"));
k.push(L.subsection("A. Graph Construction"));
k.push(P(`For API sessions, each of 12 endpoints is a node, and each pair of consecutive calls adds an edge in both directions. A node carries six features: call count, mean duration, mean response size, maximum privilege level, write ratio, and a payload suspicion score that counts tokens from a fixed lexicon of SQL keywords, privilege terms, and export terms. For flows, each query flow becomes the center of an ego graph with its k = ${cfg.k} nearest neighbors from the reference pool, in standardized feature space. The center connects to every neighbor, and each node also connects to its ${cfg.k_inner} nearest nodes within the graph. The payload score does not exist for flows, so on CICIDS2017 all graph models see the same ${E1.n_features} features.`));
k.push(L.subsection("B. Detector and Defenses"));
k.push(P(`The detector has two GraphSAGE layers with mean aggregation and 64 and 32 hidden units {sage}, ReLU, dropout 0.25, and a linear output layer, with 13,090 parameters on flows and 5,026 on API sessions (4,898 without the payload feature). Session graphs are classified from the mean node embedding and flow graphs from the center node. Before each forward pass, GNNGuard-style filtering removes edges whose endpoint features have cosine similarity below τ = ${cfg.gnnguard_tau} {gnnguard}, recomputed on whatever features the model receives, including perturbed ones. Adversarial training minimizes ½L(x) + ½L(x + δ*), where δ* comes from PGD with K_{p} = ${cfg.pgd_train_steps} steps, a random start, step 2.5ε/K_{p}, and ε_{adv} = ${cfg.pgd_eps} {madry}, applied only to nodes the adversary controls.`));
k.push(L.subsection("C. Private Federated Training"));
k.push(P(`Each organization trains locally with DP-SGD {abadi}: every private record enters a batch independently with probability q, each record's gradient of the combined clean and adversarial loss is clipped to norm C = ${cfg.dp_clip}, Gaussian noise with standard deviation σC is added to the sum, and the result is divided by the expected batch size before an Adam step. Per-example gradients are exact. Adam only post-processes the noisy gradient, so keeping its state across rounds does not change the guarantee. Privacy loss is tracked with a Rényi-DP accountant over orders from 1.1 to 1024 {mironov,opacus} and converted at δ = 10^{-6} {balle}. The server averages client models weighted by local dataset size {fedavg}. Algorithm 1 gives the procedure; centralized training is the case of one organization and one round.`));
k.push(L.spacer());
k.push(L.algorithm("**Algorithm 1:** FAGI training with DP-SGD", [
  "**Input:** organizations 1..m with private data D_{i} and reference pool P_{i};",
  "   rounds R; local epochs E; rate q; noise σ; clip C; budget ε_{adv}, steps K_{p}",
  "1: initialize global weights W_{0}; Adam state S_{i} ← ∅ for each i",
  "2: **for** r = 1..R **do**",
  "3:   **for** each organization i **in parallel do**",
  "4:     W ← W_{r−1}; resume Adam from S_{i}",
  "5:     **repeat** ⌈E/q⌉ times:",
  "6:       B ← each record of D_{i} independently with prob. q",
  "7:       **for** each record x ∈ B **do**",
  "8:         G ← ego graph of x with neighbors from P_{i}",
  "9:         δ* ← PGD(f_{W}, G; ε_{adv}, K_{p})  on the adversary's nodes",
  "10:        g_{x} ← ∇_{W}[ ½L(f_{W}(G), y) + ½L(f_{W}(G+δ*), y) ]",
  "11:        g_{x} ← g_{x} / max(1, ‖g_{x}‖_{2} / C)",
  "12:      W ← Adam(W, (Σ_{x} g_{x} + N(0, σ^{2}C^{2}I)) / (q|D_{i}|))",
  "13:      record one step at (q, σ) in the accountant",
  "14:    W_{i} ← W; save Adam state to S_{i}",
  "15:  W_{r} ← Σ_{i} (|D_{i}| / Σ_{j}|D_{j}|) W_{i}",
  "16: **return** W_{R} and ε_{DP,i} for each organization",
]));
k.push(L.caption("f_W applies GNNGuard to its input graph on every call. At inference, a query flow's neighbors come from the reference pool of the organization that runs the detector; in centralized CICIDS2017 experiments, from the fold's single pool.", false));
k.push(L.subsection("D. Cost"));
k.push(P(`Per-example gradients and PGD dominate cost. On a two-core CPU, one CICIDS2017 fold (${int(nP)} private training graphs, ${cfg.epochs} epochs) takes about 15 seconds for the undefended GNN and two to six minutes for FAGI. Inference is one nearest-neighbor query against the reference pool and one forward pass over a nine-node graph.`));

// ---------- V ----------
k.push(L.section("V. Evaluation"));
k.push(L.subsection("A. Data and Protocol"));
k.push(P(`**CICIDS2017.** We use all eight day files of the MachineLearningCSV release {cicids} (${int(ds.raw_rows)} flows, 15 labels). We remove ${int(ds.dropped_nonfinite)} flows with non-finite values, ${int(ds.dropped_duplicates)} exact duplicates, a duplicated header column, and ${ds.dropped_constant_features.length} constant features, leaving ${int(ds.clean_rows)} flows and ${E1.n_features} features. Mis-encoded Web Attack labels are normalized; ${int(ds.rows_with_conflicting_labels)} flows share features with a flow of another label and are kept {engelen}. Per-example DP-SGD and adaptive attacks on two million flows were not feasible on our hardware, so we draw a stratified sample: up to ${int(ds.sample_per_class_cap)} flows from each attack label (all flows of rarer ones) and an equal number of benign flows, ${int(ds.sample_rows)} in total. The task is binary. We use 5-fold cross-validation stratified on the original label, with ${nSeeds} seeds per fold, and a temporal split described in Section V-G. Each training fold is split in half, stratified, into a private training set (${int(nP)} flows) and a reference pool (${int(nR)} flows). Features are transformed with sign(x)·log(1 + |x|) and standardized with pool statistics.`));
k.push(P(`**Synthetic API sessions.** The generator produces 4,000 sessions across five organizations with four attack types, each present in three organizations. Attack sessions in the original generator always carry tokens that benign sessions never do. We therefore also use a harder variant in which 30% of benign sessions add one to three admin, export, or query calls with admin, export, SQL, or filter tokens, and each attack call carries ordinary tokens with probability 0.5 (camouflaged exfiltration calls also lose their enlarged responses). Results on synthetic data are a sanity study of the API setting, not evidence about real APIs. These use 5-fold cross-validation with one seed.`));
k.push(P(`**Models.** On CICIDS2017 we compare a random forest (100 trees, raw features), a k-NN majority vote over the same ${cfg.k} pool neighbors the graph models see, an MLP (128-64), the same MLP with PGD training, the PGD-trained MLP with DP-SGD, a 1D CNN over the feature vector (two convolutions with 32 channels, max-pooling to 8, a 64-unit layer), an undefended GNN, and FAGI. The private MLP needs no pool, so it trains with DP-SGD on the whole training fold (${int(nP + nR)} flows); it is the pool-free private baseline. Neural models train for ${cfg.epochs} epochs with Adam, expected batch size ${cfg.batch_size}, learning rate ${cfg.lr} for graph models and ${cfg.flat_lr} for flat ones, and weight decay 10^{-4}. These settings were carried over from earlier versions of this work and from common GraphSAGE practice rather than searched on these folds. Earlier versions did report CICIDS2017 results, so the settings are not entirely free of selection, and a strict accounting would charge some privacy budget for that choice {papernot_tuning}. For FAGI, q = ${cfg.batch_size}/${int(nP)} = ${q.toFixed(4)} and training runs ${int(steps)} steps.`));
k.push(P(`**Attacks.** We attack ${int(split.attacked_malicious_per_fold)} randomly chosen malicious test flows per fold, the same flows for every model, and leave benign flows unperturbed. Every model faces benign mimicry, which moves the flow toward each of its 32 nearest benign pool flows within the budget and succeeds if any of the 32 candidates evades, and two transfer attacks, which replay the MLP's PGD examples and the GNN's adaptive-PGD examples. Flat neural models also face white-box PGD (${cfg.pgd_eval_steps} steps, random start). Graph models are attacked through their own retrieval with PGD followed by re-retrieval and with adaptive PGD, which re-retrieves neighbors from the perturbed flow at every one of ${cfg.pgd_eval_steps} steps, runs ${cfg.pgd_restarts} random restarts, and keeps the first iterate that evades. A flow counts as detected only if every attack against that model fails, and we report this worst-case attack recall. PGD on the fixed, unperturbed neighborhood is reported separately and excluded from the worst case, because a deployed model never sees that graph. Budgets ε_{adv} are L∞ distances in standardized log-feature units; a budget of 0.1 corresponds to scaling each raw feature by a factor between roughly e^{−0.1s} and e^{0.1s}, where s is that feature's standard deviation in log space.`));
k.push(P(`**Statistics and reporting.** We compare models with paired Wilcoxon signed-rank tests over the ${5 * nSeeds} fold-seed pairs. The k-NN vote has no random state, so any test that involves it uses the five fold means, for which the smallest attainable two-sided p is 0.0625; seed-0 ablations are reported the same way, with fold win counts. Percentages are rounded to one decimal place (two for clean accuracy and FPR), and differences are computed before rounding. Every cached result is keyed by a hash of the configuration and of the source files that produce it, so results computed in separate sessions cannot mix code versions.`));

k.push(L.subsection("B. Clean Detection"));
k.push(...L.tableTitle("I", `CICIDS2017: clean detection and worst-case attack recall (%, mean ± std over 5 folds × ${nSeeds} seeds)`));
const shortName = { mlp_dp: "MLP + PGD + DP", knn: "k-NN vote", mlp_at: "MLP + PGD" };
k.push(L.table(["Method", "Clean acc.", "FPR", "Recall ε=0.1", "Recall ε=0.5", "ε_{DP}"],
  E1.models.map((b) => ({ bold: b.key === "fagi", cells: [shortName[b.key] || b.name, pm(b.clean, 2), pct(M(b.fpr), 2),
    pm(b["worst_recall_eps0.1"]), pm(b["worst_recall_eps0.5"]), b.epsilon ? fx(M(b.epsilon)) : "—"] })),
  [1290, 1000, 520, 920, 920, 450]));
k.push(L.caption(`Recall: share of attacked malicious flows detected under the worst of all attacks against that model. k-NN statistics are over fold means. δ = 10^{-6}; the private MLP's ε_{DP} is lower because it trains on twice as many records.`, false));
k.push(P(`Table I shows that clean detection on CICIDS2017 is close to solved for every model: the random forest reaches ${pct(M(B.rf.clean), 2)}% with a ${pct(M(B.rf.fpr), 2)}% false-positive rate, the k-NN vote ${pct(M(B.knn.clean), 2)}%, the undefended GNN ${pct(M(B.gnn.clean), 2)}%, and FAGI ${pct(M(B.fagi.clean), 2)}%. The k-NN result is a warning about the benchmark, not only a baseline: random cross-validation places near-duplicate flows from the same attack burst on both sides of the split {arp}, so neighborhood methods do well almost by construction; Section V-G tests this. FAGI trails the GNN by ${pp(-T1.fagi_vs_gnn_acc.mean_diff, 0, 2)} points (Wilcoxon ${pv(T1.fagi_vs_gnn_acc.p)}, ${T1.fagi_vs_gnn_acc.n} pairs), almost all of it from DP.`));
k.push(P(`The reference pool has its own cost, because half of each training fold is no longer used for training. A non-private GNN trained on the whole fold, with neighbors from the whole fold, reaches ${pct(M(FP.clean), 2)}% (seed 0), against ${pct(M(AB.gnn.clean), 2)}% for the same model with the pool split, a cost of ${pp(M(FP.clean), M(AB.gnn.clean), 2)} points. The pool-free alternative is a private flat model: the PGD-trained MLP with DP-SGD on all training flows reaches ${pct(M(B.mlp_dp.clean), 2)}% at ε_{DP} = ${fx(eM)}, ${pp(T1.fagi_vs_mlp_dp_acc.mean_diff, 0, 2)} points below FAGI (${pv(T1.fagi_vs_mlp_dp_acc.p)}).`));
k.push(L.figure(path.join(REPO, "figures/fig1_cicids_benchmark.png"), 3.35, 2.4));
k.push(L.caption("Fig. 1. Clean accuracy and worst-case attack recall at ε_{adv} = 0.1 on CICIDS2017. MLP PGD: PGD-trained MLP; MLP DP: the same with DP-SGD on all training flows."));
const rareRows = rare.filter((r) => PC("gnn")[r]);
k.push(P(`DP's cost falls on rare labels (Table II). With few examples, a class contributes few clipped gradients to sums that carry the same noise as everything else, the disparate impact Bagdasaryan et al. describe {bagdasaryan}. The intervals are wide because these labels have few test flows, and Table II gives the counts.`));
k.push(...L.tableTitle("II", "Recall on rare CICIDS2017 labels (seed 0, pooled over folds; 95% Wilson interval)"));
k.push(L.table(["Label (n)", "RF", "GNN", "FAGI"],
  rareRows.map((r) => [`${r.replace("Web Attack - ", "").replace("Sql", "SQL")} (${PC("gnn")[r].n})`,
    ...["rf", "gnn", "fagi"].map((m) => `${pct(PC(m)[r].recall, 0)} [${pct(PC(m)[r].ci95[0], 0)}–${pct(PC(m)[r].ci95[1], 0)}]`)]),
  [1550, 1080, 1080, 1080]));
k.push(L.spacer());

k.push(L.subsection("C. Robustness"));
k.push(P(`At ε_{adv} = 0.1 the random forest detects ${pct(w("rf", "0.1"))}% of attacked flows, the MLP ${pct(w("mlp", "0.1"))}%, and the CNN ${pct(w("cnn", "0.1"))}%. The strongest attack on the random forest is ${atkNames[strongest("rf", "0.1")]}, so the forest's lack of gradients does not protect it. PGD training holds the MLP at this budget (${pct(w("mlp_at", "0.1"))}%, and ${pct(w("mlp_dp", "0.1"))}% with DP-SGD) but not at ε_{adv} = 0.5 (${pct(w("mlp_at", "0.5"))}% and ${pct(w("mlp_dp", "0.5"))}%). The undefended GNN (${pct(w("gnn", "0.1"))}%), FAGI (${pct(w("fagi", "0.1"))}%), and the k-NN vote (${pct(w("knn", "0.1"))}%) all hold at 0.1; the GNN and FAGI do not differ (${pv(T1.fagi_vs_gnn_w01.p)}), because the attacker controls one of nine nodes.`));
k.push(P(`Fig. 2 shows what happens as the budget grows. The undefended GNN falls to ${pct(w("gnn", "0.5"))}% at ε_{adv} = 0.5, and its strongest attack is ${atkNames[strongest("gnn", "0.5")]}. FAGI keeps ${pct(w("fagi", "0.5"))}% (paired difference ${pp(T1.fagi_vs_gnn_w05.mean_diff, 0)} points, Wilcoxon ${pv(T1.fagi_vs_gnn_w05.p)}), the highest mean of any model, and FAGI without DP keeps ${pct(w("fagi_no_dp", "0.5"))}%. DP lowers recall at 0.1 by ${pp(0, T3["fagi_vs_fagi_no_dp_eps0.1"].mean_diff)} points and raises it at 0.5 by ${pp(T3["fagi_vs_fagi_no_dp_eps0.5"].mean_diff, 0)} (${pv(T3["fagi_vs_fagi_no_dp_eps0.5"].p)}), so its effect on robustness is small and changes sign with the budget. The k-NN vote keeps ${pct(w("knn", "0.5"))}%; FAGI is ahead in ${T1.fagi_vs_knn_w05.wins_a} of ${T1.fagi_vs_knn_w05.n} folds, but the difference of ${pp(T1.fagi_vs_knn_w05.mean_diff, 0)} points is not significant (${pv(T1.fagi_vs_knn_w05.p)}), so we treat the two as level. Mimicry is the strongest attack on the vote (${pct(a("knn", "mimicry32", "0.5"))}%): the attacker must move the flow until most of its pool neighbors are benign, and pulling it toward benign flows is the direct way to do that.`));
k.push(P(`How the attack treats retrieval matters more than its strength. Against FAGI at ε_{adv} = 0.5, PGD on the fixed neighborhood leaves ${pct(a("fagi", "pgd_fixed_graph", "0.5"))}% of attacked flows detected, the same perturbation followed by re-retrieval leaves ${pct(a("fagi", "pgd_reretrieved", "0.5"))}%, and adaptive PGD leaves ${pct(a("fagi", "pgd_adaptive", "0.5"))}%. The strongest single attack on FAGI is not a gradient attack at all: benign mimicry leaves ${pct(a("fagi", "mimicry32", "0.5"))}% (${pct(a("fagi_no_dp", "mimicry32", "0.5"))}% without DP). PGD training teaches the model to resist the perturbations PGD finds, and mimicry looks for a different kind. Transfer from the GNN is weak (${pct(a("fagi", "transfer_from_gnn", "0.5"))}%). Different flows fall to different attacks, so the worst case (${pct(w("fagi", "0.5"))}%) is below any single attack. An evaluation on fixed graphs, the usual choice for graph NIDS, would overstate FAGI's robustness here by ${pp(a("fagi", "pgd_fixed_graph", "0.5"), w("fagi", "0.5"), 0)} points. On seed 0 we also ran adaptive PGD with 100 steps and one restart. At ε_{adv} = 0.5 it leaves ${pct(M(K100.fagi["0.5"].k100))}% of FAGI's attacked flows detected, against ${pct(M(K100.fagi["0.5"].k20_restarts))}% for our 20-step attack with ${cfg.pgd_restarts} restarts (GNN: ${pct(M(K100.gnn["0.5"].k100))}% and ${pct(M(K100.gnn["0.5"].k20_restarts))}%). More steps do not make the attack stronger, while restarts do; we see no sign of gradient masking {athalye,tramer}.`));
k.push(L.figure(path.join(REPO, "figures/fig2_eps_sweep.png"), 3.35, 2.3));
k.push(L.caption(`Fig. 2. Worst-case attack recall on CICIDS2017 as the budget grows (mean ± std over 5 folds × ${nSeeds} seeds).`));

k.push(L.subsection("D. What Privacy Costs and Buys"));
k.push(P(`Fig. 3 varies the noise multiplier. At σ = 1.1, FAGI's guarantee is ε_{DP} = ${fx(eC)}; σ = ${sigOrdered[sigOrdered.length - 1].sigma} gives ε_{DP} = ${fx(M(sigOrdered[sigOrdered.length - 1].epsilon))} at ${pct(sigC[sigC.length - 1], 2)}% clean accuracy, and σ = ${sigOrdered[0].sigma} gives ε_{DP} = ${fx(M(sigOrdered[0].epsilon))} at ${pct(sigC[0], 2)}%, against ${pct(M(noDP.clean), 2)}% without DP. Worst-case recall at ε_{adv} = 0.5 stays between ${pct(Math.min(...sigR))}% and ${pct(Math.max(...sigR))}% across the range and is highest at the largest σ, so in this setting more noise costs clean accuracy without costing robustness. Noise may act as a regularizer that smooths the decision boundary; one seed is not enough to say more.`));
k.push(L.figure(path.join(REPO, "figures/fig3_privacy_utility.png"), 3.35, 2.1));
k.push(L.caption("Fig. 3. Privacy-utility trade-off for FAGI on CICIDS2017 (seed 0, 5 folds). Dotted lines: FAGI without DP."));
const lossTxt = (v) => `${fx(M(v.mean_loss_members), 3)} and ${fx(M(v.mean_loss_nonmembers), 3)}`;
k.push(P(`Measuring what DP buys is harder. A loss-threshold membership attack {yeom}, run through the deployed retrieval path, scores an AUC between ${fx(Math.min(...Object.values(MIA).map((v) => M(v.auc))), 3)} and ${fx(Math.max(...Object.values(MIA).map((v) => M(v.auc))), 3)} for every model, with and without DP, and finds no members at 1% false-positive rate. It also fails on a positive control, a GNN trained for ${cfg.overfit_epochs} epochs on ${int(cfg.overfit_n)} flows (AUC ${fx(M(MIA.gnn_overfit.auc), 3)}, best per-class AUC ${fx(M(MIA.gnn_overfit.max_per_class_auc), 3)}). The reason is visible in the losses: the median loss is ${fx(M(MIA.gnn.median_loss_members), 3)} for members and non-members of every model, and mean losses are ${lossTxt(MIA.gnn)} for the GNN and ${lossTxt(MIA.gnn_overfit)} for the control. Most flows of both kinds are fit almost perfectly, so this attack cannot separate them and its silence says nothing about DP.`));
const canTxt = (x) => `${fx(CAN[x].auc_pooled, 3)} [${fx(CAN[x].auc_pooled_ci95[0], 3)}, ${fx(CAN[x].auc_pooled_ci95[1], 3)}]`;
k.push(P(`A canary audit {jagielski} has more power, and on this data it is our only empirical privacy evidence. We flip the labels of 200 private flows per fold, train with half of them, and ask whether the model's loss separates included from excluded canaries. Pooled over folds (${CAN.fagi.n_in + CAN.fagi.n_out} canaries; bootstrap 95% intervals), the undefended GNN separates them with AUC ${canTxt("gnn")} and FAGI without DP with ${canTxt("fagi_no_dp")}; FAGI with DP-SGD scores ${canTxt("fagi")}, an interval that contains 0.5. Canaries are barely memorized even without DP: ${pct(M(CAN.gnn.fit_rate_in), 0)}% of included canaries reach a loss below log 2 for the GNN, against ${pct(M(CAN.gnn.fit_rate_out), 0)}% of excluded ones, so the signal DP removes is weak. Per fold, FAGI's AUC is ${fx(M(CAN.fagi.auc_per_fold), 3)} ± ${fx(CAN.fagi.auc_per_fold.std, 3)}. Converting the best threshold into Clopper-Pearson bounds on true- and false-positive rates gives empirical lower bounds on ε of ${fx(CAN.gnn.empirical_eps_lower_bound)} for the GNN, ${fx(CAN.fagi_no_dp.empirical_eps_lower_bound)} for FAGI without DP, and ${fx(CAN.fagi.empirical_eps_lower_bound)} for FAGI, consistent with the accountant's ε_{DP} = ${fx(eC)}. With 1,000 canaries the audit can only certify small lower bounds, so it confirms that DP changes behavior, not that ${fx(eC)} is tight.`));

k.push(L.subsection("E. API-Native Evaluation"));
k.push(...L.tableTitle("III", "Synthetic API sessions, all attack types (5-fold CV, %)"));
const apiRows = [["rf", "Random forest"], ["gnn_nopayload", "GNN (5 features)"], ["gnn", "GNN (6 features)"], ["fagi_no_payload", "FAGI − payload"], ["fagi", "FAGI"]];
k.push(L.table(["Method", "Original", "Hard", "Hard, PGD"],
  apiRows.map(([key, name]) => ({ bold: key === "fagi", cells: [name, pct(M(A0[key].clean)), `${pct(M(AH[key].clean))} ± ${pct(AH[key].clean.std)}`,
    AH[key].adv ? pct(M(AH[key].adv)) : "n/a"] })), [1700, 900, 1100, 900]));
k.push(L.caption(`FAGI variants train with DP-SGD on ${int(nSynTrain)} sessions per fold (ε_{DP} = ${fx(eS)}). PGD: ε_{adv} = 0.1 in calibration-standard-deviation units on called endpoints. The random forest uses node features and the flattened adjacency, including the payload score. The single-attack runs in the text train on fewer sessions (ε_{DP} = ${fx(M(LMH.fagi.epsilon))}).`, false));
const minOrig = Math.min(M(A0.rf.clean), M(A0.gnn.clean), M(A0.fagi.clean));
k.push(P(`On the original generator every model with the payload score reaches ${minOrig >= 0.9995 ? "100%" : `at least ${(Math.floor(1000 * minOrig) / 10).toFixed(1)}%`} accuracy, so the dataset cannot support claims about payload modeling. Without the score, DP-trained FAGI detects ${pct(SQ0.models.fagi_no_payload.per_type_detection.sql_injection)}% of SQL injection sessions, since nothing else in them differs from benign ones. On the hard generator the payload score is worth ${pp(M(AH.fagi.clean), M(AH.fagi_no_payload.clean))} points to FAGI, and DP is again the largest cost: FAGI reaches ${pct(M(AH.fagi.clean))}% against ${pct(M(AH.gnn.clean))}% for the undefended GNN with the same features. Accuracy hides where that cost lands. On the hard lateral-movement task, FAGI's accuracy is ${pct(M(LMH.fagi.clean))}% but it detects only ${pct(LMH.fagi.per_type_detection.lateral_movement)}% of lateral-movement sessions, against ${pct(LMH.rf.per_type_detection.lateral_movement)}% for the random forest and ${pct(LMH.gnn_nopayload.per_type_detection.lateral_movement)}% for the GNN: under DP the model has learned to call almost everything benign.`));
k.push(P(`Random edge injection into hard-generator test graphs changes accuracy by at most ${pct(Math.max(...Object.values(GS.drop_at_80pct).map((v) => M(v))), 2)} points at +80% edges for any model (GNN ${pct(M(GS.drop_at_80pct.gnn), 2)}, FAGI without GNNGuard ${pct(M(GS.drop_at_80pct.fagi_no_guard), 2)}, FAGI ${pct(M(GS.drop_at_80pct.fagi), 2)} points). Random injection is a weak structure attack, and these differences do not establish a benefit from GNNGuard.`));

k.push(L.subsection("F. Federated Generalization"));
k.push(P(`Organizations 0–3 train and organization 4, the only one with all four attack types, is held out. Each training organization keeps 20% of its sessions as non-members. We run ${R.exp4.rounds} rounds of one local epoch, keep each client's Adam state across rounds, and repeat ${R.exp4.repeats} times. Client DP-SGD gives ε_{DP} = ${fx(eF)} per organization, since each holds about ${int(Math.round(nFedMembers / 4))} member sessions; the "ε matched" row trains centrally with σ = ${fx(M(F0.central_dp_eps_matched.sigma))}, chosen so that its ε_{DP} equals the per-client value. The central DP row (ε_{DP} = ${fx(M(F0.central_dp.epsilon))}) differs from Table III's ${fx(eS)} only because it trains on ${int(nFedMembers)} member sessions instead of ${int(nSynTrain)}.`));
k.push(...L.tableTitle("IV", "Accuracy on the held-out organization (%, mean ± std over repeats)"));
k.push(L.table(["Setting", "Original", "Hard", "ε_{DP}"],
  FED_TEXT.rows.map(([key, name]) => ({ bold: key === "fedavg_dp", cells: [name, pm(F0[key].accuracy), pm(FH[key].accuracy),
    F0[key].epsilon ? fx(M(F0[key].epsilon)) : "—"] })), [2050, 1000, 1000, 550]));
k.push(L.caption("Local rows average the four organizations, each trained alone; their ε_{DP} is from their own accountant.", false));
k.push(P(FED_TEXT.body));

k.push(L.subsection("G. Temporal Split"));
k.push(P(TEMP_TEXT.intro));
k.push(...L.tableTitle("V", "CICIDS2017 temporal split: train on the earlier 70% of each day, test on the later 30% (%, mean over 3 seeds)"));
k.push(L.table(["Method", "Clean acc.", "Det. rate", "FPR", "Recall ε=0.1", "Recall ε=0.5"],
  TP.models.map((b) => ({ bold: b.key === "fagi", cells: [shortName[b.key] || b.name, pct(M(b.clean), 2), pct(M(b.detection_rate), 1), pct(M(b.fpr), 2),
    pct(M(b["worst_recall_eps0.1"])), pct(M(b["worst_recall_eps0.5"]))] })), [1450, 850, 750, 600, 800, 800]));
k.push(L.spacer());
k.push(P(TEMP_TEXT.body));

// ---------- VI ----------
k.push(L.section("VI. Ablation and Case Study"));
k.push(L.subsection("A. Ablation"));
k.push(...L.tableTitle("VI", "Ablation (seed 0, 5 folds, %)"));
const abRows = [["fagi", "FAGI (full)"], ["fagi_no_guard", "− GNNGuard"], ["fagi_no_pgd", "− PGD training"], ["fagi_no_dp", "− DP-SGD"], ["gnn", "GNN baseline"]];
k.push(L.table(["Variant", "CICIDS clean", "Recall ε=0.1", "Recall ε=0.5", "Hard clean", "Hard PGD"],
  abRows.map(([key, name]) => { const hk = key === "gnn" ? "gnn_nopayload" : key;
    return { bold: key === "fagi", cells: [name, pct(M(AB[key].clean), 2), pct(M(AB[key]["worst_recall_eps0.1"])), pct(M(AB[key]["worst_recall_eps0.5"])),
      ABH[hk] ? pct(M(ABH[hk].clean)) : "—", ABH[hk] ? pct(M(ABH[hk].adv)) : "—"] }; }),
  [1250, 850, 800, 800, 700, 700]));
k.push(L.caption("CICIDS recall is worst-case attack recall. On the hard synthetic data the GNN baseline row has no payload score.", false));
const tw = (t) => `${t.wins_a} of ${t.n} folds`;
k.push(P(`Removing PGD training lowers worst-case recall at ε_{adv} = 0.5 from ${pct(M(AB.fagi["worst_recall_eps0.5"]))}% to ${pct(M(AB.fagi_no_pgd["worst_recall_eps0.5"]))}%, and the full model is ahead in ${tw(TA.fagi_vs_no_pgd_w05)}. Removing GNNGuard raises it to ${pct(M(AB.fagi_no_guard["worst_recall_eps0.5"]))}% (the full model is ahead in only ${tw(TA.fagi_vs_no_guard_w05)}, ${pv(TA.fagi_vs_no_guard_w05.p)}), so the filter may hurt slightly and certainly does not help: the attacks change node features, and a filter that prunes edges between dissimilar nodes gives an adaptive attacker nothing to fear. Removing DP-SGD raises clean accuracy from ${pct(M(AB.fagi.clean), 2)}% to ${pct(M(AB.fagi_no_dp.clean), 2)}% in every fold, while worst-case recall at ε_{adv} = 0.5 is ${pct(M(AB.fagi_no_dp["worst_recall_eps0.5"]))}% without DP and ${pct(M(AB.fagi["worst_recall_eps0.5"]))}% with it. Adversarial training is the component that makes the graph detector robust; DP buys a formal guarantee at a measurable cost in clean accuracy; GNNGuard does not earn its place in this pipeline. With five folds, the smallest attainable p is 0.0625, so these are fold counts, not significance claims.`));
k.push(L.subsection("B. Case Study: A Multi-Stage Campaign"));
const stages = [["schema_recon", "reconnaissance"], ["sql_injection", "injection"], ["lateral_movement", "lateral movement"], ["slow_exfiltration", "exfiltration"]];
const worstStage = stages.reduce((x, s) => (AH.fagi.per_type_detection[s[0]] < AH.fagi.per_type_detection[x[0]] ? s : x), stages[0]);
k.push(P(`A campaign of the kind CISA describes {cisa} moves through reconnaissance, injection, lateral movement, and exfiltration, and the four synthetic attack types match these stages. On the hard generator, with all attack types in training, ${stages.map(([t, n]) => `${n} recall is ${pct(AH.rf.per_type_detection[t])}% for the random forest, ${pct(AH.gnn.per_type_detection[t])}% for the six-feature GNN, and ${pct(AH.fagi.per_type_detection[t])}% for FAGI`).join("; ")}. Under DP, FAGI is weakest at ${worstStage[1]}. This is a reading of synthetic per-type recall, not a replay of a real campaign.`));

// ---------- VII ----------
k.push(L.section("VII. Discussion"));
k.push(L.subsection("A. What the Evidence Supports"));
k.push(P(`Three findings hold across our experiments. Adversarial training is worth its cost when the attack budget is non-trivial, and it is what separates FAGI from an undefended GNN once the attacker can move flows far enough to change their neighborhoods. DP-SGD at σ = 1.1 costs one to two points of accuracy on CICIDS2017, more on rare classes, and much more when organizations hold little data; the σ sweep lets a practitioner pick the trade, and in our sweep robustness did not fall as σ grew. And the cheapest model in our study, a k-NN vote over a shared reference pool, is as robust as FAGI at large budgets. Any claim that a graph detector is robust on this benchmark should be checked against that baseline. ${TEMP_TEXT.discussion}`));
k.push(L.subsection("B. Limitations"));
k.push(P(`Our temporal split is within each day file, so attack types never seen in training are not tested, and the cross-validation results share near-duplicate flows across the split {arp}. We use the original labels, which have known errors {engelen}, and a sample of ${int(ds.sample_rows)} flows rather than all ${int(ds.clean_rows)}. The attacks work in feature space, and a perturbed flow need not be traffic an attacker can send {apruzzese23,pierazzi,zhan25}; budgets are also applied to features the attacker does not control, such as backward-direction statistics, which makes the attacker stronger than a real one. Our strongest attacks are ours; a parameter-free suite such as AutoAttack {autoattack} adapted to retrieval would be a stronger check. The reference pool is assumed shareable; if it must be private, the node-level methods of {daigavane,gap} apply. The guarantee is per record, while a host contributes many flows. The API data are synthetic and the hard generator is our own design. The payload score is a keyword count; the repository's DistilBERT embedder is not used in any experiment. Federation is simulated in one process, and we evaluate no poisoning defense.`));
k.push(L.subsection("C. Future Work"));
k.push(P(`Problem-space attacks that respect protocol constraints {pierazzi,sheatsley}, optimized structure attacks against GNNGuard, splits that hold out whole days or attack types, and host-level privacy accounting are the next tests this design should face.`));

k.push(L.section("VIII. Conclusion"));
k.push(P(`FAGI combines graph detection, adversarial training, edge filtering, and differentially private federated learning, and this paper measures each component instead of assuming it helps. On CICIDS2017, clean detection is easy for every model. Under a symmetric suite of adaptive attacks, adversarial training keeps FAGI at ${pct(w("fagi", "0.5"))}% worst-case recall at ε_{adv} = 0.5 where an undefended GNN keeps ${pct(w("gnn", "0.5"))}%, GNNGuard adds nothing, and an untrained k-NN vote does about as well as FAGI. Drawing neighbors from a separate reference pool makes DP-SGD's guarantee valid per record (ε_{DP} = ${fx(eC)}) and costs under half a point of clean accuracy on its own; DP itself costs more, and its cost falls on rare attacks. Code, data loaders, and the scripts that produce every number in this paper are public.`));

k.push(new L.Paragraph({ children: [new L.TextRun({ text: "Acknowledgment", font: L.FONT, size: 20, smallCaps: true })], alignment: L.AlignmentType.CENTER, spacing: { before: 200, after: 100 } }));
k.push(P(`The author thanks Dr. Ihsan Said (North American University) for guidance on research methodology. CICIDS2017 was provided by the Canadian Institute for Cybersecurity, University of New Brunswick. The author used an AI assistant (Claude, Anthropic) in developing experiment code, running analyses, and drafting and editing text; the author reviewed all code, results, and text and takes responsibility for the content. Code, loaders, generator, tests, and the scripts that regenerate every table and figure are at https://github.com/Shahzaib-Ejaz/FAGI.`));

k.push(new L.Paragraph({ children: [new L.TextRun({ text: "References", font: L.FONT, size: 20, smallCaps: true })], alignment: L.AlignmentType.CENTER, spacing: { before: 200, after: 100 } }));
order.forEach((key, i) => k.push(L.reference(i + 1, REFS[key])));
const unused = Object.keys(REFS).filter((x) => !order.includes(x));
if (unused.length) console.log("uncited references (omitted):", unused.join(", "));

L.build(title, k, OUT, "MEASURING THE COSTS OF DEFENSES IN GRAPH-BASED THREAT DETECTION").then(() => {
  console.log(`wrote ${OUT}`);
  console.log(`${checks.length} directional claims verified against paper_results.json`);
  console.log(`${order.length} references cited`);
});
