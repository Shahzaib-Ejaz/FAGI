# Paper builder

`build_paper.js` writes the paper as DOCX from `../experiments/results/paper_results.json`. Every number in the text and tables is computed from that file, and every directional statement ("FAGI is level with k-NN at ε_adv = 0.5", "DP lowers clean accuracy in every fold") is guarded by a `claim()` check: if a rerun changes the result, the build stops and names the claim. `temporal_text.js` and `fed_text.js` hold the temporal-split and federated sections and their checks.

    npm install
    node build_paper.js            # writes out/FAGI.docx and prints the number of claims checked

The figures it embeds come from `scripts/generate_figures.py`.
