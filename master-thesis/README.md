# Forecast Atlas 2.1

A public research companion by **Developer Noorullah Adel**, based on the submitted experimental package for *Towards Trustworthy Financial Time Series Forecasting*.

Start at `index.html` on GitHub Pages, or use the included local launcher. Choose **Simulation** or **Documentation**.

## What is included

- **Simulation:** nine areas covering the observatory, Research Questions, Forecast Lab, Model Arena, Explanation Studio, Walk-forward Design, Learn, Research Guide, and Evidence Vault.
- **Research Questions:** choose RQ1, RQ2, or RQ3. Its final study conclusion appears immediately above interactive supporting evidence. Filter datasets, horizons, windows, models, fixed explanation probes, and exploratory significance thresholds.
- **Forecast Lab:** recorded forecast replay, a four-slider historical scenario search covering all seven models, verified Linear Regression input interventions, and a seven-model mixture with normalized weight sliders and CSV export. Mixtures are exploratory and were not fitted thesis models.
- **Learn:** nine pathways, seven model workshops, 31 equations with symbols and examples, 54 terminology entries including all 15 appendix abbreviations, 16 knowledge checks, and device-local progress. Graphical activities cover returns, folds, metrics, additive explanations, stability, model mechanics, and block-aware DM comparisons computed in a browser worker.
- **Documentation:** all eight notebooks with their complete source cells, Markdown, recorded outputs, 32 figures, downloads, hashes, search, and print support. Also includes methods, a learning handbook, and 47 bibliography entries.
- **Evidence:** 184 approved original submission files, including 100 CSVs. Source bytes are preserved; hashes are independently checked. Browser datasets contain 129,360 matched forecasts and 78,400 attributions. No server or account is needed.

## Keep the thesis manuscript private

**The complete thesis manuscript is deliberately excluded.** There is no Word/PDF download, full manuscript transcript, hidden full-text JSON, or manuscript figure extraction. Research summaries and learning explanations are newly authored. The original submission archive stays outside this public folder.

**When updating, move the previous `master-thesis` folder outside your local checkout, then copy this edition into the repository.** A fresh folder prevents files from an earlier full-text reader carrying over into the public site. If a manuscript was already committed to a public GitHub repository, deleting the current file does not remove it from Git history. This package makes no changes to your repository or its history.

## Upload with GitHub Desktop

1. Extract the ZIP. Copy the complete `master-thesis` folder into your local `Hands-on-Projects` repository. Keep `research`, `assets`, `data`, `simulation`, and `documentation` together.
2. In GitHub Desktop, review the changes. Only upload this public companion; keep the original submission ZIP and thesis Word/PDF outside the repository.
3. Commit and push. Reuse your existing GitHub Pages setup. No npm installation or build is required for the delivered static site.
4. With Pages publishing the repository root, the gateway will be `https://nad-web.github.io/Hands-on-Projects/master-thesis/`. The two entry paths are `simulation/` and `documentation/` beneath it. These are expected URLs after your deployment, not a claim that this ZIP has been published.

An optional workflow example is included outside the public folder. Use it only if you need a new Pages workflow; it publishes this companion and should not replace an existing multi-app deployment without adapting it. Relative URLs support repository subfolders and names containing spaces.

## Local preview

On macOS, double-click `START_HERE.command` (Python 3 is required). On Windows, use `START_HERE.bat`. Or run this inside the folder:

```bash
python3 tools/serve.py
```

Open the localhost URL printed by the launcher. Opening HTML directly with `file://` cannot load the simulation's JSON datasets. The dedicated notebook HTML pages can still be read directly.

## Meaning of “live”

Controls recalculate values immediately from the **fixed submitted research snapshot**. The app does not fetch current stock prices or silently retrain seven models. Recorded forecasts are genuine exported out-of-sample results. The Linear what-if lab performs inference with OLS coefficients reconstructed from the exact training folds and verified against all 30,240 native Linear forecasts (maximum absolute discrepancy below 3 × 10⁻¹³). No serialized nonlinear fitted weights were supplied.

In **Forecast Lab → Historical scenarios**, choose a reference date and move the four sliders through observed lag values. The app ranks dates using the root mean square gap across the four inputs and includes every tie at the cutoff. The earlier-outcomes setting requires a candidate’s target to predate the reference origin. Click a match to see seven archived forecasts, the actual outcome, the persistence benchmark and the source dates. A share link preserves the inputs, search period, selected date and model. The CSV preserves their full precision.

The worked DAX example compares all seven forecasts issued on 23 February 1994 with the observed five-session cumulative log return of −5.847%. The worked Sunspot example opens 152 rolling one-day forecast origins that share four zero lags. Their next observed values range from 0 to 17. Comparing two dates shows how much information a four-number profile leaves out.

Model-mechanics activities use explicitly labeled teaching examples. User-chosen ensembles, metric transformations, and input interventions do not alter the reported study conclusions. In-browser DM inference reproduces the supplied block-aware unweighted HAC lag-sum calculation and uses a normal-tail approximation; the original CSV remains authoritative. A different exploratory threshold does not change the study's q = 0.05 criterion.

Financial series use log-return inputs and cumulative future log-return targets; Sunspot retains native values. Neural sequences have 21 warm-up rows and four folds, while other models use six. Matched comparisons use 924 common origins per setting. ARIMA native counterfactuals are distinct from SHAP; TFT is the supplied simplified adapter. No unexported ALE curves or causal effects are fabricated.

## Offline availability

From Simulation choose **Keep available offline** on HTTPS or localhost. This saves the interface, full interactive datasets, learning content, and notebook documentation. Original research downloads are cached when opened. Browser storage quotas and platform eviction still apply. The downloaded folder and local launcher provide a reproducible local copy. The updated service worker removes older caches for this companion's scope, including the earlier full-text reader cache.

## Verification and regeneration

For a standard-library data check:

```bash
python3 tools/verify.py
```

With Node.js installed, run `npm test` and `npm run check`. There are no packages to install. The checks independently recompute all 140 metric cells, compare 420 DM tests, verify source hashes and every full notebook code cell, and validate the public package boundary.

To rebuild browser data from the included approved sources, install NumPy in your development environment and run `python3 tools/build_data.py`. This compiler copies only explicitly approved submission paths when given an external source folder; it never extracts a Word manuscript. The bibliography is already curated in `data/guide.json`.

To rebuild the authored classroom: `python3 tools/curate_learning.py`. To regenerate complete notebook HTML and documentation: `node tools/build_docs.mjs`. After changes, run `python3 tools/prepare_release.py`, followed by the verification and Node checks. Model training itself is not required to use this site; the full executed notebooks are available for research reproduction. `tools/run_models.py` provides isolated working folders and checks notebook-compatible dependencies before execution. See `research/requirements.txt` and recorded provenance before attempting full training.

## Attribution and licensing

Application code is licensed under `LICENSE-APP`. This does not relicense the thesis, submitted datasets, notebooks, cited papers, or exported research figures. Their original rights and source attribution remain applicable. Marked and KaTeX are bundled locally with their licenses; no CDN dependency is needed. Bibliography links lead to external sources, whose availability is outside this package.
