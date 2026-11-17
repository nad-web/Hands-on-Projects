# Forecast Atlas

An interactive research companion to **“Towards Trustworthy Financial Time Series Forecasting: An Experimental Evaluation of Explainable Machine Learning Models for Stock Price Prediction.”**

Built by **Developer Noorullah Adel**. The app is a static website: no account, API key, backend, CDN, analytics, or npm installation is needed. All interface code, charts, datasets, and the Markdown renderer are included locally.

## Open it locally

On your Mac, open Terminal in this `master-thesis` folder and run:

```bash
python3 tools/serve.py
```

Your browser opens automatically at **http://127.0.0.1:8000/**. Keep Terminal open while using the app; press Control+C to stop it. You can also double-click `START_HERE.command` if your Mac permits running that local launcher. On Windows, use `START_HERE.bat` or `py -3 tools/serve.py`.

Opening `index.html` directly as a file does not load the JSON datasets; use the launcher or the published website. Another dependency-free option is `python3 -m http.server 8000` from this folder.

## Upload with GitHub Desktop

1. Extract the delivery ZIP. It contains the **master-thesis** app folder and an optional **github-setup** folder.
2. In GitHub Desktop, open your local clone of `nad-web/Hands-on-Projects`, then choose **Repository → Show in Finder**.
3. Copy the entire **master-thesis** folder into the repository root. Keep its `assets`, `data`, `research`, `tests`, and `tools` folders together. Your main entry file is now `Hands-on-Projects/master-thesis/index.html`.
4. Commit the added files and choose **Push origin**. Do not upload the delivery ZIP in place of its extracted contents.
5. Enable GitHub Pages using one of the methods below. You do not need to modify the app’s paths for this repository or folder name.

Once the repository root is published, the expected app address is:

**https://nad-web.github.io/Hands-on-Projects/master-thesis/**

This is the expected address after you publish; the delivery does not itself push to GitHub or enable Pages.

### Option A: publish from the branch root

Use your repository’s **Settings → Pages → Build and deployment**, select **Deploy from a branch**, and select the default branch (usually `main`) with **/(root)**. Keep existing root pages and apps in place.

Put an empty `.nojekyll` file at the **repository root** to serve the files directly. A copy is in `github-setup/.nojekyll`; on a Mac, Command+Shift+Period shows hidden files. The app also includes its own `.nojekyll`, but the publishing-root copy is the one that disables Jekyll for branch-root publication.

### Option B: use a Pages workflow

If the repository already has a working Pages workflow, keep that workflow and ensure its published output includes `master-thesis/`.

Otherwise, copy `github-setup/pages.yml` to **.github/workflows/pages.yml at the repository root**, choose **GitHub Actions** as the Pages source, and push. The example publishes the repository root so other static projects remain included. It also checks the app’s data before deployment.

GitHub allows one Pages site per repository. Avoid running two workflows that deploy different subsets to the same Pages site.

Official documentation:

- https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site
- https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

## Explore the app

| Area | What you can do |
|---|---|
| Observatory | See the study’s scale, three research questions, and matched financial performance summary. |
| Forecast Lab | Choose any of seven models, five datasets, two horizons, and two evaluation schemes. Play or scrub 924 matched origins per setting. Compare models, overlay all models, adjust the visible window, switch financial forecasts to implied target levels, and export a CSV or SVG. |
| Linear what-if lab | Change four lag inputs using sliders. Predictions recompute with verified OLS coefficients from the exact executed training intervals. Select a different origin to use its corresponding training fold. Edited inputs and outputs are explicitly hypothetical. |
| Model Arena | Compare all financial settings or one selected experiment. Inspect relative RMSE, rank, DA, and R². Click pairwise-matrix cells to inspect raw p values and adjusted q values. |
| Explanation Studio | Scrub actual attribution probes, choose explained folds, and inspect signed lag contributions. Compare relative reliance, temporal/regime stability, and faithfulness diagnostics. ARIMA native counterfactuals and TFT native signals are labeled separately. |
| Walk-forward Design | Inspect the exact rolling/expanding training and test intervals for each model family, including purge, embargo, and neural warm-up. |
| Thesis & Literature | Read all eight main chapters, abstract, bibliography, and appendices, with extracted tables and embedded figures. Search the whole manuscript or browse its 47 references. |
| Evidence Vault | Search all 261 original files, browse 40 figures, inspect CSVs and executed notebooks with stored outputs, download originals, and verify SHA-256 hashes. |

The side menu includes a guided tour. Press `/` to search the evidence vault. In forecast replay, Space plays/pauses, and the left/right arrows step through origins. All sliders also support normal keyboard input. The link button copies the selected experimental setting, including the replay position; edited what-if inputs are not encoded in the link.

## What is being simulated?

The app gives a live interface to a **fixed, executed research snapshot**. It does not fetch current prices or silently retrain models when a control changes.

- **Forecast replay:** original held-out predictions for all seven models, with live recomputation of metrics over the displayed observations.
- **Explanation replay:** original probe-level attributions, relative importance, stability, and recorded faithfulness diagnostics.
- **Real Linear inference:** OLS coefficients reconstructed with the executed standardized lag features and exact fold intervals. All **30,240** native Linear forecasts are verified; maximum absolute reconstruction error is approximately **2.56 × 10⁻¹³**. Slider changes evaluate this fitted linear function.
- **Full model execution:** the supplied Python/Jupyter notebooks, accessible in the vault and usable with the isolated runner below.

The input package contains no serialized fitted weights for fresh browser inference with ARIMA, Random Forest, XGBoost, LightGBM, LSTM, or TFT. Their recorded forecasts and explanation probes remain fully interactive. Their full training and inference code is preserved in the executed notebooks.

Financial targets are cumulative future **log returns**, and features are lagged log returns at 1, 5, 21, and 252 observations. Sunspot uses native future values. Financial directional accuracy is sign agreement, not a trading strategy or profit measure. Implied target levels use `origin level × exp(predicted log return)`; their level plot is a transformation of the same recorded forecasts.

## Evidence and version precedence

The uploaded package contains materials from different stages of the research. The app uses:

1. **research/Master Thesis.docx** for the reading edition and bibliography.
2. **research/results/cross_model/** for corrected matched-origin comparisons and cross-model explanations.
3. **research/notebooks/*_Executed.ipynb** for the reproduced model implementations and stored execution outputs.

`research/source/Revised_Master_Thesis.md` describes an earlier experiment and contains conclusions that do not match the corrected executed results. It is preserved and labeled as legacy material. The original `source/*.py` scripts are also retained as earlier source files; use the corrected executed notebooks for reruns. The second Word document and supplementary PowerPoint are retained unchanged.

The primary seven-model comparison uses **18,480 distinct matched origins**, or **129,360 model–origin predictions** across 20 dataset/horizon/scheme settings. Native coverage contains **188,160 predictions** and is used separately for supplementary model–benchmark inference. Each matched setting contains four blocks of 231 origins. Explanation probes have their own coverage; the interface does not equate those probes with replay predictions.

The reading edition preserves text, tables, and embedded images from Word. Formulas are extracted as plain text; Word remains authoritative for equation typography, pagination, and submission layout. Literature topic labels are navigation aids, not new literature-review conclusions.

## Verify the delivery

No extra Python libraries are needed for verification:

```bash
python3 tools/verify.py
```

This verifies all original file hashes, exact forecast and attribution round-trips, all 140 independently recomputed metric cells, all native Linear inferences, notebook execution states, key inference labels, manuscript structure, and required local assets. Results are written to `verification-report.json`.

If Node.js is installed, these optional checks use only its standard library:

```bash
npm run check
npm test
```

The included research execution audit reports 1,648 passing checks. That is preserved source evidence. App verification does not claim that every original training pipeline was re-executed while building the interface.

## Rebuild the browser datasets

The shipped JSON files are ready to use. Only regeneration needs NumPy:

```bash
python3 -m pip install numpy
python3 tools/build_data.py
python3 tools/prepare_release.py
python3 tools/verify.py
```

You can also pass a separate complete research folder to `build_data.py`. It copies that source tree into `research/`, compiles compact JSON, extracts the Word reading edition, and reconstructs/verifies the OLS folds. The compiler’s count and alignment assertions intentionally target this thesis experiment; if you change its design, update those assertions and the app’s labels deliberately.

`prepare_release.py` regenerates the content revision and offline cache list after changing app assets or generated data. Regenerate before publishing an update. Keep `data/verification` or report timestamps out of scientific claims; they describe app checks, not new experiments.

## Run the actual notebooks separately

This step is optional and separate from using the website. The original execution environment was Python 3.12 on Linux; exact recorded package versions are listed in `requirements-research.txt` and the original provenance files. Platform-specific package availability and numerical behavior can differ.

In a suitable Python environment:

```bash
python3 -m pip install -r requirements-research.txt
python3 tools/run_models.py --models Linear
```

To run all seven model notebooks followed by the cross-model synthesis:

```bash
python3 tools/run_models.py --models all
```

The runner copies the frozen input data into a fresh `experiment-runs/<timestamp>/` directory, runs notebook code there, and saves newly executed notebooks and results in that directory. **The supplied evidence under research/ is preserved.** Neural-model execution can take substantial time. Use `--output /path/to/new-run` to choose a fresh location.

New experiments do not automatically replace the app’s research snapshot. Audit their coverage, reconstruct the intended complete source package, and regenerate the browser datasets deliberately before publishing different results.

## Offline use and privacy

On GitHub Pages, the app supports installation as a standalone web app. “Keep available offline” saves the interface, all interactive datasets, the reading edition, and its embedded images on the current device. Original research artifacts are cached when opened; unvisited downloads require a connection. Browser storage can be cleared or evicted by the browser. The complete downloaded app always works with the local launcher.

The service worker is scoped to this folder, including when it lives below `/Hands-on-Projects/master-thesis/`, and its cache namespace is scoped to that path. It does not control sibling projects. Theme preference and offline caches stay in your browser. External bibliography links open the cited websites only when clicked.

## Files and implementation

```text
master-thesis/
  index.html                 App entry point
  assets/                    Local CSS, ES modules, icons, Markdown renderer, extracted images
  data/                      Compact generated datasets, manifest, inventory, reading edition
  research/                  All 261 supplied research artifacts, unchanged
  tools/                     Compiler, independent verifier, local server, isolated notebook runner
  tests/                     Numeric and data-alignment regression checks
  manifest.webmanifest       Installable-web-app metadata
  sw.js                      Scope-safe offline support
  cache-manifest.json        Generated offline-pack list and revision
  verification-report.json   App verification results
```

The interface uses semantic HTML, responsive CSS, browser-native ES modules, accessible range inputs, SVG charts, and lazy-loaded data partitions. Its local Markdown renderer is Marked, with its license included under `assets/vendor/`. Notebook HTML is restricted to a safe rendering subset. There is no remote application service to configure or maintain.

## Rights

The newly written app code is MIT licensed in `LICENSE-APP.txt`. That license does **not** grant rights to the thesis text, source research artifacts, original data, figures, or third-party packages. Those retain their existing authorship, licenses, and source terms. The package does not add a license to the supplied research materials.
