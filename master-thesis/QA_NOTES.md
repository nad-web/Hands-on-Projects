# Verification notes

Validated on 17 November 2026.

## Data and source checks

- All 261 supplied artifacts match their recorded inventory SHA-256 hashes.
- All 129,360 matched forecasts round-trip exactly to the original CSV.
- All 140 primary metric cells recompute independently in Python and in the browser numeric module.
- All 78,400 recorded attribution values and their fold/probe/date/regime labels round-trip exactly.
- All 30,240 native Linear forecasts match inference with reconstructed OLS coefficients to numerical precision.
- Eight notebooks have sequential stored execution counts and no stored error outputs.
- The Word reading edition includes eight main chapters, abstract, references, appendices, tables, and embedded images; its 47 reference entries match the extracted bibliography.
- JavaScript syntax checks and six numeric/data-alignment regression tests pass.

See verification-report.json for the machine-readable check scope.

## Browser checks

Tested in Chromium, including a 390 CSS-pixel iframe for narrow-screen layout. The mobile navigation drawer and compact chart layout were inspected. This is not a physical iPhone-device test.

The forecast timeline, playback, visible-window metrics, comparison overlays, exact Linear feature sliders, all seven explanation channels, probe/fold controls, Sunspot settings, temporal splits, adjusted pairwise-test inspection, full-thesis search, bibliography, source-table filtering, notebook code/output preview, hash checking, light/dark themes, and guided tour were exercised.

CSV and SVG exports were downloaded and checked: the CSV contained the 120 visible observations plus its header and all seven forecast columns; the SVG had an explicit SVG namespace and resolved color variables.

The full interface offline pack completed. The preview server was then stopped, the app was reloaded, and the forecast timeline and metrics remained operational from the cache. Unvisited original artifacts require a connection, as the interface states.

## Scope limits

The seven original model training pipelines were not freshly re-executed during app construction. The optional isolated notebook runner requires the separate research dependencies; those are not needed to serve or use the static app. The supplied execution audit remains source evidence.

The repository was not modified or pushed, and GitHub Pages was not activated. The example workflow and publication instructions are supplied for the repository owner.
