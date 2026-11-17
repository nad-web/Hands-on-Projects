# Executed evidence statement — LSTM

Execution version: `corrected-independent-rerun-v1`

This notebook was executed in the recorded environment. Cross-model conclusions are defined only after matched-origin synthesis.

## RQ1
Status: model_execution_complete_cross_model_answer_in_synthesis

- Native-coverage fold metrics are exported with observation-level predictions; seven-way claims use only matched origins in the cross-model synthesis.
  - Value: `{'dax|h1': 0.009716150186117335, 'dax|h5': 0.021774293191613052, 'djia|h1': 0.009571550397234762, 'djia|h5': 0.021768532196843972, 'monash_sunspot|h1': 38.82878956905387, 'monash_sunspot|h5': 37.84219584009698, 'nasdaq|h1': 0.011520184694689933, 'nasdaq|h5': 0.02593483532391226, 'sp500|h1': 0.008648043898467562, 'sp500|h5': 0.019020181343517125}`
  - Artifact: `tables/fold_metrics.csv and tables/observation_predictions.csv`

## RQ2
Status: model_execution_complete_cross_model_answer_in_synthesis

- Consecutive-fold attributions use identical, contiguous, out-of-sample probe origins; Spearman, cosine, and top-feature agreement are exported.
  - Value: `{'KernelSHAP_lag_aggregated|bear': 0.19999999999999998, 'KernelSHAP_lag_aggregated|boundary': 0.39999999999999997, 'KernelSHAP_lag_aggregated|bull': 0.39999999999999997, 'KernelSHAP_lag_aggregated|crisis': 0.19999999999999998, 'KernelSHAP_lag_aggregated|na': 0.39999999999999997}`
  - Artifact: `tables/stability.csv`

## RQ3
Status: model_execution_complete_economic_identification_bounded

- Faithfulness is evaluated diagnostic by diagnostic; no feature ranking is interpreted as a causal economic driver.
  - Value: `{'max_shap_additivity_residual': 3.814697265625e-06}`
  - Artifact: `tables/faithfulness.csv`

## Limitations

- Only lagged prices or native Sunspot values are available; causal economic-driver identification is not possible.
- Regime labels are ex post data-driven strata, not independently dated historical crises.
- Zero-baseline interventions and synthetic local-surrogate samples can leave the natural data manifold.
- Model-specific limitations are documented in the thesis.