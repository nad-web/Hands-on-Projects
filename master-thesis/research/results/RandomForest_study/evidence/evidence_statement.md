# Executed evidence statement — RandomForest

Execution version: `thesis-execution-v1`

This notebook was executed in the recorded environment. Cross-model conclusions are defined only after matched-origin synthesis.

## RQ1
Status: model_execution_complete_cross_model_answer_in_synthesis

- Native-coverage fold metrics are exported with observation-level predictions; seven-way claims use only matched origins in the cross-model synthesis.
  - Value: `{'dax|h1': 0.011903822794148594, 'dax|h5': 0.02632579225871044, 'djia|h1': 0.010929127447099262, 'djia|h5': 0.02321184012314656, 'monash_sunspot|h1': 18.484311114172133, 'monash_sunspot|h5': 29.815514802714777, 'nasdaq|h1': 0.012989967013465864, 'nasdaq|h5': 0.02876575999118022, 'sp500|h1': 0.010292334561175545, 'sp500|h5': 0.021790380575956766}`
  - Artifact: `tables/fold_metrics.csv and tables/observation_predictions.csv`

## RQ2
Status: model_execution_complete_cross_model_answer_in_synthesis

- Consecutive-fold attributions use identical, contiguous, out-of-sample probe origins; Spearman, cosine, and top-feature agreement are exported.
  - Value: `{'TreeSHAP_full_ensemble|bear': 0.7999999999999999, 'TreeSHAP_full_ensemble|boundary': 0.7999999999999999, 'TreeSHAP_full_ensemble|bull': 0.7999999999999999, 'TreeSHAP_full_ensemble|crisis': 0.7999999999999999, 'TreeSHAP_full_ensemble|na': 0.7999999999999999}`
  - Artifact: `tables/stability.csv`

## RQ3
Status: model_execution_complete_economic_identification_bounded

- Faithfulness is evaluated diagnostic by diagnostic; no feature ranking is interpreted as a causal economic driver.
  - Value: `{'max_shap_additivity_residual': 1.9184653865522705e-13}`
  - Artifact: `tables/faithfulness.csv`

## Limitations

- Only lagged prices or native Sunspot values are available; causal economic-driver identification is not possible.
- Regime labels are ex post data-driven strata, not independently dated historical crises.
- Zero-baseline interventions and synthetic local-surrogate samples can leave the natural data manifold.
- Model-specific limitations are documented in the thesis.