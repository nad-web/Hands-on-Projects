# ARIMA Benchmark Correction and Invariance Report

## Correction

ARIMA forecast training-mean benchmark now uses the horizon-specific supervised target over the exact fold training interval. The raw univariate training mean remains separately available only for native counterfactual replacement.

## Executed notebooks

| Notebook | SHA-256 | Execution counts | Stored errors |
|---|---|---|---:|
| ARIMA_corrected | `f4b51700d676e787d5e808a23a9ad7d99fdb48c8704527e37b14e19d93bc3348` | [1, 2, 3, 4, 5] | 0 |
| synthesis_corrected | `03bab453fe281241eaa9c0354bac84b8893a4353768b308bbea573132a4e2eca` | [1, 2] | 0 |

## ARIMA prediction and explanation invariance

All 30,240 realized targets, ARIMA forecasts, persistence forecasts, and model-loss fields are unchanged. All 30,240 exported training-mean predictions changed; the maximum old/new absolute difference is 0.824000000000012.

The attribution, stability, faithfulness, native-parameter, and native fold-metric tables are byte-identical to the preserved pre-correction exports.

## Cross-model invariance

Primary matched prediction and metric fields are unchanged. Only ARIMA training-mean predictions and benchmark-dependent RMSE fields changed in the matched exports. Pairwise DM tests, matched model rankings, stability, faithfulness, and relative-attribution tables are byte-identical.

## Descriptive and inferential effects

ARIMA's financial descriptive lower-loss count against the training mean changed from 4/16 to 1/16.
Three expanding five-session ARIMA training-mean DM statistics changed sign: dax, nasdaq, sp500. All three remain nonsignificant after BH adjustment.

Financial BH14: 0 significant wins, 26 significant losses, and 198 nonsignificant results. Losses comprise 11 against persistence and 15 against the training mean.
Financial BH2 sensitivity: 0 significant wins and 39 significant losses (15 persistence; 24 training mean).
No negative corrected financial differential has unadjusted p < 0.05 (count: 0).

## Independent validation

The independent reconstruction audit passed 1,648/1,648 checks. It checked 760 native fold groups for supervised-target benchmark semantics and native origin coverage. Maximum absolute benchmark discrepancy: 1.42e-14.

## Scientific interpretation

The correction changes benchmark predictions and benchmark-dependent descriptive/inferential values only. ARIMA forecasts, realized targets, persistence predictions, native explanations, selected parameters, model losses, matched seven-way rankings, pairwise tests, stability, and faithfulness evidence are unchanged. No corrected financial model-minus-benchmark differential is a significant win.
