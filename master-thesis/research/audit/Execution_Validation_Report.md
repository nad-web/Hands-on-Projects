# Execution Validation Report

**Result:** 1648/1648 checks passed.

Native predictions: 188,160; native fold metrics: 6; attribution rows: 78,400.

| Model | Predictions | Attributions | Runtime (s) | Checks |
|---|---:|---:|---:|---|
| Linear | 30,240 | 12,800 | 19.89 | 14/14 |
| ARIMA | 30,240 | 12,800 | 494.75 | 14/14 |
| RandomForest | 30,240 | 12,800 | 520.20 | 14/14 |
| XGBoost | 30,240 | 12,800 | 120.89 | 14/14 |
| LightGBM | 30,240 | 12,800 | 125.09 | 14/14 |
| LSTM | 18,480 | 7,200 | 1351.25 | 14/14 |
| TFT | 18,480 | 7,200 | 2010.26 | 14/14 |

## Raw-target reconstruction

- dax: maximum absolute error 9.97e-17
- djia: maximum absolute error 9.99e-17
- monash_sunspot: maximum absolute error 0
- nasdaq: maximum absolute error 9.97e-17
- sp500: maximum absolute error 9.97e-17

## Benchmark semantics

Supervised-target training-mean fold groups checked: 760.
Maximum absolute discrepancy: 1.42e-14.
Native test-origin coverage is also reconstructed independently for every fold.
