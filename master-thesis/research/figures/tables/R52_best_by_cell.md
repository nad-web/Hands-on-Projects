| series   | h   | scheme   | best     | best_rmse   | rel_rmse   | runner   | runner_rmse   | p_BH     | sig   |
|:---------|:----|:---------|:---------|:------------|:-----------|:---------|:--------------|:---------|:------|
| dax      | 1   | exp      | XGBoost  | 0.008775    | 0.998862   | Linear   | 0.008779      | 0.934084 | no    |
| dax      | 1   | roll     | ARIMA    | 0.010943    | 1.001954   | Linear   | 0.010944      | 0.993977 | no    |
| dax      | 5   | exp      | Linear   | 0.019679    | 0.993733   | ARIMA    | 0.019685      | 0.953225 | no    |
| dax      | 5   | roll     | LightGBM | 0.024013    | 0.999807   | LSTM     | 0.024042      | 0.997007 | no    |
| djia     | 1   | exp      | XGBoost  | 0.009886    | 0.996898   | LightGBM | 0.009888      | 0.935176 | no    |
| djia     | 1   | roll     | LightGBM | 0.009754    | 1.000564   | XGBoost  | 0.009756      | 0.976622 | no    |
| djia     | 5   | exp      | ARIMA    | 0.021394    | 0.985896   | XGBoost  | 0.021430      | 0.685662 | no    |
| djia     | 5   | roll     | Linear   | 0.021386    | 0.997935   | XGBoost  | 0.021561      | 0.075965 | no    |
| nasdaq   | 1   | exp      | ARIMA    | 0.008168    | 0.980695   | XGBoost  | 0.008321      | 0.101864 | no    |
| nasdaq   | 1   | roll     | TFT      | 0.015126    | 1.001432   | LSTM     | 0.015228      | 0.508837 | no    |
| nasdaq   | 5   | exp      | LightGBM | 0.020520    | 0.995020   | ARIMA    | 0.020557      | 0.819824 | no    |
| nasdaq   | 5   | roll     | LSTM     | 0.031977    | 1.015898   | Linear   | 0.032045      | 0.998788 | no    |
| sp500    | 1   | exp      | ARIMA    | 0.006123    | 0.997145   | Linear   | 0.006139      | 0.513182 | no    |
| sp500    | 1   | roll     | TFT      | 0.011419    | 1.002822   | XGBoost  | 0.011482      | 0.632097 | no    |
| sp500    | 5   | exp      | ARIMA    | 0.014004    | 0.991458   | XGBoost  | 0.014008      | 0.948408 | no    |
| sp500    | 5   | roll     | Linear   | 0.024398    | 1.005963   | TFT      | 0.024423      | 0.930663 | no    |
