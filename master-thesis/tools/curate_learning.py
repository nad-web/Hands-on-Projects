#!/usr/bin/env python3
"""Author the public learning companion, without distributing the thesis manuscript.

The text below is educational exposition written for the companion. Recorded
findings remain traceable to the public CSVs and complete executed notebooks.
"""
from pathlib import Path
import json
import random

ROOT = Path(__file__).resolve().parents[1]
equations = []

def eq(id, title, tex, meaning, symbols, example, limit, topic):
    equations.append(dict(id=id, title=title, tex=tex, meaning=meaning,
                          symbols=symbols, example=example, limit=limit, topic=topic))

eq('log-return', 'One-session log return', r'r_t=\log\!\left(\frac{P_t}{P_{t-1}}\right)',
   'Convert an adjusted closing-level ratio into an additive logarithmic change. The financial inputs in this experiment are lagged log returns rather than lagged price levels.',
   [[r'P_t', 'Adjusted closing level at observation t'], [r'r_t', 'One-session log return']],
   'An index moving from 100 to 102 has log return log(1.02) ≈ 0.019803, or 1.9803%. Its simple percentage return is 2%.',
   'A log return and a simple percentage return are close for small changes, but they are not identical.', 'foundations')
eq('target', 'Cumulative future financial target', r'y_t^{(h)}=\sum_{j=1}^{h}r_{t+j}=\log\!\left(\frac{P_{t+h}}{P_t}\right)',
   'The target is attached to a forecast origin t and becomes observable h trading sessions later. Summing future daily log returns gives the return across that horizon.',
   [[r'h', 'Forecast horizon: one or five observations'], [r'y_t^{(h)}', 'Realized target associated with origin t']],
   'If two future daily log returns are 0.01 and −0.02, their cumulative target is −0.01. The corresponding level ratio is exp(−0.01).',
   'Adjacent five-session targets overlap. That overlap matters when estimating serial dependence in forecast-loss differences.', 'foundations')
eq('sunspot', 'Native Sunspot target', r'y_t^{(h)}=s_{t+h}',
   'The nonfinancial robustness task predicts a future native Sunspot value. It does not turn Sunspot observations into financial returns.',
   [[r's_t', 'Native Sunspot observation'], [r'h', 'One or five daily observations ahead']],
   'If the native value five days after an origin is 42, its h = 5 target is 42, regardless of the value at the origin.',
   'Sunspot and financial errors have different units. They are not pooled into the financial performance summaries.', 'foundations')
eq('level', 'Convert a return forecast into an implied level', r'\widehat P_{t+h}=P_t\exp\!\left(\widehat y_t^{(h)}\right)',
   'Transform a cumulative log-return forecast back to the level scale using the observed level at its origin. This changes presentation, not the fitted model.',
   [[r'P_t', 'Observed origin level'], [r'\widehat y_t^{(h)}', 'Predicted cumulative future log return']],
   'From an origin level of 100, a predicted log return of 0.02 implies a target level of 100 × exp(0.02) ≈ 102.02.',
   'The implied-level plot must not be mistaken for a separately trained price-level model or a forecast made with later prices.', 'foundations')
eq('standardization', 'Training-only feature scaling', r'z_{t,j}=\frac{x_{t,j}-\mu_j^{\mathrm{train}}}{\sigma_j^{\mathrm{train}}}',
   'Fit one mean and scale per feature on the permitted training interval. Reuse those fixed quantities on validation, test, and explanation inputs.',
   [[r'x_{t,j}', 'Raw lagged input'], [r'\mu_j^{\mathrm{train}}', 'Training-feature mean'], [r'\sigma_j^{\mathrm{train}}', 'Training-feature standard deviation']],
   'An input of 12 with training mean 10 and training standard deviation 2 becomes a standardized value of 1.',
   'Using future test observations to estimate the mean or scale would expose preprocessing to information outside the training interval.', 'design')
eq('ols', 'Ordinary least-squares prediction', r'\widehat y=\beta_0+\sum_{j=1}^{4}\beta_jz_j,\qquad \widehat\beta=\arg\min_\beta\sum_i(y_i-\widehat y_i)^2',
   'Choose the intercept and four feature weights to minimize squared training residuals. The prediction is an explicit linear combination of standardized inputs.',
   [[r'\beta_0', 'Intercept'], [r'\beta_j', 'Coefficient of feature j'], [r'z_j', 'Standardized lag feature']],
   'An intercept of 0.001 plus coefficients [0.003, −0.002] applied to inputs [2, 1] produces 0.005: a 0.5% log-return prediction.',
   'A transparent coefficient describes the fitted association conditional on the other inputs. It does not establish an economic causal effect.', 'models')
eq('arima', 'ARIMA family', r'\phi(B)(1-B)^d u_t=c+\theta(B)\varepsilon_t',
   'Combine autoregressive history, differencing, and moving-average innovations in a univariate time-series model. The executed pipeline selects an order by training AIC, then advances its fitted state through observed history.',
   [[r'B', 'Backshift operator: B u_t = u_{t-1}'], [r'p,d,q', 'Autoregressive, differencing, and moving-average orders'], [r'\varepsilon_t', 'Innovation or model disturbance']],
   'For an illustrative AR(1) with coefficient 0.5, zero intercept, and zero future innovation, a last observation of 0.02 gives a next-step conditional mean of 0.01.',
   'This recursion is a teaching example. The thesis selects among seven orders, and its native lag counterfactuals are not SHAP values.', 'models')
eq('aic', 'Akaike Information Criterion', r'\operatorname{AIC}=2k-2\log\widehat L',
   'Compare fitted candidate models by balancing maximized likelihood and parameter count. Within the executed ARIMA candidate set, the smallest finite training AIC is selected.',
   [[r'k', 'Number of estimated parameters'], [r'\widehat L', 'Maximized likelihood for the fitted candidate']],
   'A candidate with k = 3 and log likelihood 100 has AIC 6 − 200 = −194. Among otherwise comparable candidates, smaller AIC is preferred.',
   'AIC is a selection criterion within the fitted candidate comparison, not proof of superior held-out forecasting skill.', 'models')
eq('forest', 'Averaging regression trees', r'\widehat f(x)=\frac{1}{T}\sum_{b=1}^{T}f_b(x)',
   'Random Forest regression averages predictions from randomized constituent trees. The thesis also averages the outputs of its two independently seeded forests.',
   [[r'T', 'Number of trees'], [r'f_b(x)', 'Prediction from tree b']],
   'Three teaching-tree outputs of 0.01, 0.02, and −0.03 average to zero. Each executed forest has 100 trees per seed.',
   'The mean of illustrative tree votes is not a newly fitted thesis forest. More trees do not guarantee an improvement over persistence.', 'models')
eq('boosting', 'Additive boosting stages', r'F_M(x)=F_0(x)+\eta\sum_{m=1}^{M}f_m(x)',
   'Successive trees add corrections to a running predictor. The learning rate scales those additions. This is different from averaging separately trained tree outputs.',
   [[r'F_0', 'Initial predictor'], [r'\eta', 'Learning rate'], [r'M', 'Number of boosting stages']],
   'Starting at 0.01, a learning rate of 0.05 and two corrections −0.10 and 0.20 give 0.01 + 0.05 × 0.10 = 0.015.',
   'The teaching stages illustrate addition. Their correction values are not estimated model weights or new research results.', 'models')
eq('regularization', 'Training loss plus complexity penalty', r'\mathcal L=\sum_i\ell(y_i,\widehat y_i)+\sum_m\Omega(f_m)',
   'A regularized objective balances fit to training targets with penalties on model complexity or weights. The executed boosted-tree configurations include L1 and L2 regularization.',
   [[r'\ell', 'Per-observation training loss'], [r'\Omega', 'Regularization contribution for a constituent tree']],
   'Two ensembles can have the same training error but different objective values if one uses more complex or larger leaf weights.',
   'This is the general objective structure. Exact tree penalties and optimization details belong to the implementation in the relevant library and notebook.', 'models')
eq('lstm', 'LSTM gates and memory', r'\begin{aligned}f_t&=\sigma(W_f[x_t,h_{t-1}]+b_f)\\i_t&=\sigma(W_i[x_t,h_{t-1}]+b_i)\\g_t&=\tanh(W_g[x_t,h_{t-1}]+b_g)\\c_t&=f_t\odot c_{t-1}+i_t\odot g_t\\o_t&=\sigma(W_o[x_t,h_{t-1}]+b_o)\\h_t&=o_t\odot\tanh(c_t)\end{aligned}',
   'Forget and input gates control retained and proposed cell memory. The output gate controls how much transformed memory is exposed as the hidden state.',
   [[r'f_t,i_t,o_t', 'Forget, input, and output gates'], [r'g_t', 'Candidate memory'], [r'c_t,h_t', 'Cell and hidden states'], [r'\odot', 'Elementwise multiplication']],
   'A scalar teaching cell with previous memory 0.8, forget gate 0.7, input gate 0.4, and candidate 0.5 has new memory 0.76. Its hidden output is o × tanh(0.76).',
   'The calculator is a scalar illustration. The executed LSTM uses two recurrent layers, 64 hidden units, and 21-step input sequences.', 'models')
eq('attention', 'Scaled dot-product attention', r'\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V',
   'Query–key similarity scores are normalized into weights that combine value representations. The simplified TFT applies four-head attention after its LSTM encoder.',
   [[r'Q,K,V', 'Query, key, and value matrices'], [r'd_k', 'Key dimension'], [r'\operatorname{softmax}', 'Normalized exponential weighting']],
   'Scores [0, 1] receive softmax weights approximately [0.269, 0.731]. Combining values [10, 20] with those weights gives about 17.31.',
   'An internal attention weight is not automatically a faithful explanation or an economic causal effect.', 'models')
eq('softmax', 'Softmax with teaching temperature', r'w_j=\frac{\exp(a_j/\tau)}{\sum_k\exp(a_k/\tau)}',
   'Transform scores into nonnegative weights that sum to one. A positive temperature controls concentration in the teaching demonstration.',
   [[r'a_j', 'Unnormalized score'], [r'\tau', 'Positive temperature'], [r'w_j', 'Normalized weight']],
   'Reducing temperature concentrates the mass on the largest scores; increasing it makes the weighting more uniform.',
   'The adjustable temperature is a teaching control. It is not an estimated thesis hyperparameter or a reported sensitivity experiment.', 'models')
eq('mse', 'Mean squared error', r'\operatorname{MSE}=\frac1n\sum_{i=1}^{n}(y_i-\widehat y_i)^2',
   'Average squared prediction residuals. Squaring means large errors influence the result more strongly than small errors.',
   [[r'n', 'Number of evaluated targets'], [r'y_i,\widehat y_i', 'Realized target and prediction']],
   'Errors of 1 and 3 have squared errors 1 and 9; their MSE is 5.',
   'MSE uses squared target units. Mixing Sunspot MSE and financial-return MSE would mix different quantities.', 'metrics')
eq('rmse', 'Root mean squared error', r'\operatorname{RMSE}=\sqrt{\frac1n\sum_i(y_i-\widehat y_i)^2}',
   'Take the square root of MSE to express the error summary on the target scale. Lower RMSE means less squared-error loss on the evaluated observations.',
   [[r'y_i-\widehat y_i', 'Prediction residual'], [r'n', 'Evaluation-sample size']],
   'For errors [1, 3], RMSE is √5 ≈ 2.236. A single large miss can move RMSE substantially.',
   'A lower descriptive RMSE does not itself establish statistical significance, economic relevance, or profitability.', 'metrics')
eq('mae', 'Mean absolute error', r'\operatorname{MAE}=\frac1n\sum_i|y_i-\widehat y_i|',
   'Average absolute residual magnitudes without squaring. An additional error unit has the same direct contribution wherever it occurs.',
   [[r'|y_i-\widehat y_i|', 'Absolute prediction error']],
   'Errors [1, 3] have MAE 2, compared with RMSE approximately 2.236.',
   'MAE is a magnitude measure. It does not show how often financial forecast signs were correct.', 'metrics')
eq('r2', 'Out-of-sample coefficient of determination', r'R^2=1-\frac{\sum_i(y_i-\widehat y_i)^2}{\sum_i(y_i-\bar y)^2}',
   'Compare model squared error with a constant equal to the realized evaluation-sample mean. Negative R² means the model has more squared error than that evaluation reference.',
   [[r'\bar y', 'Mean of the realized evaluation targets']],
   'If model squared error is 12 and reference squared error is 10, R² = 1 − 12/10 = −0.2.',
   'The denominator is not the supervised training-target mean. Test-mean R², training-mean forecasts, and persistence are different references.', 'metrics')
eq('da', 'Financial directional accuracy', r'\operatorname{DA}=\frac1n\sum_i\mathbf1\{\operatorname{sign}(y_i)=\operatorname{sign}(\widehat y_i)\}',
   'Count the share of financial targets whose realized sign matches the predicted sign. Zero is treated as its own sign by the implementation.',
   [[r'\mathbf1', 'Indicator equal to one when its condition holds'], [r'\operatorname{sign}', 'Negative, zero, or positive sign']],
   'Three matching signs out of four produce DA = 75%, even if a correctly signed prediction is far too large.',
   'The study does not define DA for Sunspot. Sign accuracy alone does not establish a profitable trading strategy.', 'metrics')
eq('relative', 'Error relative to persistence', r'R_{\mathrm{pers}}=\frac{\operatorname{RMSE}_{\mathrm{model}}}{\operatorname{RMSE}_{\mathrm{persistence}}}',
   'Express model RMSE on the scale of a transparent reference forecast. Values below one indicate a descriptive reduction in RMSE.',
   [[r'R_{\mathrm{pers}}', 'Relative RMSE'], [r'\mathrm{persistence}', 'Zero future return for finance; current native value for Sunspot']],
   'Model RMSE 0.011 divided by persistence RMSE 0.010 gives 1.1: the model has 10% higher RMSE.',
   'A ratio below one is not automatically a statistically significant improvement. The loss comparison requires separate inference.', 'metrics')
eq('shap', 'Additive attribution identity', r'f(x)=\phi_0+\sum_{j=1}^{4}\phi_j',
   'Represent the explained predictor as a baseline plus signed input contributions under the stated explanation construction. Deep coordinate contributions are summed into four lag groups.',
   [[r'\phi_0', 'Explanation baseline or expected output'], [r'\phi_j', 'Signed contribution of feature or lag group j']],
   'A baseline of 0.001 and contributions [0.002, −0.001, 0, 0.003] reconstruct an output of 0.005.',
   'The native ARIMA construction is distinct from SHAP. The identity describes reconstruction, not causal influence.', 'explanations')
eq('additivity', 'Maximum additivity residual', r'r_{\max}=\max_x\left|f(x)-\phi_0-\sum_j\phi_j\right|',
   'Measure the largest numerical discrepancy between the fitted prediction and its reconstructed explanation over the evaluated probes.',
   [[r'r_{\max}', 'Largest absolute reconstruction discrepancy']],
   'A prediction of 0.005 and reconstruction of 0.0049 have an absolute residual of 0.0001 at that probe.',
   'A tiny residual can coexist with unstable rankings, weak intervention behavior, or a lack of economic identification.', 'explanations')
eq('pfi', 'Permutation feature importance', r'\Delta\operatorname{MSE}_j=\operatorname{MSE}(y,f(X^{(j)}))-\operatorname{MSE}(y,f(X))',
   'Shuffle one feature across probes and calculate the resulting change in prediction error. Deep permutation moves whole lag channels between probes while preserving their within-sequence order.',
   [[r'X^{(j)}', 'Inputs with feature or channel j permuted']],
   'If unpermuted MSE is 0.010 and permuting lag_5 yields 0.014, the importance estimate is 0.004 for that sample.',
   'Dependent lags and altered input combinations can affect this intervention and cause disagreement with other importance methods.', 'explanations')
eq('lime', 'Weighted ridge surrogate', r'\widehat g=\arg\min_g\sum_iw_i\bigl(f(z_i)-g(z_i)\bigr)^2+\lambda\|\beta\|_2^2',
   'Approximate a fitted predictor with a simpler weighted regression over perturbed inputs. The executed LIME-style routine also reports hold-out approximation error.',
   [[r'f', 'Explained predictor'], [r'g', 'Simpler surrogate'], [r'w_i', 'Proximity weight'], [r'\lambda', 'Ridge penalty']],
   'A line can approximate a nonlinear curve near one input while being inaccurate elsewhere. Hold-out surrogate MSE checks the approximation on additional perturbations.',
   'The thesis uses its own LIME-style weighted ridge routine with standard-normal perturbations, rather than an external LIME-package run.', 'explanations')
eq('ale', 'Accumulated local effects: theoretical context', r'\operatorname{ALE}_j(x)=\int_{z_0}^{x}\mathbb E\!\left[\frac{\partial f(X)}{\partial x_j}\mid X_j=z\right]dz-C',
   'Accumulate local prediction changes within observed input regions, then center the effect. ALE belongs to the conceptual explanation-method discussion.',
   [[r'C', 'Centering constant'], [r'X_j', 'Feature whose local effect is summarized']],
   'A centered curve can summarize how a fitted prediction changes as one feature moves through regions represented by the data.',
   'The submitted executed exports contain no ALE result curves. The companion does not invent them.', 'explanations')
eq('deletion', 'Intervention-curve area', r'\operatorname{AUC}=\frac1K\sum_{k=0}^{K-1}\frac{L_k+L_{k+1}}2',
   'Use the average trapezoidal area over successive deletion or insertion losses. Compare model-ranked feature orderings with equivalent random group orderings.',
   [[r'L_k', 'Prediction MSE after intervention step k'], [r'K', 'Number of feature-group steps']],
   'An effective deletion ranking raises error faster than random deletion; an effective insertion ranking restores low error faster than random insertion.',
   'The source exports retain areas and success indicators, not every intermediate curve value. Standardized-zero interventions can leave natural input combinations.', 'explanations')
eq('spearman', 'Signed-vector rank correlation', r'\rho=\operatorname{corr}\!\bigl(\operatorname{rank}(a),\operatorname{rank}(b)\bigr)',
   'Compare feature orderings using average ranks for ties. The consecutive-fold diagnostic ranks the signed attribution vectors at the same probe origin.',
   [[r'a,b', 'Four-feature vectors from two fitted folds'], [r'\rho', 'Rank correlation between −1 and 1']],
   'Vectors [1, 2, 3, 4] and [10, 20, 30, 40] have rank correlation 1 despite different scales. Reversing their ordering gives −1.',
   'Four-feature rankings are coarse. A constant vector has undefined rank correlation.', 'stability')
eq('cosine', 'Signed-vector cosine similarity', r'\operatorname{cos}(a,b)=\frac{a^\top b}{\|a\|_2\|b\|_2}',
   'Compare attribution-vector orientation. Multiplication by a positive constant preserves cosine; reversing a vector changes its orientation.',
   [[r'a^\top b', 'Dot product'], [r'\|a\|_2', 'Euclidean norm']],
   'The cosine of [1, 0] and [2, 0] is 1. The cosine of [1, 0] and [0, 1] is 0.',
   'Cosine is undefined for a zero-norm vector and does not separately quantify an overall change in vector magnitude.', 'stability')
eq('top-feature', 'Largest-attribution feature agreement', r'I_{\mathrm{top}}=\mathbf1\!\left\{\arg\max_j|a_j|=\arg\max_j|b_j|\right\}',
   'Check whether two explanation vectors give the greatest absolute attribution to the same feature. This is complementary to comparing all four ranks.',
   [[r'I_{\mathrm{top}}', 'One for a matching top feature, zero otherwise']],
   'Vectors [0.4, −0.1, 0.2, 0] and [0.3, 0.05, −0.1, 0] agree on the first feature even though their other entries differ.',
   'Top-feature agreement discards the remaining ordering and contribution sizes. Ties follow the implementation’s first-index argmax convention.', 'stability')
eq('dm', 'Implemented block-aware HAC–DM statistic', r'\begin{aligned}d_t&=e_{A,t}^2-e_{B,t}^2\\\widehat S&=\widehat\gamma_0+2\sum_{k=1}^{h-1}\widehat\gamma_k\\DM&=\bar d\,/\sqrt{\widehat S/n}\end{aligned}',
   'Use observation-level squared-loss differences and estimate autocovariances within contiguous forecast blocks. The synthesis uses an unweighted lag sum through h−1.',
   [[r'd_t', 'Loss of A minus loss of B'], [r'\widehat\gamma_k', 'Estimated differential autocovariance at lag k'], [r'\bar d', 'Mean loss difference']],
   'A negative mean differential favors A. At horizon one no positive lag enters; at horizon five the sum includes lags one through four.',
   'This is the executed variance approximation, not a Bartlett-weighted Newey–West formula. Nonpositive or invalid variance produces unavailable inference.', 'inference')
eq('bh', 'Benjamini–Hochberg adjustment', r'q_{(i)}=\min\!\left(1,\min_{j\geq i}\frac{m\,p_{(j)}}j\right)',
   'Order valid raw p values, apply the family-size/rank multiplier, and enforce monotone adjusted values from the largest rank backward.',
   [[r'm', 'Number of tests in the declared family'], [r'p_{(j)}', 'jth smallest raw p value'], [r'q_{(i)}', 'Adjusted value at ordered rank i']],
   'The primary matched comparison adjusts 21 model pairs within each dataset–horizon–scheme setting. Native benchmark inference uses a separate 14-test primary family.',
   'Changing the test family changes the adjustment. An exploratory threshold control must not rewrite the submitted q = 0.05 conclusion.', 'inference')

models = [
 dict(id='Linear', title='Linear Regression', intuition='A visible weighted sum of four standardized lagged inputs.',
      mechanics=['Construct lag_1, lag_5, lag_21, and lag_252 on the correct target scale.', 'Fit means and scales using only the fold training interval.', 'Estimate an intercept and coefficients by ordinary least squares.', 'Apply the same fitted mapping out of sample and decompose it with exact linear contributions.'],
      strengths='Compact structure, fast fitting, explicit coefficients, and exact additive reconstruction make the mapping easy to inspect.',
      risks='A linear model may miss nonlinear relationships; correlated predictors also limit the interpretation of an individual coefficient. Neither a coefficient nor an attribution is automatically causal.',
      study='The source uses one seed and six native forecast folds per scheme. Browser OLS reconstruction is checked against all 30,240 native Linear forecasts, with maximum absolute error about 2.6 × 10⁻¹³.',
      equation='ols', activity='linear', resource='https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares'),
 dict(id='ARIMA', title='ARIMA', intuition='Forecast from an explicit statistical model of history and innovations.',
      mechanics=['Fit seven candidate orders and select the smallest finite training AIC.', 'Keep the selected parameters fixed for the fold.', 'Advance the state through all observed raw rows before each forecast origin.', 'Sum the next h return forecasts for finance, or select the hth native forecast for Sunspot.'],
      strengths='The candidate order and fitted statistical structure provide a model-specific way to inspect temporal dependence.',
      risks='Conclusions depend on the candidate set and fitting interval. Selected-lag counterfactual contributions are native model mechanics, not SHAP values or causal explanations.',
      study='Candidates are (1,0,0), (2,0,0), (1,0,1), (2,0,1), (3,0,0), (1,1,0), and (2,1,0). ARIMA has the best mean financial rank, 3.125, in the supplied matched summary.',
      equation='arima', activity='arima', resource='https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html'),
 dict(id='RandomForest', title='Random Forest', intuition='Combine diverse regression trees through averaging.',
      mechanics=['Train 100 randomized trees per seed.', 'Limit maximum depth to 10 and require at least 20 observations in a leaf.', 'Average the trees within each forest, then average the two seeded forests.', 'Average TreeSHAP decompositions across the same two predictor constituents.'],
      strengths='Trees can represent nonlinear relationships and feature interactions without specifying a linear equation for each interaction.',
      risks='Many trees reduce direct transparency. Shared or correlated input information complicates importance interpretation, and randomized fits can yield different attributions.',
      study='The tree pipelines use seeds 20261017 and 20261018. At the study threshold, Random Forest has no significant financial pairwise wins and 15 significant losses.',
      equation='forest', activity='forest', resource='https://scikit-learn.org/stable/modules/ensemble.html#forest'),
 dict(id='XGBoost', title='XGBoost', intuition='Add regularized tree corrections to a running prediction.',
      mechanics=['Reserve a chronological validation tail of at least 50 rows.', 'Fit at most 200 boosting rounds with depth at most 6 and learning rate 0.05.', 'Use row and column sampling 0.8, L1 0.1, L2 1.0, and 25-round early stopping.', 'Average both seeded predictors and their full-ensemble TreeSHAP decompositions.'],
      strengths='Additive tree models can learn flexible nonlinear interactions while using regularization and validation to constrain fitting.',
      risks='Training fit and computational flexibility do not guarantee a held-out advantage. Explanation quality remains conditional on the actual ensemble and its evaluation probes.',
      study='The primary comparison evaluates the same two-seed predictor at the common 924 origins per selected setting. TreeSHAP explains both constituents, rather than one convenient seed.',
      equation='boosting', activity='boosting', resource='https://xgboost.readthedocs.io/en/stable/tutorials/model.html'),
 dict(id='LightGBM', title='LightGBM', intuition='Use histogram-based boosting and best-first leaf growth to build an additive predictor efficiently.',
      mechanics=['Bucket continuous feature values into bins for split calculations.', 'Grow the leaf with greatest loss reduction subject to configured restrictions.', 'Apply the executed depth, regularization, learning-rate, and early-stopping settings.', 'Average the two seeded forecasts and matching TreeSHAP decompositions.'],
      strengths='Efficient split calculations and flexible tree structure permit nonlinear forecasting with explicit fitting limits.',
      risks='Best-first growth can overfit a small sample without suitable restrictions. Efficient training and a good descriptive rank do not establish a significant benchmark gain.',
      study='The recorded settings include 200 maximum rounds, depth 6, and learning rate 0.05. Its mean relative financial RMSE is approximately 1.002912: lowest among the models, but still above persistence on average.',
      equation='boosting', activity='boosting', resource='https://lightgbm.readthedocs.io/en/stable/Features.html'),
 dict(id='LSTM', title='Long Short-Term Memory', intuition='Learn a recurrent memory state that selectively retains and updates sequence information.',
      mechanics=['Construct uninterrupted 21-row histories of four standardized lag channels.', 'Process those histories with two recurrent layers of 64 hidden units.', 'Train with Adam and a chronological validation tail, then restore the best validation state.', 'Apply KernelSHAP to the fitted network and sum signed time-coordinate contributions into four lag groups.'],
      strengths='A learned recurrent state can encode relationships that depend on input sequences rather than isolated tabular rows.',
      risks='The hidden-state computation is not intrinsically transparent. Sequence alignment, training randomness, and explanation sampling affect interpretation and require separate checks.',
      study='One neural seed, four forecast folds per scheme, lookback 21, batch size 64, and at most 25 epochs with patience 8. A 252-row test block contributes 231 evaluated origins after warm-up.',
      equation='lstm', activity='lstm', resource='https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html'),
 dict(id='TFT', title='Simplified Temporal Fusion Transformer', intuition='Reweight lag channels, encode their sequence, and combine temporal representations with attention.',
      mechanics=['Calculate softmax variable-selection weights over four lag channels.', 'Reweight the sequence and encode it with a one-layer 64-unit LSTM.', 'Apply four-head attention and a linear prediction head.', 'Keep native selection/attention signals separate from lag-grouped KernelSHAP.'],
      strengths='Native weighting mechanisms offer additional inspection channels alongside post-hoc decompositions.',
      risks='This is the thesis’s compact adapter, not every component of a full production TFT. Neither attention nor variable selection proves faithful explanation or causal influence.',
      study='The adapter uses lookback 21, four attention heads, one neural seed, and the same 25-epoch cap with patience 8. Native weighting and KernelSHAP remain distinct evidence channels.',
      equation='attention', activity='attention', resource='https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html')
]

terms = []
def term(key, name, definition, example, caution, group='Concept', listed=False):
    terms.append(dict(key=key, name=name, definition=definition, example=example,
                      caution=caution, group=group, listed=listed))

term('AIC', 'Akaike Information Criterion', 'A model-selection criterion combining fitted likelihood and the number of estimated parameters. Smaller values are preferred among the comparable fitted candidates.', 'ARIMA selects the lowest training AIC from seven candidate orders.', 'A lower AIC does not establish better out-of-sample loss than persistence.', 'Abbreviation', True)
term('ARIMA', 'Autoregressive Integrated Moving Average', 'A univariate model family combining dependence on past observations, optional differencing, and past innovations.', 'ARIMA(2,0,1) uses two AR terms, no differencing, and one MA term.', 'The selected order and native lag counterfactuals do not identify an economic mechanism.', 'Abbreviation', True)
term('BH', 'Benjamini–Hochberg', 'A multiple-testing procedure that adjusts a declared family of raw p values to address false discoveries under its assumptions.', 'The primary matched analysis adjusts 21 model pairs within each setting.', 'Different adjustment families answer different testing questions; they are not interchangeable.', 'Abbreviation', True)
term('DA', 'Directional Accuracy', 'The proportion of financial forecasts whose predicted sign matches the realized target’s sign. Zero has its own sign in the implementation.', 'Three matching signs from four targets give DA of 75%.', 'Directional accuracy ignores error magnitude, transaction costs, and portfolio construction.', 'Abbreviation', True)
term('DM', 'Diebold–Mariano', 'A comparison of forecast loss based on the time series of observation-level loss differences between two predictors.', 'Compare squared errors of Linear and TFT at identical origins.', 'A descriptive rank and a significance decision are different claims.', 'Abbreviation', True)
term('HAC', 'Heteroscedasticity and Autocorrelation Consistent', 'A variance-estimation perspective accommodating nonconstant variance and serial dependence. The synthesis uses a block-aware, unweighted autocovariance sum.', 'At horizon five, autocovariances at lags one through four enter the implemented variance estimate.', 'The implemented formula is not a Bartlett-weighted Newey–West estimator.', 'Abbreviation', True)
term('LIME', 'Local Interpretable Model-Agnostic Explanations', 'A family of simpler surrogates used to approximate a complex predictor near an input according to a perturbation and weighting scheme.', 'The notebooks fit weighted ridge surrogates and assess hold-out approximation MSE.', 'The executed routine is LIME-style; standard-normal perturbations may create unrealistic input combinations.', 'Abbreviation', True)
term('LSTM', 'Long Short-Term Memory', 'A recurrent neural network with gated memory and hidden states that process ordered input sequences.', 'The executed LSTM uses a 21-step history and two 64-unit layers.', 'A learned gate is a computational mechanism, not an economic explanation.', 'Abbreviation', True)
term('MAE', 'Mean Absolute Error', 'The average absolute difference between predictions and their realized targets, expressed in target units.', 'Errors of 1 and 3 have MAE 2.', 'MAE values with different target units should not be pooled directly.', 'Abbreviation', True)
term('OLS', 'Ordinary Least Squares', 'Fitting a linear predictor by minimizing the sum of squared training residuals.', 'The Linear model learns an intercept and four standardized-lag coefficients.', 'Transparency of the fitted equation does not make its coefficients causal.', 'Abbreviation', True)
term('PFI', 'Permutation Feature Importance', 'The change in evaluation loss when a feature or lag channel is shuffled across probe observations.', 'An MSE increase from 0.010 to 0.014 gives a PFI estimate of 0.004.', 'Correlated inputs and distribution shifts caused by shuffling can affect interpretation.', 'Abbreviation', True)
term('RMSE', 'Root Mean Squared Error', 'The square root of average squared prediction error. Squaring increases the influence of large misses.', 'Errors of 1 and 3 have RMSE √5, about 2.236.', 'A smaller descriptive value does not by itself demonstrate a significant improvement.', 'Abbreviation', True)
term('SHAP', 'SHapley Additive exPlanations', 'An additive feature-attribution framework whose interpretation depends on the explained predictor, background, and explanation construction.', 'A baseline and four lag contributions reconstruct an explained prediction.', 'Additivity and a plausible ranking do not establish causal influence or universal faithfulness.', 'Abbreviation', True)
term('TFT', 'Temporal Fusion Transformer', 'A forecasting architecture family using feature-selection and temporal mechanisms. This study implements a simplified LSTM-and-attention adapter.', 'The adapter reweights four lag channels and uses four attention heads.', 'Its result should not be described as a test of every component in a full TFT implementation.', 'Abbreviation', True)
term('XAI', 'Explainable Artificial Intelligence', 'Methods for inspecting aspects of model behavior and evaluating how informative or dependable those inspections are.', 'Compare signed attributions, their temporal stability, and intervention diagnostics.', 'An explanation is not a guarantee of forecast skill, human usefulness, or causal validity.', 'Abbreviation', True)
term('ML', 'Machine Learning', 'Learning a predictive mapping from observed examples rather than specifying every decision rule manually.', 'Train a tree ensemble from lagged inputs and future-return targets.', 'A learning algorithm’s flexibility does not replace an appropriate evaluation design.', 'Abbreviation')
term('DL', 'Deep Learning', 'Learning layered neural representations, including recurrent and attention-based sequence transformations.', 'LSTM and the simplified TFT are the two deep comparators.', 'Greater architectural complexity does not ensure lower held-out error.', 'Abbreviation')
term('RF', 'Random Forest', 'An ensemble of randomized decision trees whose numeric outputs are averaged in a regression task.', 'The study averages two seeded forests of 100 trees each.', 'An explanation must cover the complete predictor rather than a selected tree or seed.', 'Abbreviation')
term('MSE', 'Mean Squared Error', 'The average squared prediction residual, used in fitting, forecasting evaluation, and several explanation diagnostics.', 'Residuals [1,3] give MSE 5.', 'Its units are squared target units, unlike RMSE or MAE.', 'Abbreviation')
term('AUC', 'Area Under the Curve', 'An area summary of successive intervention losses in this study’s deletion and insertion diagnostics.', 'Compare model-ranked deletion AUC with the mean from 32 random group orderings.', 'This is not classification ROC AUC.', 'Abbreviation')
term('ALE', 'Accumulated Local Effects', 'A model-agnostic method that summarizes accumulated local prediction changes over observed input regions.', 'A centered effect curve can describe a fitted model’s response to one feature.', 'No executed ALE result curves are supplied in the submitted exports.', 'Abbreviation')
term('TSF', 'Time Series Forecasting file format', 'The supplied Monash file stores the original T1 series used to verify the execution CSV’s aligned values.', 'All 11,109 execution rows align with the source TSF without a value mismatch.', 'The source TSF has a longer date span than the execution subset.', 'Abbreviation')
term('CSV', 'Comma-Separated Values', 'A text format for rows and columns used for the source prediction, metric, attribution, and diagnostic exports.', 'Prediction rows include origin date, target date, realized target, forecast, and benchmarks.', 'Interpret columns according to the sample and target definitions, not just their file names.', 'Abbreviation')
term('R²', 'Coefficient of determination', 'A squared-error comparison with the realized evaluation-sample mean reference.', 'Model squared error 12 versus reference squared error 10 gives R² = −0.2.', 'The denominator is distinct from the persistence and supervised training-mean forecasts.', 'Metric')
term('Log return', 'Logarithmic proportional change', 'The natural logarithm of the ratio of successive adjusted closing levels. It is additive across observations.', 'log(102/100) is about 0.019803.', 'A log return is close to a simple percentage return only for sufficiently small changes.', 'Data')
term('Lag', 'An earlier observation used as an input', 'A lag identifies how far into the past an input comes from relative to its modeled feature row.', 'Financial lag_5 is a past daily log return, not the future five-session target.', 'A timing relationship does not establish an economic causal relationship.', 'Data')
term('Horizon', 'How far ahead the target is realized', 'The number of observations between a forecast origin and its target realization.', 'Financial h = 5 means five trading sessions; Sunspot h = 5 means five days.', 'Changing horizon changes the target and its overlap with adjacent targets.', 'Data')
term('Persistence', 'A transparent reference forecast', 'Predict zero future cumulative return for financial series, or the currently observed native value for Sunspot.', 'A relative RMSE of 1.01 means 1% higher RMSE than persistence.', 'A model can beat another model yet still fail to improve on persistence.', 'Evaluation')
term('Walk-forward', 'Chronological repeated evaluation', 'Fit on earlier observations, evaluate later blocks, and move the information boundary forward through the series.', 'Each training interval precedes a temporal gap and a future test block.', 'Chronological splitting alone does not fix incorrectly aligned features or targets.', 'Evaluation')
term('Rolling window', 'A fixed-size moving training interval', 'Advance both ends of the training interval to retain a limited amount of recent history.', 'The executed rolling fits use the latest 500 available training rows.', 'Discarding older history can increase adaptation while also increasing fit variability.', 'Evaluation')
term('Expanding window', 'A growing training interval', 'Keep the initial training start fixed and add newly available past observations before later folds.', 'The initial training interval contains 500 rows and subsequent intervals grow.', 'A larger sample can include outdated relationships as well as additional information.', 'Evaluation')
term('Purge', 'A pre-test separation component', 'Five frame rows form the first component of the executed gap before a test interval.', 'Five purge rows plus five embargo rows create a ten-row separation.', 'The label describes implemented interval arithmetic, not an independent proof of zero leakage.', 'Evaluation')
term('Embargo', 'The second pre-test separation component', 'Five additional frame rows separate supervised training from testing in this particular implementation.', 'Training ends ten frame rows before the test block begins.', 'Terminology differs across studies; follow the actual code intervals.', 'Evaluation')
term('Leakage', 'Information crossing an impermissible boundary', 'Preprocessing, fitting, or selection uses information that would not be permitted at the relevant forecast origin.', 'Estimating a scaler on the complete series lets future observations influence training transformations.', 'A correct outer split does not automatically validate every inner selection or alignment step.', 'Evaluation')
term('Matched origins', 'Shared evaluation timestamps', 'Forecast origins retained in the intersection of all seven model pipelines for primary comparisons.', 'Each setting has 924 shared origins per model, forming four blocks of 231.', 'Matching coverage does not equalize information sets, tuning effort, or model complexity.', 'Evaluation')
term('Native coverage', 'The original evaluation sample of a pipeline', 'Use every forecast observation exported by a model, before the seven-way intersection is imposed.', 'Static pipelines export 1,512 origins per setting; neural pipelines export 924.', 'Native model–benchmark tests are separate from matched seven-model rankings.', 'Evaluation')
term('Probe', 'An input chosen for explanation evaluation', 'A fixed input origin explained by several refitted models so explanations can be compared without changing the input.', 'Static methods use 40 common probes and neural methods use 30.', 'Explanation probes are not interchangeable with the complete forecast-replay sample.', 'Explainability')
term('Attribution', 'A feature-level contribution or reliance measure', 'A quantity describing how an input feature participates in an explained output under a specified explanation construction.', 'A negative signed contribution lowers the output relative to its explanation baseline.', 'Different channels need not share the same baseline, units, or mathematical interpretation.', 'Explainability')
term('Baseline', 'A context-dependent reference quantity', 'Forecast benchmarks, explanation expected outputs, and intervention replacement values are different kinds of reference.', 'Standardized zero corresponds to a training-feature mean, while financial persistence is zero future return.', 'Name the reference before interpreting a difference or diagnostic.', 'Explainability')
term('Faithfulness', 'Agreement with the explained predictor’s behavior', 'The extent to which an explanation reflects specified computational or intervention properties of its fitted model.', 'An additive explanation may reconstruct a prediction while its ranking fails deletion checks.', 'Being faithful to a model does not mean identifying real economic causes.', 'Explainability')
term('Stability', 'Agreement under a declared perturbation', 'Compare explanations after a specified change, such as a refitted training fold or a different fitting seed.', 'Consecutive-fold rank correlation compares fixed probes across successive fitted models.', 'Temporal, seed, and explanation-sampling stability are different robustness constructs.', 'Explainability')
term('Regime', 'An ex post descriptive stratum', 'A label assigned using the implemented rolling return and volatility rules with full-series quantile thresholds.', 'The crisis rule identifies the highest-volatility stratum and overrides prior labels.', 'These are not independently dated historical crises or real-time trading signals.', 'Evaluation')
term('Causality', 'A claim about effects under intervention', 'An economic causal claim requires an identification design capable of distinguishing a mechanism from association.', 'A large lag_1 attribution does not establish why a market index moved.', 'The lag-only observational design cannot identify genuine causal economic drivers.', 'Interpretation')
term('Black box', 'A predictor difficult to inspect directly', 'A model whose full mapping is hard to understand because of many interacting learned components or latent states.', 'KernelSHAP can inspect an LSTM without replacing its recurrent architecture.', 'Adding a post-hoc explanation does not make an architecture intrinsically interpretable.', 'Interpretation')
term('Early stopping', 'Validation-controlled training duration', 'Monitor a held-out chronological training tail and retain the best recorded fitting state.', 'The neural execution allows at most 25 epochs, with patience eight.', 'The validation tail is distinct from the final test sample; this code adds no extra inner purge or validation-tail refit.', 'Model fitting')
term('Seed', 'A pseudorandom initialization setting', 'Initialize a computational random sequence to repeat a particular randomized path in a compatible environment.', 'The tree pipelines use seeds 20261017 and 20261018.', 'Two seeded fits provide limited replication, not a complete uncertainty analysis.', 'Model fitting')
term('Attention', 'A learned weighted information combination', 'Query–key scores determine weights that combine value representations inside a model.', 'Softmax gives more weight to a larger similarity score.', 'A learned weight is not automatically a faithful explanation or a causal effect.', 'Model fitting')
term('Variable selection', 'Native lag-channel weighting', 'The simplified TFT learns softmax weights that reweight its four input channels before sequence encoding.', 'The four weights are nonnegative and sum to one.', 'These internal weights are evaluated separately from KernelSHAP attributions.', 'Model fitting')
term('Overfitting', 'Learning sample-specific patterns that fail to generalize', 'A predictor captures noise or idiosyncratic structure in the fitting data instead of dependable predictive relationships.', 'Training loss improves while later held-out loss deteriorates.', 'Apparent training fit does not establish generalization.', 'Model fitting')
term('Non-stationarity', 'Changing distributional properties over time', 'Statistical relationships, variability, or other distributional properties need not remain fixed across a series.', 'A mapping fitted in one market period can behave differently in a later period.', 'Descriptive regime labels do not amount to a complete structural-break analysis.', 'Data')
term('Feature dependence', 'Shared statistical information between inputs', 'Predictors can be correlated or related, making independently perturbed combinations unlike naturally observed histories.', 'Changing one lag while fixing all others can create an unusual input sequence.', 'Perturbation and importance findings depend on how that dependence is handled.', 'Explainability')
term('Explanation background', 'The reference sample used by an explainer', 'A set of inputs used to define comparisons or expected outputs for an explanation method.', 'Deep KernelSHAP uses selected training sequences as its background.', 'Attributions can depend on the background; they are not unconditional economic truths.', 'Explainability')
term('Structural break', 'A change in a data-generating relationship', 'A shift in model parameters or distributional behavior that may invalidate a previously fitted relationship.', 'A coefficient linking a predictor to returns could change after a major market transition.', 'The experiment’s ex post regime labels are not independently estimated break dates.', 'Data')
term('Trading profitability', 'Economic outcome of a specified strategy', 'A strategy’s returns depend on portfolio rules, position sizing, timing, costs, and risk, beyond a forecast’s sign or numerical error.', 'A high-DA forecaster can still be unprofitable after costs or large adverse moves.', 'The study does not infer profitability from DA or explanation plots.', 'Interpretation')

lessons = [
 dict(id='foundations', title='From an observation to a forecast target', tag='START HERE', time='8 min', intro='Define the data, timing, and units before interpreting any model output.',
      paragraphs=['The experiment evaluates four daily equity-index series and one nonfinancial Sunspot robustness series. Adjusted financial closing levels are transformed into log returns. Sunspot targets remain native values, so the two kinds of tasks require different unit interpretations.', 'A forecast origin identifies when a prediction is made. The financial target is the sum of future returns over one or five observations; the four lag positions instead identify past inputs. A level trend alone does not demonstrate predictability of future returns.'],
      equations=['log-return','target','sunspot','level'], activity='returns', takeaway='Every predictive statement needs an explicit target, horizon, time origin, and unit.'),
 dict(id='design', title='Time boundaries and fair evaluation', tag='EVALUATION DESIGN', time='10 min', intro='Follow the information available to fitting, preprocessing, testing, and explanation.',
      paragraphs=['The supervised frames remove the 252-lag warm-up and the final rows needed to construct the longest target. Feature scaling uses each training interval. The outer design separates supervised training and testing with five purge rows and five embargo rows.', 'Static models have six native forecast folds per scheme, while neural models have four. Neural sequence construction excludes 21 initial test positions, leaving 231 forecasts from a 252-row block. The primary seven-model comparison retains their common origin intersection: 924 origins in each of 20 settings.', 'Fixed explanation probes form a separate evaluation sample. Holding an input fixed across consecutive fitted folds makes explanation changes interpretable as changes under refitting, rather than merely changes of input. Shared probes and related indices still limit independence.'],
      equations=['standardization','target'], activity='design', takeaway='Matching timestamps equalizes coverage; it does not equalize every model or information-set choice.'),
 dict(id='models', title='Seven models, from weights to attention', tag='MODEL WORKSHOP', time='20 min', intro='Inspect the implemented configurations and the arithmetic behind each model family.',
      paragraphs=['A compact linear equation exposes its mapping directly. Complex ensembles and neural states require additional inspection tools. Post-hoc explanation can clarify aspects of a fitted black box while the architecture remains difficult to interpret.', 'The model workshops distinguish an exact reconstructed Linear predictor from teaching calculations for tree averaging, additive boosting, AR recursion, LSTM gates, and attention. Teaching controls illustrate mechanisms; their outputs are not inserted into the submitted results.'],
      equations=[], activity='models', takeaway='First understand what the predictor computes; then ask what an explanation of that computation can establish.'),
 dict(id='metrics', title='The metric microscope', tag='LIVE CALCULATION', time='10 min', intro='Explore how magnitude, direction, and reference benchmarks can disagree.',
      paragraphs=['RMSE and MAE summarize error magnitude differently. Directional accuracy measures financial sign agreement. R² compares loss with the realized evaluation-mean reference. Relative RMSE compares the model with persistence. These quantities are complementary rather than interchangeable.', 'The teaching lab starts from recorded matched forecasts. Changing a multiplier or additive bias creates a hypothetical transformation while keeping the realized targets fixed. Recomputed metrics describe that edited set of predictions, not a newly fitted research model.'],
      equations=['mse','rmse','mae','r2','da','relative'], activity='metrics', takeaway='Correct signs can coexist with large errors, negative R², and no persistence advantage.'),
 dict(id='explanations', title='What can an explanation prove?', tag='XAI & FAITHFULNESS', time='12 min', intro='Separate numerical reconstruction, intervention behavior, and economic meaning.',
      paragraphs=['The primary channels are exact Linear attributions, native ARIMA lag counterfactuals, complete-ensemble TreeSHAP, and KernelSHAP summed into four lag groups for the neural models. Native TFT selection weights remain a distinct inspection channel.', 'Additivity checks reconstruction of a fitted predictor. Deletion and insertion compare intervention effects with equivalent random feature-group orders. PFI alters inputs through shuffling. The LIME-style routine evaluates a weighted surrogate against model outputs on perturbed inputs. Agreement across these diagnostics is not guaranteed.', 'The available variables are lagged returns or native Sunspot values. Their attribution magnitudes cannot identify a macroeconomic or causal mechanism. ALE belongs to the theoretical discussion, but the provided executed artifacts contain no ALE curve results.'],
      equations=['shap','additivity','pfi','lime','deletion','ale'], activity='shap', takeaway='A numerically valid explanation of a predictor does not establish predictive skill or economic causality.'),
 dict(id='stability', title='Does the explanation survive refitting?', tag='TEMPORAL ROBUSTNESS', time='10 min', intro='Compare fixed-probe vectors and identify what each agreement measure retains or discards.',
      paragraphs=['The temporal evaluation compares signed four-lag attribution vectors at the same probe across consecutive fitted folds. Spearman correlation compares ranks; cosine compares orientation; the top-feature indicator compares the largest absolute contribution.', 'Positive rescaling leaves both Spearman correlation and cosine unchanged even when total attribution magnitude changes greatly. With four features, rankings are coarse; ties and constant vectors require explicit handling. High pooled medians can obscure poor tails or sparse regime strata.', 'Temporal refitting, changing a fitting seed, and resampling an explainer are different perturbations. The limited tree-seed analysis does not substitute for repeated neural fits or repeated fixed-model explanation sampling.'],
      equations=['spearman','cosine','top-feature'], activity='stability', takeaway='State the perturbation, the fixed elements, the diagnostic, and the number of supporting comparisons.'),
 dict(id='inference', title='From loss differences to evidence claims', tag='STATISTICAL INFERENCE', time='12 min', intro='Distinguish descriptive error reductions from adjusted inferential evidence.',
      paragraphs=['The synthesis uses observation-level squared-loss differentials at common origins. Autocovariances remain within contiguous forecast blocks; the implemented variance uses an unweighted sum through horizon h minus one.', 'Seven models create 21 unordered pairs in a selected setting. Benjamini–Hochberg adjustment operates within that family. Supplementary native model–benchmark tests instead use a 14-test primary family and a separately reported two-test sensitivity analysis.', 'An exploratory threshold changes which adjusted values pass a displayed decision rule. It does not rewrite the thesis’s q = 0.05 conclusion. A significance claim also needs the loss direction, coverage definition, and practical effect size.'],
      equations=['dm','bh'], activity='inference', takeaway='Read an adjusted decision together with its family, loss direction, sample, and interpretation limits.'),
 dict(id='glossary', title='Terminology and abbreviations', tag='REFERENCE DESK', time='At your pace', intro='Find an expanded definition, a concrete example, and a caution for each research term.',
      paragraphs=['This reference includes all 15 abbreviations listed in the thesis appendix and additional model, data, evaluation, and interpretation terms used by the public companion. Appendix-listed entries are explicitly marked. Definitions explain how the term is used in this experiment rather than assuming every paper uses identical conventions.'],
      equations=[], activity='glossary', takeaway='Precise terminology keeps a computational result from becoming an unsupported scientific claim.'),
 dict(id='equations', title='The equation atlas', tag='MATHEMATICAL REFERENCE', time='At your pace', intro='Read each formula with its symbols, purpose, worked example, and scope.',
      paragraphs=['The equation atlas distinguishes implemented metrics and model configurations from general theoretical concepts and explicitly illustrative teaching calculations. A mathematical identity establishes a computation, not automatically a forecast advantage or a causal relationship.'],
      equations=[], activity='equations', takeaway='Interpret the formula together with its data, sample, baseline, and scientific question.')
]

quizzes = []
def quiz(id, lesson, question, right, wrong1, wrong2, explanation):
    options = [right, wrong1, wrong2]
    random.Random(id).shuffle(options)
    quizzes.append(dict(id=id, lesson=lesson, question=question, options=options,
                        correct=options.index(right), explanation=explanation))

quiz('q-target','foundations','What does the five-session financial target represent?', 'The sum of the next five daily log returns', 'The fifth lagged price level', 'The mean of the preceding five prices', 'The target is forward-looking from the origin: log(P[t+5]/P[t]). Lag features come from the past.')
quiz('q-sunspot','foundations','Can Sunspot RMSE be pooled directly with financial-return RMSE?', 'No, the tasks use different target units', 'Yes, because both use horizon one or five', 'Yes, after the model names are sorted', 'Sunspot forecasts native values; financial tasks forecast cumulative log returns. The reported financial aggregates exclude Sunspot.')
quiz('q-scaling','design','Which observations determine the feature means and scales?', 'The fold training interval', 'The complete dataset including future test rows', 'The explanation probes alone', 'Training-only standardization respects the information boundary; estimating those quantities on later observations would change that boundary.')
quiz('q-match','design','What does matching forecast origins accomplish?', 'Equalizes evaluation timestamps and coverage', 'Guarantees identical information sets in every model', 'Creates independent repeated experiments', 'The shared intersection provides 924 origins per setting. It does not equalize all pipeline choices or create independent replications.')
quiz('q-complexity','models','Does adding SHAP make an LSTM intrinsically interpretable?', 'No; it adds a post-hoc inspection channel', 'Yes; SHAP removes the hidden states', 'Yes; additivity makes the network architecture linear', 'An explanation inspects the fitted predictor. It does not replace the predictor’s recurrent architecture.')
quiz('q-tft','models','Which TFT description matches the executed notebook?', 'A simplified variable-selection, LSTM, and attention adapter', 'A complete implementation of every production TFT component', 'An OLS model without a temporal encoder', 'The notebook explicitly implements a compact adapter with a 64-unit LSTM encoder and four attention heads.')
quiz('q-forest','models','What is the difference between averaging trees and additive boosting?', 'Averaging combines tree outputs; boosting adds sequential corrections', 'They are exactly the same fitting algorithm', 'Boosting never uses a learning rate', 'The arithmetic distinction helps explain the models, although actual training also includes sampling, optimization, restrictions, and validation choices.')
quiz('q-r2','metrics','What does negative out-of-sample R² mean in this evaluation?', 'Higher squared error than the realized evaluation-mean reference', 'The model forecasts a negative return at every origin', 'The model is guaranteed to lose money in a trading strategy', 'The R² denominator uses the realized evaluation mean. It is not a trading-return measure or the supervised training-mean forecast.')
quiz('q-direction','metrics','Can DA increase while RMSE gets worse?', 'Yes; correct signs and error magnitudes are different properties', 'No; both metrics encode identical information', 'Only for Sunspot, where DA is the primary metric', 'A prediction can have the correct sign but an excessive magnitude. The study does not define DA for Sunspot.')
quiz('q-additivity','explanations','What does a very small additivity residual establish?', 'Numerical reconstruction on the evaluated probes', 'Identification of a causal economic driver', 'Universal faithfulness under every possible intervention', 'The components reconstruct the explained output within numerical tolerance. Stability, behavioral diagnostics, and economic identification remain separate.')
quiz('q-native','explanations','Is an ARIMA native lag counterfactual a SHAP value?', 'No; it uses a distinct native construction', 'Yes; every signed contribution is a SHAP value', 'Yes; because ARIMA is a statistical model', 'The native selected-lag replacement diagnostic is explicitly labeled separately from SHAP-based attribution channels.')
quiz('q-rescale','stability','What does positive rescaling do to cosine against the original nonzero vector?', 'It leaves cosine equal to one', 'It forces cosine to zero', 'Cosine becomes the numerical scale multiplier', 'Positive scaling preserves orientation. Cosine does not separately measure changes in total vector magnitude.')
quiz('q-regime','stability','What do the study’s crisis labels represent?', 'Ex post high-volatility strata under the implemented rule', 'An independently dated historical crisis calendar', 'A validated real-time trading signal', 'The labels use rolling summaries and full-series quantile rules. They are descriptive strata with limited counts in some cells.')
quiz('q-bh','inference','How many pairs form each primary matched BH family?', '21 model pairs', '7 model pairs', '224 model pairs', 'Seven models yield 7 × 6 / 2 = 21 unordered pairs, adjusted within each series–horizon–scheme setting.')
quiz('q-loss','inference','Which model is favored by a negative mean A-minus-B squared-loss differential?', 'Model A', 'Model B', 'Neither, because the sign has no interpretation', 'The differential is A’s loss minus B’s loss. A negative mean indicates lower mean squared loss for A on that evaluation sample.')
quiz('q-native-tests','inference','Can native model–benchmark tests replace the primary matched ranking?', 'No; they retain different model-specific coverage', 'Yes; all native pipelines have identical test lengths', 'Yes; statistical adjustment makes coverage irrelevant', 'Native inference and matched seven-model comparisons preserve different samples and address different comparisons.')

questions = [
 dict(id=1, title='Accuracy–interpretability trade-off',
      question='In financial time series forecasting, to what extent do post-hoc explanation methods (e.g., SHAP, LIME) mitigate the accuracy–interpretability trade-off between intrinsically interpretable models and black-box architectures (e.g., LSTMs, Temporal Fusion Transformers)?',
      answer='Only partially and conditionally. Post-hoc methods make fitted nonlinear predictors easier to inspect, but the executed experiment does not show a general black-box accuracy advantage or a complete resolution of the trade-off.',
      findings=['All seven mean financial RMSE ratios remain above persistence.', 'LightGBM has the lowest mean relative RMSE; ARIMA has the best mean rank.', 'At q = 0.05, Linear records nine significant financial pairwise wins and no significant losses.', 'None of the 224 native financial model–benchmark tests is a significant model win.'],
      boundary='The claim is limited to these fitted pipelines, lagged inputs, periods, and evaluation choices. It is neither a universal ranking nor evidence of trading profitability.',
      source='research/results/cross_model/tables/financial_model_summary.csv', lesson='models'),
 dict(id=2, title='Temporal stability and regime robustness',
      question='To what extent are feature attributions from post-hoc XAI methods temporally stable and robust to non-stationarity, structural breaks, and regime shifts (e.g., bull vs. bear markets, crisis periods) in financial time series?',
      answer='Robustness depends on the model, explanation channel, and training-window scheme. The supplied pooled summaries generally favor expanding windows, while regime comparisons do not reveal a common bear- or crisis-degradation pattern.',
      findings=['Consecutive fitted folds explain the same fixed probe origins.', 'Signed rank correlation, cosine, and absolute top-feature agreement capture different properties.', 'High medians can conceal unfavorable tails and sparse regime groups.', 'Temporal refitting and two-seed tree sensitivity remain separate robustness analyses.'],
      boundary='Regimes are ex post descriptive strata rather than independently dated structural breaks. Shared probes, sparse groups, and limited replication bound generalization.',
      source='research/results/cross_model/tables/stability_by_scheme.csv', lesson='stability'),
 dict(id=3, title='Faithfulness and financial reliability',
      question='To what extent are post-hoc explanations faithful to underlying model mechanics, and do they capture genuine economic drivers rather than spurious correlations?',
      answer='Numerical reconstruction is supported for the implemented attribution channels, but behavioral faithfulness remains mixed. The lag-only observational design cannot identify genuine causal economic drivers.',
      findings=['Small additivity residuals support a specific numerical reconstruction property.', 'Deletion/insertion and permutation/surrogate comparisons supply mixed additional evidence.', 'Native ARIMA counterfactuals and TFT weighting signals are distinct channels.', 'Lag-based reliance does not distinguish an economic mechanism from a predictive or spurious association.'],
      boundary='The study does not establish that every association is spurious. It establishes that its design cannot identify which associations are causal.',
      source='research/results/cross_model/tables/faithfulness.csv', lesson='explanations')
]

payload = dict(lessons=lessons, equations=equations, models=models, terms=terms,
               quizzes=quizzes, questions=questions,
               primary_resources=[
                   dict(label='scikit-learn: ordinary least squares', url='https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares'),
                   dict(label='scikit-learn: randomized tree ensembles', url='https://scikit-learn.org/stable/modules/ensemble.html#forest'),
                   dict(label='statsmodels: ARIMA', url='https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html'),
                   dict(label='XGBoost: additive tree models', url='https://xgboost.readthedocs.io/en/stable/tutorials/model.html'),
                   dict(label='LightGBM: histogram algorithms and leaf-wise growth', url='https://lightgbm.readthedocs.io/en/stable/Features.html'),
                   dict(label='PyTorch: LSTM state equations', url='https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html'),
                   dict(label='PyTorch: multi-head attention', url='https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html')],
               scope='Newly authored public educational companion. The complete thesis manuscript and its full-text transcript are excluded.')

def validate(data):
    ids = {e['id'] for e in data['equations']}
    assert len(ids) == len(data['equations']) >= 26
    assert len(data['models']) == 7 and len(data['terms']) >= 45 and len(data['quizzes']) >= 14
    assert {t['key'] for t in data['terms'] if t['listed']} == {'AIC','ARIMA','BH','DA','DM','HAC','LIME','LSTM','MAE','OLS','PFI','RMSE','SHAP','TFT','XAI'}
    assert {l['id'] for l in data['lessons']} == {'foundations','design','models','metrics','explanations','stability','inference','glossary','equations'}
    for l in data['lessons']: assert set(l['equations']) <= ids
    for m in data['models']: assert m['equation'] in ids
    for q in data['quizzes']: assert len(q['options']) == 3 and 0 <= q['correct'] < 3
    assert {q['correct'] for q in data['quizzes']} == {0,1,2}
    for e in data['equations']:
        assert not any(ord(c)<32 for c in e['tex'])
        for symbol,_ in e['symbols']: assert not any(ord(c)<32 for c in symbol)
    assert all(Path(ROOT/q['source']).is_file() for q in data['questions'])

validate(payload)
(ROOT/'data').mkdir(exist_ok=True)
(ROOT/'data/learning.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',',':'))+'\n')
if __name__ == '__main__':
    from collections import Counter
    print(json.dumps(dict(lessons=len(lessons), equations=len(equations), models=len(models), terms=len(terms), appendix_abbreviations=sum(t['listed'] for t in terms), quizzes=len(quizzes), correct_positions=dict(Counter(q['correct'] for q in quizzes))), indent=2))
