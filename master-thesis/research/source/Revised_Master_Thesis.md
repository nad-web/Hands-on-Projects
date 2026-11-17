% Towards Trustworthy Financial Time Series Forecasting: An Experimental Evaluation of Explainable Machine Learning Models for Stock Price Prediction
% Master’s Thesis

# Abstract

This thesis evaluates whether explainable machine learning can support trustworthy financial time series forecasting when predictive accuracy, interpretability, explanation stability, faithfulness, and economic meaning are assessed separately. It preserves three research questions concerning the accuracy–interpretability trade-off, temporal attribution robustness, and the faithfulness and economic reliability of post-hoc explanations. The executed study covers Linear Regression, ARIMA, Random Forest, XGBoost, LightGBM, LSTM, and a simplified Temporal Fusion Transformer adapter. Four daily financial indices and one normalized Sunspot input are evaluated at one- and five-session horizons using four lagged predictors, rolling and expanding walk-forward schemes, and exported explanation diagnostics.

All 760 exported forecast rows and 70 model–series–horizon summary cells were consolidated and checked against the seven notebooks and result archives. LSTM and TFT occupy the lowest descriptive financial RMSE region, but their test origins and lengths are not matched to those of the static models. Linear Regression records the lowest Sunspot RMSE and the highest pooled five-session directional accuracy. Mean R² is negative in every summary cell, and the exported Diebold–Mariano results are invalid because the implementation uses constant fold-level model losses. No statistically supported seven-model ranking is therefore claimed.

The explanation results are model- and diagnostic-specific. Linear Regression and Random Forest show comparatively high median per-probe attribution stability, whereas boosted-tree and deep channels are less stable or more variable. Numerical local-additivity consistency is supported for the explained Linear, Random Forest, and LightGBM models, but not uniformly for XGBoost. Intervention outcomes and cross-channel agreement are mixed. Deep explanation findings remain exploratory because of defective probe sequences and unequal intervention controls.

The thesis concludes that post-hoc methods mitigate the accuracy–interpretability trade-off only partially and conditionally. Temporal stability is limited and model/channel dependent, with no universal bear- or crisis-regime degradation. Faithfulness is established only in narrow diagnostic-specific respects, while genuine economic drivers cannot be identified from lagged price/value attributions. The contribution is an evidence-audited, limitation-forward evaluation showing that trustworthy forecasting requires distinct evidence for accuracy, transparency, stability, faithfulness, and economic validity rather than a single favorable metric or plausible explanation plot.

## Keywords

explainable artificial intelligence; interpretable machine learning; financial time series forecasting; post-hoc explanations; SHAP; LIME; temporal stability; faithfulness

## Document Preparation Note

The author, institution, degree program, supervisor, declaration text, and official submission date were not supplied and are not invented in this manuscript. They must be completed by the author under the university’s rules before submission. The main-text page-count convention used for final verification is Chapters 1–8 only, excluding preliminary pages, references, and appendices.

# List of Figures

| Exhibit | Title | Page |
|---|---|---:|
| Figure 4.1 | Executed forecasting, explanation, and evidence-audit pipeline | 42 |
| Figure 5.1 | Descriptive financial RMSE by model and horizon | 51 |
| Figure 5.2 | Financial RMSE profiles by executed series, model, and horizon | 52 |
| Figure 5.3 | Financial fold-level R² relative to the realized test-block mean | 53 |
| Figure 5.4 | Sunspot RMSE on the normalized notebook input | 54 |
| Figure 5.5 | Distribution of top-ranked mean absolute SHAP features | 56 |
| Figure 5.6 | Consecutive-fold attribution stability by data-driven regime stratum | 58 |
| Figure 5.7 | SHAP additivity and deletion/insertion faithfulness diagnostics | 60 |

*Note.* Page numbers refer to the rendered final document and are populated after pagination verification.

# List of Tables

| Exhibit | Title | Page |
|---|---|---:|
| Table 4.1 | Verified dataset structure and temporal coverage | 42 |
| Table 4.2 | Executed fold design, forecast coverage, and explanation coverage | 44 |
| Table 4.3 | Executed forecasting and explanation configurations | 45 |
| Table 5.1 | Descriptive predictive summary across four financial indices | 50 |
| Table 5.2 | Series-level financial RMSE means by model and horizon | 52 |
| Table 5.3 | Availability and permitted interpretation of statistical and extended evaluations | 55 |
| Table 5.4 | Median per-probe consecutive-fold attribution correlation by regime stratum | 57 |
| Table 5.5 | SHAP additivity and intervention diagnostics | 59 |
| Table 5.6 | Cross-channel Spearman rank-agreement summaries | 60 |
| Table 6.1 | Answer status and remaining evidentiary gaps for the three research questions | 66 |
| Table A.1 | Verified descriptive forecast metrics for all 70 model–series–horizon cells | 83 |
| Table B.1 | Distribution of top-ranked mean absolute SHAP features | 85 |
| Table B.2 | AIC-selected ARIMA orders across 120 forecast rows | 86 |
| Table B.3 | TFT mean native variable-selection weights | 86 |
| Table B.4 | Random Forest forecast sensitivity to the two executed seeds | 86 |
| Table C.1 | Preserved notebook inventory and executed roles | 87 |
| Table C.2 | SHA-256 checksums of the preserved original notebooks | 88 |
| Table D.1 | Model-archive integrity and manifest verification | 88 |
| Table D.2 | Principal evidence files and their thesis use | 89 |
| Table E.1 | Principal mathematical notation | 90 |
| Table E.2 | Abbreviations used in the thesis | 90 |

*Note.* Page numbers refer to the rendered final document and are populated after pagination verification.

# 1. Introduction

## 1.1 Background and Motivation

### 1.1.1 Machine Learning in Financial Forecasting

Machine learning (ML) and deep learning (DL) now have a central role in quantitative financial research. Gradient-boosted trees, random forests, Long Short-Term Memory (LSTM) networks, and transformer-based architectures are increasingly used for tasks that were once dominated by linear econometric models. These tasks include return prediction, risk-premium measurement, volatility forecasting, and trading-signal generation.

The main empirical motivation is the ability of flexible models to represent nonlinear relationships. In a large study of equity risk premia, Gu et al. (2020) report that tree-based methods and neural networks provide the best out-of-sample predictive and portfolio performance. They attribute much of this gain to nonlinear predictor interactions that linear specifications do not capture. Fischer and Krauss (2018) provide a related example. Their LSTM networks outperformed a random forest, a deep neural network, and logistic regression in out-of-sample directional prediction of S&P 500 constituents from 1992 to 2015. The study also reported strong risk-adjusted returns before transaction costs.

These results, however, sit within a longer and more cautious research record. Financial prediction is difficult because markets are noisy and, in the generally accepted semi-strong form, incorporate publicly available information into prices (Fischer & Krauss, 2018, p. 654). Tsay (2005, pp. 180–183) showed that carefully tuned feed-forward neural networks did not systematically outperform a random walk with drift when forecasting monthly IBM log prices. Forecast errors also varied substantially when the same network was re-estimated.

Later comparisons have reached similarly qualified conclusions. Molnar (2022, pp. 57–58) notes that classical statistical models have often been competitive with, or superior to, ML models in systematic time series comparisons. This applies to comparisons involving LSTM networks and multi-layer neural networks as well. Even the results of Fischer and Krauss (2018) have a temporal qualification: the reported LSTM excess returns were concentrated in the earlier sample period and appeared to have largely disappeared from 2010 onward. Flexible models can therefore extract predictive signal, but their advantage may vary across periods and cannot be assumed in advance.

Benchmarking resources have developed alongside these models. The Monash Time Series Forecasting Archive introduced standardized multi-dataset evaluation across domains (Godahewa et al., 2021). FinTSB more recently systematized financial forecasting evaluation, with attention to temporal leakage, non-stationarity, regime diversity, and market frictions (Hu et al., 2026). These resources support more rigorous comparisons of predictive performance. They do not, by themselves, establish whether the resulting models are interpretable or whether their explanations are dependable.

### 1.1.2 The Black-Box Problem and the Regulatory Context

The flexibility of complex models also makes them difficult to inspect. A black-box model has an internal structure that humans cannot readily comprehend directly. A glass-box model, in contrast, has a mechanism that can in principle be understood through inspection (Biecek & Burzykowski, 2021, pp. 8–10). Random forests, gradient-boosting ensembles, LSTMs, and the Temporal Fusion Transformer (TFT) fall on the black-box side of this distinction. Gu et al. (2020) also identify deep neural networks as among the least transparent methods used in empirical asset pricing. Green (2026, §1.4) takes a more qualified view but still describes deep networks as, at best, “grey boxes” whose learned features require exploration.

Opacity would be less concerning if predictive accuracy were the only relevant criterion. James et al. (2013, pp. 24–26) explain that interpretability generally decreases as flexibility increases. A researcher might therefore choose the most flexible model for stock-price prediction. Yet the same authors caution that less flexible methods can be more accurate because highly flexible methods are vulnerable to overfitting.

Financial practice introduces further requirements. Decisions can involve substantial monetary risk, fiduciary duties, and strict regulatory frameworks. In high-stakes applications, opaque AI systems may create legal, ethical, and financial risks, including regulatory penalties and loss of stakeholder trust (Hoang et al., 2026, p. 193). In credit modeling, interpretability may even constitute a legal requirement (Hull, 2021, p. 135). MiFID II governance and auditability expectations for algorithmic trading, together with the European Union’s Artificial Intelligence Act (EU AI Act), have also increased attention to transparency and traceability. Explainability methods are discussed as one response to these demands (von Zahn et al., 2025).

Explanations also affect the people who use model outputs. Experimental evidence reported by von Zahn et al. (2025) indicates that explanations can improve users’ metaknowledge: the alignment between their confidence and their actual performance. This improvement can support better decisions about when to delegate a task to an AI system. The value of an explanation therefore depends not only on its visual appeal but also on whether it represents the model reliably.

Explainable artificial intelligence (XAI) offers two main responses to the black-box problem (Molnar, 2022, p. 3; Biecek & Burzykowski, 2021, p. 7). The first is to use an intrinsically interpretable model, such as a linear model, a rule-based system, or a small decision tree. Its transparency follows from its structure, although this choice may reduce predictive performance (Biecek & Burzykowski, 2021, p. 7; see Section 2.3.2).

The second response is to retain a black-box model and apply a post-hoc explanation method. Examples include SHapley Additive exPlanations (SHAP), Local Interpretable Model-agnostic Explanations (LIME), permutation feature importance, partial dependence, accumulated local effects (ALE), counterfactual explanations, and gradient-based attribution. Native signals such as global feature importance and attention weights can provide additional diagnostic information (Molnar, 2022, pp. 12–13; Arsenault et al., 2025, pp. 265:3–265:4). This thesis uses SHAP and LIME as established examples of the post-hoc toolbox, not as an exhaustive definition of XAI. Its central concern is whether these methods provide dependable transparency for financial time series models or merely create the appearance of transparency.

### 1.1.3 Properties of Financial Time Series

Several properties of financial time series make both forecasting and explanation difficult. The first is a low signal-to-noise ratio. Daily returns have a mean close to zero and, at most, weak low-order serial correlation. Their predictable component is therefore small relative to their innovation variance (Tsay, 2005, pp. 19, 99). In the illustrative S&P 500 analysis of James et al. (2013, pp. 4–5), yesterday’s market movement offers no simple strategy for predicting today’s return. Even a moderately successful classifier achieves only about 60 percent directional accuracy. Under these conditions, spurious strategies can accumulate, and the precision of an individual discovery test is limited (López de Prado, 2020, p. 106).

Second, financial time series are non-stationary. Strict stationarity means that the joint distribution does not change over time. Weak stationarity instead requires time-invariant mean and autocovariance structures. These concepts underpin much of classical time series analysis and justify using past observations to learn about future observations (Tsay, 2005, p. 25). Asset price levels commonly have a unit root: price has no fixed long-run level, and shocks can have permanent effects. Empirical work therefore often uses transformed series such as log returns,

$$r_t = \ln(P_t) - \ln(P_{t-1}), \tag{1.1}$$

where $P_t$ is the asset price at time $t$ and $r_t$ is the corresponding log return (Tsay, 2005, p. 64). Transformation does not remove all temporal dependence. Financial series still experience bull and bear markets, crisis episodes, structural breaks, and changes in market microstructure. These changes conflict with the independent and identically distributed (i.i.d.) assumption used in many ML evaluation procedures. Benchmarks that address this problem characterize datasets through unit-root tests and explicitly select series with pronounced non-stationarity (Hu et al., 2026).

Third, financial returns exhibit conditional heteroscedasticity. Volatility changes over time and occurs in clusters. It can remain high in some periods and low in others, evolve continuously rather than only through jumps, and respond asymmetrically to positive and negative shocks. The last pattern is known as the leverage effect (Tsay, 2005, pp. 98–99). Return distributions are also heavy-tailed and have high excess kurtosis relative to the normal distribution (Tsay, 2005, p. 19).

These properties weaken the assumption that relationships learned in one period will remain unchanged in another. The same qualification applies to explanations. An attribution estimated in one volatility regime need not describe model behavior in a different regime. Evaluation procedures designed for static i.i.d. data may not reveal this failure.

### 1.1.4 Overfitting and Transparency

Overfitting occurs when a flexible model learns noise alongside signal. Because noise is unpredictable by definition, this behavior reduces the reliability of forecasts on new data (López de Prado, 2020, pp. 7–8). Polynomial classifiers can overfit noisy observations, and neural networks can do so as their capacity increases (Kubat, 2015). Overfitting can affect the training sample directly. It can also affect the test sample when repeated evaluation and selection create selection bias under multiple testing. Strategies may then appear replicable in backtests but fail in implementation (López de Prado, 2020, pp. 8–9, 105). This concern motivates the broader warning that many published empirical-finance discoveries may be false (López de Prado, 2018, 2020, p. 8).

Interpretability alone does not prevent overfitting. Intrinsically interpretable models can also fit noise. Single decision trees are easy to visualize but can be unstable under small changes in the training data (Chan et al., 2022, Chapter 2). Conversely, a heavily regularized black-box model may generalize well. The relevant distinction is therefore not that black-box models always overfit more. It is that their failures are harder to inspect when they learn spurious correlations, exploit leakage, or fail under a new market regime.

When a model’s internal structure is complex, analysts may not know which features influence its decisions or how large those influences are. This makes it difficult to assess whether model behavior is consistent with domain knowledge (Biecek & Burzykowski, 2021, pp. 8–10). Post-hoc explanations promise to improve this transparency and traceability, not to protect against overfitting itself. Whether they fulfill that promise under financial conditions is an empirical question addressed in this thesis.

## 1.2 Problem Statement

ML and DL methods have improved predictive performance in stock-price forecasting and related financial tasks (Fischer & Krauss, 2018; Gu et al., 2020). However, prominent model classes such as gradient boosting, random forests, LSTM networks, and transformers remain difficult for humans to comprehend directly (Molnar, 2022; Biecek & Burzykowski, 2021, pp. 8–10). In financial settings, this opacity creates operational risks and complicates governance under frameworks such as MiFID II and the EU AI Act (Hoang et al., 2026, p. 193; von Zahn et al., 2025).

The statistical environment intensifies the problem. Low signal-to-noise ratios, non-stationarity, and heteroscedasticity allow flexible models to learn spurious historical relationships or lose accuracy when market conditions change (Tsay, 2005; López de Prado, 2018, 2020, pp. 7–9). An intrinsically interpretable model at least allows some of these failures to be inspected. A black-box model conceals them unless a separate explanation procedure is applied.

Post-hoc methods do not automatically resolve this difficulty. Many were developed for static i.i.d. settings, and their outputs are not inherently faithful to the model being explained (Arsenault et al., 2025, p. 265:19). Practitioners therefore face a difficult choice. They can accept the possible performance cost of an intrinsically interpretable model, or use a black-box model whose explanations may be unstable, unfaithful, or economically misleading (Biecek & Burzykowski, 2021, p. 7).

This thesis addresses that choice through a systematic experimental evaluation. It defines trustworthy forecasting as the combination of predictive accuracy with explanations that are understandable, temporally robust, and empirically evaluated for faithfulness. The aim is not to declare one model universally superior. It is to determine what evidence the executed models and explanation methods provide for each dimension of trustworthiness.

## 1.3 Research Gap

Financial forecasting and XAI have developed substantially, but their evaluation practices remain fragmented. Forecasting studies often prioritize predictive performance and give less attention to interpretability (Fischer & Krauss, 2018; Gu et al., 2020). General XAI research focuses more heavily on method development, usually outside the specific statistical conditions of financial time series (Molnar, 2022; Biecek & Burzykowski, 2021). A recent survey finds few rigorous tests of explanation faithfulness and reliability in financial time series forecasting. It also identifies a lack of shared benchmarks and quantitative interpretability metrics (Arsenault et al., 2025, p. 265:30). Three gaps follow from this literature.

**Gap 1 — Limited integration of predictive and explanation evaluation in finance.** Predictive performance and interpretability are commonly assessed separately. Financial forecasting benchmarks increasingly address leakage and realistic market conditions but do not necessarily combine these protocols with systematic XAI evaluation (Hu et al., 2026). At the same time, explanation-quality criteria such as fidelity, stability, comprehensibility, accuracy, and completeness remain difficult to quantify (Arsenault et al., 2025, p. 265:24). No standardized framework covers explanation fidelity, temporal stability under regime shifts, and computational overhead in financial forecasting.

**Gap 2 — Limited evidence on temporal instability and regime shifts.** Financial time series are non-stationary, heteroscedastic, and subject to structural breaks (Tsay, 2005, pp. 64, 98–99). XAI evaluation practice nevertheless relies heavily on static settings and i.i.d. assumptions (Biecek & Burzykowski, 2021). Evidence is limited on whether explanations from LSTMs and transformers remain consistent across bull and bear markets, high- and low-volatility periods, and crises. Applied observations suggest that attribution magnitudes can drift even when broad rankings appear stable (Fourné, 2026, p. 44).

**Gap 3 — Insufficient separation of model faithfulness and economic reliability.** In noisy financial markets, ML models can learn spurious, data-mined relationships (López de Prado, 2018; López de Prado, 2020, p. 106). A mathematically coherent explanation can therefore describe an economically meaningless relationship. Improved predictions do not, on their own, identify economic mechanisms (Gu et al., 2020, p. 2229; see Section 3.2.2). Most interpretability methods describe the model rather than the external world (Molnar, 2022, pp. 28–29). A standardized method for deciding whether highlighted features represent genuine economic drivers remains unavailable.

## 1.4 Research Objectives and Research Questions

The main objective is to evaluate how well explainable ML approaches address predictive accuracy and explanation quality jointly in financial time series forecasting. The study compares intrinsically interpretable models with black-box models supplemented by post-hoc explanations. It has three specific objectives: to quantify the accuracy–interpretability trade-off under walk-forward evaluation with explicit temporal separation; to measure the temporal stability of feature attributions across market regimes; and to assess explanation faithfulness and economic plausibility. These objectives lead to three research questions.

**RQ1 (Accuracy–Interpretability Trade-off and Post-hoc Explanations)**

> In financial time series forecasting, to what extent do post-hoc explanation methods (e.g., SHAP, LIME) mitigate the accuracy–interpretability trade-off between intrinsically interpretable models and black-box architectures (e.g., LSTMs, Temporal Fusion Transformers)?

**RQ2 (Temporal Stability & Market Regime Robustness)**

> To what extent are feature attributions from post-hoc XAI methods temporally stable and robust to non-stationarity, structural breaks, and regime shifts (e.g., bull vs. bear markets, crisis periods) in financial time series?

**RQ3 (Faithfulness & Financial Reliability)**

> To what extent are post-hoc explanations faithful to underlying model mechanics, and do they capture genuine economic drivers rather than spurious correlations?

The questions form a connected sequence. RQ1 asks whether explanations can recover some of the transparency given up when moving toward more flexible models. RQ2 asks whether those explanations remain dependable under the temporal conditions characteristic of financial data. RQ3 asks whether they faithfully represent the model and whether they reveal genuine economic mechanisms. Each question has a corresponding component in the experimental design, results, discussion, and conclusion.

## 1.5 Scope and Delimitation

The executed study is narrower than the initial proposal, and this thesis follows the implementation rather than the planned breadth. The primary domain is daily forecasting for four market indices: the S&P 500, DAX, NASDAQ Composite, and Dow Jones Industrial Average. The executed robustness block contains one daily Sunspot series from the supplied Monash-related materials. Forecast horizons are one and five sessions for the financial series and one and five days for Sunspot. Weekly, monthly, yearly, high-frequency, options, constituent-level, and additional benchmark experiments are outside the executed scope.

The forecasting task uses four lagged predictors: `lag_1`, `lag_5`, `lag_21`, and `lag_252`. Financial targets are cumulative future log returns, while the Sunspot target is the native future value. No macroeconomic, liquidity, rolling-volatility, drawdown, or other exogenous predictors entered the executed models. The model set is fixed to Linear Regression, ARIMA, Random Forest, XGBoost, LightGBM, LSTM, and a simplified Temporal Fusion Transformer adapter. GARCH and other econometric models remain theoretical background only.

The XAI scope comprises exact linear attributions, TreeSHAP, KernelSHAP, a LIME-style local ridge surrogate, permutation feature importance, accumulated local effects where exported, deletion/insertion diagnostics, native ARIMA parameters, and simplified TFT variable-selection and attention signals. These procedures support model-specific diagnostics. They do not provide a comprehensive human-factors, regulatory-compliance, fairness, or causal evaluation. Broader questions of AI governance are discussed only as motivation for reliable explanations.

Several implementation boundaries are part of the study’s scope. ARIMA has a raw/frame index-alignment defect. Deep models have different fold coverage and shorter sequence-constructed test windows. Deep explanation probes are not valid contiguous histories, and their intervention comparison is not like-for-like. The normalized Sunspot input is not numerically traceable to the supplied raw archive values. These conditions restrict inference and are reported explicitly rather than repaired through unauthorized re-execution.

## 1.6 Scientific Contributions

The completed study makes four bounded contributions at the intersection of financial forecasting, machine learning, and explainable AI.

**Contribution 1 — An evidence-audited joint evaluation of predictive and explanation outcomes.** The thesis compares seven executed model families on the same four financial datasets and two horizons, with a separate Sunspot robustness block. It brings forecast errors, directional agreement, R², attribution stability, additivity, intervention diagnostics, and cross-method agreement into one evidence framework. The contribution is not a claim of statistically matched superiority. Differences in fold construction, ARIMA alignment, and test-window length remain visible throughout the interpretation.

**Contribution 2 — Model- and channel-specific evidence on temporal attribution stability.** The study reports per-probe consecutive-fold correlations across rolling and expanding estimation schemes and data-driven bull, bear, boundary, and crisis strata. Linear Regression and Random Forest retain comparatively high exported stability, whereas boosted-tree and deep channels differ across models and regimes. The evidence therefore supports a differentiated account rather than a simple expectation of universal crisis degradation.

**Contribution 3 — A distinction between numerical consistency, intervention faithfulness, and economic meaning.** Near-zero SHAP additivity residuals for Linear Regression, Random Forest, and LightGBM are reported alongside mixed deletion/insertion outcomes and weak cross-channel agreement. XGBoost’s materially nonzero residuals and the deep diagnostics’ comparability defects prevent a blanket faithfulness conclusion. Attributions over lagged prices or values also cannot identify macroeconomic or causal drivers. Economic plausibility therefore remains a separate and largely unanswered construct.

**Contribution 4 — A preserved and traceable experimental package.** The seven original notebooks, exported tables, raw-data snapshots, and checksum manifests were audited without retraining or altering the experiments. All model-archive and raw-data manifest hashes match, and the thesis’s numerical claims are linked to source files and transparent aggregations. This establishes artifact integrity and traceability, not independent computational reproducibility. A complete dependency lock, detailed hardware specification, observation-level predictions, and an independent rerun are unavailable.

## 1.7 Structure of the Thesis

Chapter 2 develops the statistical, forecasting, and explainability foundations required for the study. Chapter 3 synthesizes the authorized literature on financial machine learning, explanation methods, evaluation quality, and economic interpretation. Chapter 4 describes the experiments as actually executed, including the four-lag target construction, walk-forward folds, model configurations, explanation diagnostics, and validity restrictions. Chapter 5 reports the verified predictive, stability, and faithfulness results without converting descriptive rankings into statistical conclusions. Chapter 6 interprets these findings in relation to the three research questions and the reviewed literature. Chapter 7 consolidates the data, methodological, explainability, and external-validity limitations. Chapter 8 provides bounded answers to the research questions and identifies future work that would require a separately authorized experimental phase.

# 2. Theoretical Background

This chapter presents the concepts and formal definitions used in the experimental evaluation. Section 2.1 characterizes financial time series as a data-generating environment — their stylized facts, the resulting limits to predictability, and the non-stationarity that motivates the temporal stability question of RQ2. Section 2.2 introduces the forecasting models benchmarked in this thesis, from classical baselines to deep sequence models, and clarifies the role of overfitting. Section 2.3 formalizes interpretability and the post-hoc explanation toolbox, including Shapley-value-based attribution and local surrogate models alongside the broader family of model-agnostic methods.

Section 2.4 reviews how the quality of explanations themselves can be evaluated, covering faithfulness, stability, and the pitfalls that arise when explanations are computed on dependent, non-stationary financial data. Throughout, terms introduced here (post-hoc explanation methods, intrinsically interpretable models, black-box architectures, feature attribution, faithfulness, stability) are used consistently in the remainder of the thesis. Abbreviations follow Appendix E, Table E.2.

## 2.1 Financial Time Series: Properties and Stylized Facts

Empirical research has identified several recurring properties of asset returns, often called stylized facts. Forecasting models and their explanations must be interpreted in light of these properties (Tsay, 2005, pp. 19–20; Green, 2026, §18.3.1). This section reviews the four clusters of properties most consequential for this thesis: the non-stationarity of price levels and its resolution through returns; the time-varying volatility of returns; the low signal-to-noise ratio that limits predictability; and the prevalence of structural breaks and regime shifts.

### 2.1.1 Stationarity, Unit Roots, and Log Returns

Stationarity is a foundation of classical time series analysis. A time series $\{r_t\}$ is strictly stationary if the joint distribution of any finite collection $(r_{t_1}, \ldots, r_{t_k})$ is identical to that of its time-shifted counterpart $(r_{t_1+\tau}, \ldots, r_{t_k+\tau})$ for all shifts $\tau$. Because this condition is difficult to verify empirically, the weaker notion of weak (or covariance) stationarity is typically assumed, requiring only that the mean and the autocovariances are time-invariant (Tsay, 2005, p. 25; Brockwell & Davis, 2016, Definition 1.4.2). Formally, $\{r_t\}$ is weakly stationary if

$$E[r_t] = \mu, \qquad \mathrm{Cov}(r_t, r_{t-\ell}) = \gamma_\ell, \tag{2.1}$$

where $\mu$ is a constant mean and the autocovariance $\gamma_\ell$ depends only on the lag $\ell$, not on time $t$ (Brockwell & Davis, 2016, Definition 1.4.2; Kirchgässner et al., 2013, pp. 12–14). Weak stationarity matters for forecasting because it supports inference from past observations to future ones: as Tsay (2005, p. 25) notes, weak stationarity enables one to make inferences concerning future observations (e.g., prediction).

Most real time series, however, are non-stationary (Aggarwal, 2015, p. 465), and asset prices are a canonical example: for a price series, non-stationarity arises mainly due to the fact that there is no fixed level for the price (Tsay, 2005, p. 64). A series whose stochastic trend does not revert to any fixed level is called unit-root non-stationary, the best-known example being the random walk,

$$P_t = P_{t-1} + \varepsilon_t, \tag{2.2}$$

where $P_t$ is the price at time $t$ and $\varepsilon_t$ is an unpredictable shock. Under such a model the stock price is not predictable or mean-reverting (Tsay, 2005, p. 64), and shocks have permanent effects because their impact does not decay over time (Tsay, 2005, pp. 66–67). A random walk is a special case of non-stationarity in which the characteristic equation of the underlying autoregressive representation has a unit root (Dixon et al., 2020, Ch. 6, pp. 195–201).

A standard solution is to model transformed series rather than raw prices. Following the conventions of financial econometrics, this thesis works with (continuously compounded) log returns, defined as the first difference of log prices,

$$r_t = \ln P_t - \ln P_{t-1}, \tag{2.3}$$

where $P_t$ denotes the closing price at time $t$. Log returns are approximately stationary, are additive over time, and for daily data differ negligibly from simple net returns (Tsay, 2005, p. 19). The transformation is standard in applied work and is routinely justified by unit-root tests such as the Augmented Dickey–Fuller test, which recent benchmarking practice also uses to characterize the non-stationarity of financial forecasting datasets (Stankiewicz, 2015, p. 15; Hu et al., 2026).

Integrated models of the ARIMA class formalize this idea: an ARIMA$(p,d,q)$ process is one that reduces to a stationary ARMA process after differencing $d$ times, making such models useful for representing data with trend (Brockwell & Davis, 2016, p. 158).

Empirically, daily returns of market indices and individual stocks exhibit high excess kurtosis — fat tails relative to the normal distribution — while the mean of a daily return series is close to zero, and market indices display smaller standard deviations than individual stocks (Tsay, 2005, pp. 19–20; Kirchgässner et al., 2013, pp. 18–19). These distributional properties matter for this thesis in two ways: they constrain the magnitude of the predictable component in returns (Section 2.1.3), and they imply that explanations attributing predictions to features must be interpreted in an environment where extreme observations are systematic rather than aberrant.

### 2.1.2 Volatility Clustering, Heteroscedasticity, and the Leverage Effect

While returns themselves are close to serially uncorrelated, they are far from independent; their dependence is concentrated in the second moments. Tsay (2005, pp. 98–99) summarizes four characteristics of asset volatility: first, volatility clusters, so that high-volatility periods alternate with calm periods; second, volatility evolves continuously in that jumps are rare; third, volatility does not diverge and is often stationary itself; and fourth, volatility reacts asymmetrically to large price increases versus large price drops — the leverage effect.

Volatility clustering has long been documented: Mandelbrot already observed that small (large) shocks tend to be followed by small (large) shocks, generating leptokurtic return distributions (Kirchgässner et al., 2013, p. 281; Tsay, 2005, pp. 105–108). In an empirical illustration with daily DAX returns, Kirchgässner et al. (2013, pp. 281–283) find a kurtosis far above the Gaussian value of three and strongly autocorrelated squared residuals, which contradicts the assumption of a constant, time-invariant variance.

Conditional heteroscedasticity of this kind is the subject of the ARCH/GARCH family of models, which treat the conditional variance as time-dependent (Tsay, 2005, Ch. 3; Dixon et al., 2020, pp. 199–201). In the notation of this thesis, with $r_t$ decomposed as $r_t = \mu_t + a_t$ into a conditional mean $\mu_t$ and an innovation $a_t = \sigma_t \varepsilon_t$, an ARCH-type model specifies the conditional variance $\sigma_t^2$ as a function of past squared innovations; EGARCH-type variants additionally capture the asymmetry induced by large positive and negative returns (Tsay, 2005, pp. 98–99, 124). In line with the delimitation of this thesis, GARCH models enter only as background on the data-generating environment; they are not among the benchmarked models, since the forecasting targets are prices and returns rather than volatility.

Heteroscedasticity is directly relevant to the research questions of this thesis. Regimes of elevated volatility (crisis periods) and calm regimes (bull-market drift) imply that the conditional distribution of returns changes over time even when no structural break in the mean equation occurs. Feature attributions computed globally over a sample that pools such regimes may therefore mask regime-specific model behaviour — a concern formalized in Section 2.4.2 and addressed empirically under RQ2.

### 2.1.3 Low Signal-to-Noise, Random Walks, and Predictability Limits

Financial theory gives several reasons to expect weak return predictability. If investors were risk neutral, returns would be unpredictable and prices would follow martingales; more generally, prices scaled by marginal utility are martingales, and returns can be predictable only to the extent that investors are risk averse and conditional second moments of returns and discount factors vary over time — a more plausible configuration at long horizons than at short ones (Cochrane, 2005, pp. 21–22). Intuitively, if today's price lay far below the expected price tomorrow, investors would buy until the discrepancy vanished, so returns should not be predictable; … returns should be like coin flips (Cochrane, 2005, pp. 21–22).

Empirically, returns of indices in efficient markets behave approximately like a pure random process: standard autocorrelation tests such as Ljung–Box and Breusch–Godfrey typically fail to indicate serial correlation, even though normality is clearly rejected because of excess kurtosis (Kirchgässner et al., 2013, pp. 19–20, Example 1.3). Asset returns accordingly exhibit little to no autocorrelation, and past returns contain little information about future returns (Green, 2026, §18.3.1).

Together, these properties imply a very low signal-to-noise ratio. Prediction tasks on financial time series are notoriously difficult, since noise dominates the series and markets are generally accepted to be efficient in the semi-strong form (Fischer & Krauss, 2018, p. 654; see Section 1.1.1). López de Prado (2020, pp. 10, 106) emphasizes that only a small fraction of the information in financial data is signal, which is systematically suppressed by arbitrage forces, and that in a field like financial economics, where the signal-to-noise ratio is low, false strategies abound. Textbook evidence reinforces the point: for daily S&P 500 data, correlations between lagged returns and today's return are close to zero, and a classifier trained on data up to 2004 achieved a test error rate of 52% in 2005 — worse than random guessing (James et al., 2013, pp. 158, 162).

Tsay's (2005, pp. 180–183) controlled comparison of feed-forward neural networks against a random walk with drift for monthly IBM log returns is the canonical cautionary result: the network's out-of-sample mean squared forecast error was essentially identical to that of a simple AR(1) and the random-walk benchmark, direction-prediction accuracy varied substantially across estimation runs, and across 500 simulation runs the network does not outperform the simple model that assumes a random walk with drift (Tsay, 2005, p. 183).

These limits shape the design and interpretation of the thesis. First, when predictive accuracy differentials between models are small by construction of the problem, the accuracy–interpretability trade-off examined under RQ1 must be evaluated statistically rather than by point estimates alone. Second, where predictability exists, it is more plausibly attributed to time-varying risk premia than to fixed statistical patterns (Cochrane, 2005, pp. 21–22), so explanations that suggest stable, universal drivers of returns warrant scepticism (RQ3).

### 2.1.4 Structural Breaks and Regime Shifts

Beyond smooth variation in volatility, financial time series are subject to discrete structural change. A fundamental difficulty is that a time series represents only one realization of a stochastic process, so only special forms of non-stationarity can be handled — among them deterministic trends and stochastic (unit-root) trends, with deterministic trends generally inappropriate for long-run forecasts (Kirchgässner et al., 2013, pp. 155–157). Structural breaks can also distort statistical inference: if a (trend-)stationary process experiences a structural break, standard unit-root tests lose power asymptotically, tending to indicate a unit root even where none exists, which motivates break-adjusted testing procedures in the tradition of Perron (Kirchgässner et al., 2013, pp. 178–179).

In the nonlinear econometric literature, discrete regimes are modelled explicitly, for example through self-exciting threshold autoregressive (SETAR) models in which the dynamics switch between regimes according to the level of the series itself (Tsay, 2005, pp. 161–173).

The machine learning literature describes a related phenomenon as concept drift: the data-generating process changes over time, so recent observations are typically more informative than older ones, which become stale for analytical purposes (Aggarwal, 2015, p. 393); abrupt drift can be viewed as a change point rather than ordinary noise (Aggarwal, 2015, §12.5). For forecasting practice, regime shifts are destructive in a specific way documented by Dixon et al. (2020, pp. 39–40) in the context of mortgage models across the financial crisis: when the evaluation distribution lies in the future, we cannot draw testing data from the distribution we care about, so standard cross-validation cannot rescue a model whose training distribution no longer applies.

The market regimes considered in this thesis include high- and low-volatility periods, bull and bear markets, and crisis episodes. These regimes are not merely descriptive categories. They can violate the stationarity assumptions under which forecasting models and their explanations are estimated. This motivates the temporal-splits discipline and the regime-conditioned evaluation of explanation stability that RQ2 addresses.

## 2.2 Forecasting Models for Financial Time Series

The research design fixes the model set: baselines (linear regression, ARIMA), tree ensembles (Random Forest, XGBoost, LightGBM), and deep sequence models (LSTM, Temporal Fusion Transformer). This section introduces each class at the level required for the experiments, emphasizing the properties that matter for interpretability analysis. One methodological point is important at the outset. Supervised machine learning treats the data-generating process as unknown and replaces classical in-sample diagnostics with out-of-sample evaluation and explicit management of the bias–variance trade-off (Dixon et al., 2020, pp. 16–17). All flexible models — irrespective of how interpretable they are — can overfit, that is, fit the noise of the training data rather than the underlying process; the expected test error decomposes into variance, squared bias, and irreducible error, and highly flexible learners reduce bias at the cost of variance (James et al., 2013, pp. 34–37; Kubat, 2015, pp. 81–83; Green, 2026, pp. 74–76).

This risk is separate from interpretability: structurally simple models such as decision trees are themselves prone to overfitting and instability (James et al., 2013, p. 320; Chan et al., 2022, pp. 70–76). The central problem with black-box architectures is therefore not necessarily a greater tendency to overfit. Instead, their overfitting and other failure modes are difficult to diagnose from the model structure. Post-hoc explanation methods are intended to reduce this opacity (Biecek & Burzykowski, 2021, pp. 8–10; see Section 1.1.4).

Overfitting on the training set can be mitigated by resampling, regularization, and ensembling, whereas overfitting to the test set arises through repeated model selection on the same evaluation data and requires multiple-testing discipline (López de Prado, 2020, pp. 7–9); the executed protocol and its remaining leakage and comparability limitations are specified in Chapter 4.

### 2.2.1 Baselines: Linear Regression and ARIMA

Linear regression models the target as an affine function of the input features,

$$\hat{r}_{t} = \beta_0 + \sum_{j=1}^{p} \beta_j x_{tj}, \tag{2.4}$$

where $x_{tj}$ is the value of feature $j$ at time $t$ and the coefficients $\beta_j$ are estimated by least squares (James et al., 2013, Ch. 3). Linear regression is the prototypical intrinsically interpretable model: each coefficient can be related directly to the contribution of a feature, which is why structurally restricted models of this kind serve as the interpretable reference point in the trade-off analysis (Molnar, 2022, pp. 3, 10; Biecek & Burzykowski, 2021, pp. 8–10). Its limitations in the financial context are equally well documented: the assumption of normally distributed, homoscedastic errors is routinely violated by heavy-tailed, skewed financial data (Kim et al., 2026, pp. 49–50), and linearity precludes the interaction effects to which the predictive gains of more flexible methods are commonly attributed (Gu et al., 2020, p. 2223).

ARIMA$(p,d,q)$ models are the classical time series baseline. An autoregressive component of order $p$ expresses the current value as a linear combination of its own lags,

$$r_t = \phi_0 + \sum_{i=1}^{p} \phi_i r_{t-i} + a_t, \tag{2.5}$$

where $a_t$ is a white-noise innovation; the moving-average and integration components extend this recursion, and $d$ orders of differencing accommodate non-stationary series, since ARIMA processes reduce to ARMA processes when differenced finitely many times (Brockwell & Davis, 2016, p. 158; Aggarwal, 2015, Ch. 14). An AR($p$) model is stationary only if the roots of its characteristic equation lie outside the unit circle; a unit root yields the random-walk-type non-stationarity discussed in Section 2.1.1 (Dixon et al., 2020, pp. 195–199; Tsay, 2005, p. 64). ARIMA remains a competitive baseline: large-scale forecasting comparisons have found statistical models to match or exceed the accuracy of machine learning models, including LSTMs, on standard benchmarks (Molnar, 2022, p. 58), and the evidence on LSTM versus ARIMA in financial applications is mixed and context-dependent (Mao et al., 2026, pp. 2–3).

### 2.2.2 Tree Ensembles: Random Forest, XGBoost, LightGBM

Single decision trees are highly interpretable — they are even easier to explain than linear regression (James et al., 2013, p. 320) — but they are unstable, sensitive to small perturbations of the training data, and prone to overfitting (Kubat, 2015, Ch. 6; Chan et al., 2022, pp. 70–76). Ensemble methods sacrifice this transparency for accuracy. Bagging averages many trees fitted to bootstrap resamples, exploiting the variance reduction of averaging (James et al., 2013, p. 321); Random Forest additionally decorrelates the trees by considering only a random subset of candidate predictors at each split, typically of size $\sqrt{p}$ (James et al., 2013, p. 324).

Gradient boosting instead builds trees sequentially, each fitted to the residual structure left by its predecessors (James et al., 2013, Ch. 8); XGBoost and LightGBM are regularized, computationally optimized implementations of this principle that have become the standard tabular-data workhorses, and XGBoost in particular is popular in financial time series forecasting (Arsenault et al., 2025, p. 265:20).

The accuracy gain has an interpretability price: once many trees are aggregated, it is no longer possible to represent the resulting statistical learning procedure using a single tree, so bagging improves prediction accuracy at the expense of interpretability (James et al., 2013, p. 323). Ensembles nonetheless retain a weak form of intrinsic interpretability through variable-importance summaries — the average reduction in residual sum of squares or node impurity attributable to each predictor across the trees (James et al., 2013, pp. 323–324, 335). Two cautions attach to such measures and are developed in Section 2.4.3: impurity-based importance is an in-sample quantity, so a variable that appears to be significant for explanatory purposes (in-sample) may be irrelevant for forecasting purposes (out-of-sample) (López de Prado, 2020, p. 80), and variable importance does not have any theoretical basis and as such cannot be used with confidence (Chan et al., 2022, p. 76).

### 2.2.3 Deep Learning: Long Short-Term Memory Networks

Recurrent neural networks (RNNs) maintain a hidden state that is propagated across time steps, allowing information to persist; in a time series forecasting application, outputs may be required at each time stamp to predict the next value of the series (Aggarwal, 2018, Ch. 7). Plain RNNs suffer from vanishing or exploding gradients over long horizons, which gated architectures address. The Long Short-Term Memory (LSTM) network augments the recurrent cell with input, forget, and output gates that regulate the flow of information into, within, and out of a cell state, giving the network fine-grained control over what is remembered (Aggarwal, 2018, §7.5, pp. 292–297; Green, 2026, §7.2.5).

The gated recurrent unit (GRU) is a lighter variant with a smaller parameter footprint (Aggarwal, 2018, pp. 296–297), and in applied work the LSTM has become the default recurrent baseline (Hull, 2021, p. 265).

For financial markets, the LSTM is the canonical black-box architecture in this thesis. Its parameters — gate weights, cell states, recurrent connections — admit no direct mapping to feature effects, and the architecture is described as uninterpretable in the standard literature (Aggarwal, 2018, p. 437). Its forecasting credentials in equity markets rest prominently on Fischer and Krauss (2018, p. 654), who report that LSTM networks outperformed random forests, deep feed-forward networks, and logistic regression in directional prediction of S&P 500 constituents over 1992–2015 — while also documenting that the LSTM's excess returns decayed after approximately 2010, fluctuating around zero after transaction costs.

This decay is itself an illustration of regime dependence (RQ2): edges in financial data are arbitraged away as markets adapt. Hybrid architectures combining ARIMA with LSTM or GRU components, evaluated under walk-forward validation, represent a further strand of applied work (Sert, 2025, pp. 35–52).

### 2.2.4 Attention Mechanisms and Temporal Fusion Transformers

Attention mechanisms equip sequence models with content-dependent weighting of input elements: at each step, the model computes compatibility scores between a query and the keys of candidate inputs and forms a weighted combination of the associated values, so the model makes hard decisions about specific parts of the data to attend to (Aggarwal, 2018, §10.2, p. 421). Self-attention generalizes this idea by letting queries, keys, and values all derive from the same sequence; combined with positional encoding and stacked encoder–decoder blocks, it defines the transformer architecture (Green, 2026, §7.3.10). Because attention weights are recalculated at each time step, the model can in principle adapt to changing temporal relationships, and features deemed irrelevant receive low attention weights that reduce their influence on the output (Habiba et al., 2026, §3.5.3, pp. 81–82).

The Temporal Fusion Transformer (TFT) is the attention-based architecture considered in this thesis. It is representative of a design philosophy that builds interpretability directly into the architecture: variable-selection networks and attention weights are intended to expose which inputs and which time steps drive a prediction, in contrast to purely post-hoc analysis of an opaque model (Habiba et al., 2026, pp. 8–9).

This interpretable-by-design positioning makes the TFT the natural counterpart to the LSTM in the accuracy–interpretability comparison of RQ1. However, whether attention weights constitute reliable explanations is contested; Section 2.3.4 reviews the reliability critique, and the thesis treats attention weights as intrinsic, model-internal attributions whose faithfulness and stability must be evaluated rather than assumed (Arsenault et al., 2025, p. 265:18).

Predictive performance across all models is quantified with root mean squared error (RMSE), mean absolute error (MAE), directional accuracy, and R²; their executed definitions and the limits of the available statistical comparisons are specified in Chapter 4 and are not duplicated here.

## 2.3 Interpretability and Explainable AI

If the models of Section 2.2 differ in how transparently their mechanics can be inspected, the purpose of explainable AI (XAI) is to make the behaviour of the opaque members of that set comprehensible and checkable. This section first disentangles the terminology (Section 2.3.1), then formalizes the intrinsic-versus-post-hoc distinction and the associated trade-off (Section 2.3.2), describes the post-hoc toolbox in breadth — with Shapley-value-based and surrogate-based methods as the representatives examined empirically in this thesis (Section 2.3.3) — and closes with the contested status of attention weights as explanations (Section 2.3.4).

### 2.3.1 Taxonomy: Interpretability, Explainability, and Transparency

The vocabulary of the field is unsettled, and terminological confusion is well documented: the terms interpretable and explainable are frequently used interchangeably even in technical publications (Arsenault et al., 2025, p. 265:2). A regulatory analysis of EU legislation likewise concludes that transparency, traceability, interpretability, and explainability remain ill-defined and inconsistently applied, and proposes to treat them as layered, interdependent dimensions of algorithmic opacity, which itself arises at multiple levels from institutional secrecy to the structural inaccessibility of high-dimensional machine learning systems (Buttaboni & Floridi, 2026, §1).

This thesis adopts the following working conventions. Interpretability is understood as a graded property of a model relative to a human observer: the degree to which a human can understand the cause of a decision, or, in an operational variant, the degree to which a human can consistently predict the model's output (Molnar, 2022, p. 8). There is no conclusive mathematical definition of when a model is interpretable; measurable aspects of interpretability can nonetheless be distinguished into human-based evaluations and function-level, mathematical evaluations (Molnar, 2022, p. 9). A model is a black-box when its structure is too complex for humans to understand — for most people the threshold lies closer to ten than to a hundred free parameters — and a glass-box (or white-box) when it is easy to understand.

The practical consequence of opacity is that one cannot determine which features influenced a prediction and by how much, making it difficult to decide whether the model is consistent with our domain knowledge (Biecek & Burzykowski, 2021, pp. 8–10). Following the survey usage in the financial forecasting domain, an interpretable (transparent) model can be understood by direct inspection, whereas an explainable model is a black-box model paired with a distinct explainability method (Arsenault et al., 2025, pp. 265:3–265:4). Explanations themselves are distinguished by scope into instance-level (local) and dataset-level (global) accounts (Biecek & Burzykowski, 2021, p. 7). The field as a whole covers three strands: inherently interpretable models, modifications of complex models towards interpretability, and post-hoc interpretation methods (Molnar, 2022, p. 3).

A useful normative yardstick is provided by three requirements that any predictive model should ideally fulfil: for every prediction, one should be able to verify the strength of the supporting evidence (prediction's validation), to understand which variables affect the prediction and to what extent (prediction's justification), and to anticipate how the prediction would change if the inputs changed (prediction's speculation) (Biecek & Burzykowski, 2021, pp. 6–7). These requirements anticipate the evaluation criteria of Section 2.4.

### 2.3.2 Intrinsic versus Post-hoc Interpretability and the Accuracy–Interpretability Trade-off

The central tension motivating RQ1 is the claim that flexibility and interpretability trade off against each other: In general, as the flexibility of a method increases, its interpretability decreases (James et al., 2013, p. 25), with subset selection and linear models at the interpretable end and bagging, boosting, and support vector machines at the flexible end. Structurally restricted models such as decision rules or linear models permit component-wise interpretation, whereas well-tuned neural networks and gradient-boosted trees lack any direct mapping from parameters to feature effects (Molnar, 2022, p. 3). Two routes to satisfying explanation requirements follow: use models that are interpretable by design — accepting that the price of transparency may be a reduction in performance — or keep a complex model and explain its predictions with post-hoc tools that approximate or simplify its behaviour (Biecek & Burzykowski, 2021, p. 7).

In finance, the tension is concrete: practitioners report weighing interpretability in the decision whether to use neural networks against statistical methods sometimes even if the latter's predictive accuracy is inferior (Dixon et al., 2020, p. 167), and reviews of XAI in financial decision-making reiterate that highly interpretable models such as decision trees or linear regression often lack the predictive power of more complex models like deep neural networks (Hoang et al., 2026, pp. 206–208).

Three qualifications prevent a naive reading of the trade-off, all of which shape the design of this thesis. First, the trade-off is an empirical regularity, not a law: it is commonly believed that complex ML models always outperform more interpretable models in terms of accuracy … However, there are several examples where interpretable models have proven to be serious competitors, including time series forecasting, where statistical models have systematically outperformed LSTMs and multi-layer perceptrons (Molnar, 2022, pp. 57–58). Second, the trade-off may be reversed by overfitting: even where interpretability is not a concern — the textbook example being stock-price prediction — we will often obtain more accurate predictions using a less flexible method, because highly flexible methods are prone to overfitting (James et al., 2013, p. 26). Third, the Rudin critique, acknowledged across the literature surveyed here, holds that for high-stakes decisions interpretable models should be favoured over explained black boxes, that interpretable models should always be included in benchmarks, and that choosing a black-box model must be justified by a relevant performance increase (Molnar, 2022, p. 3; Arsenault et al., 2025, pp. 265:29–30).

Accordingly, the thesis includes linear and tree baselines by design, and treats the magnitude — not the existence — of the trade-off as the empirical question. A final conceptual subtlety concerns where the boundary lies: a model is usually called intrinsically interpretable when a human can relate an individual component to the real world, but interpreting components of complex models blurs the line between intrinsic interpretability and post-hoc analysis of black boxes (Molnar, 2022, p. 13) — the debate around attention weights (Section 2.3.4) is an instance of this boundary dispute.

### 2.3.3 Post-hoc Explanation Methods

Post-hoc explanation methods analyse an already trained model without altering it. Model-agnostic variants require access only to the prediction function, not to internal parameters, so the same method can be applied to a neural network, a random forest, or a linear model; this modularity also enables comparison of explanations across models (Molnar, 2022, p. 11). Mechanistically, most such methods follow a common pipeline — sampling data, applying an intervention to features, obtaining predictions, and aggregating the results — formalized as the SIPA framework (Molnar, 2022, p. 14).

Table 2.1 summarizes the methods relevant to this thesis; consistent with the framing of the research questions, SHAP and LIME are treated as prominent examples within this broader post-hoc toolbox, not as its exhaustive representatives.

Table 2.1 Overview of post-hoc, model-agnostic explanation methods considered in this thesis

| Method | Scope | Core idea | Key caveat |
|---|---|---|---|
| Partial dependence plot (PDP) | Global | Average prediction as one feature varies, others marginalized | Ignores interactions; extrapolates under dependence |
| Accumulated local effects (ALE) | Global | Accumulate local finite differences of the prediction | Developed for dependent features; may not recover true data-generating structure |
| Permutation feature importance (PFI/MDA) | Global | Increase in loss when a feature is shuffled | Substitution effects under correlated features |
| Shapley values / SHAP | Local (aggregable to global) | Average marginal contribution over all feature coalitions | Independence assumptions; computational cost |
| LIME | Local | Fit an interpretable surrogate on perturbed neighbours | Locality and kernel definition; instability |
| Counterfactual explanations | Local | Minimal feature changes that alter the prediction | Plausibility of the constructed instance |

Sources: Molnar (2022, pp. 12–24); López de Prado (2020, pp. 80–84); Biecek and Burzykowski (2021, pp. 12–13); Arsenault et al. (2025, pp. 265:20–22).

The partial dependence plot describes the average change in the predicted outcome when one or more features are changed while the remaining features are marginalized (Molnar, 2022, p. 15). It has been criticized in the financial context for ignoring interaction effects and for lacking a one-to-one correspondence with regression coefficients, and — because it treats the model as a black box — for providing little theoretical insight into how architecture choices affect interpretability (Dixon et al., 2020, p. 186). Accumulated local effects plots were developed as an alternative especially for the case of dependent features, decomposing the prediction function into an intercept, main effects, and interaction effects, though they need not reproduce the structures of the true data-generating process (Molnar, 2022, pp. 16–17). Permutation feature importance (PFI) — in financial machine learning also known as mean decrease in accuracy (MDA) — is defined as the increase in loss when a feature's values are permuted (Molnar, 2022, p. 20.

López de Prado, 2020, p. 81); evaluated in cross-validated experiments, it assesses importance without strong distributional assumptions and irrespective of model specification, which is why it has been argued to be more insightful and accurate than classical in-sample significance testing (López de Prado, 2020, pp. 80–82, 90–91). Counterfactual explanations describe minimal changes to the features of an instance that would substantially change the prediction; they are contrastive and selective, which makes them attractive candidates for human-oriented explanations (Molnar, 2022, pp. 5, 21). Local methods can be aggregated to global ones: Shapley values aggregated over the data yield global statements about feature importance, effects, and interactions (Molnar, 2022, p. 13).

Shapley-value-based attribution has a particularly clear axiomatic foundation. Shapley values derive from cooperative game theory, where they provide the unique payoff allocation satisfying four axioms — efficiency, dummy, symmetry, and additivity (Molnar, 2022, p. 28); the connection between the characteristic function of a coalitional game and fair division of payoffs is precisely what makes Shapley values interpretable as individual feature contributions to a prediction (Nguyen, 2026, pp. 7–8). Transferred to machine learning, the features are the players, the prediction is the payout, and the attribution of feature $j$ for an instance with prediction function $f$ is

$$\phi_j = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|!\,(M - |S| - 1)!}{M!}\,\big[ f(S \cup \{j\}) - f(S) \big], \tag{2.6}$$

where $\mathcal{F}$ is the set of all $M$ features, $S$ is a coalition excluding feature $j$, and $f(S)$ denotes the model's expected prediction when only the features in $S$ are known (Molnar, 2022, pp. 12–13, 28). The dummy axiom implies that a feature that never changes the prediction receives zero attribution; the efficiency axiom implies that attributions sum to the difference between the individual prediction and the average prediction (Molnar, 2022, p. 28; Gün & Kartal, 2025, p. 112). SHAP (SHapley Additive exPlanations) embeds Shapley values in the class of additive feature attribution methods, in which the explanation is a linear model of simplified inputs,

$$g(x') = \phi_0 + \sum_{j=1}^{M} \phi_j\, x'_j, \tag{2.7}$$

where $x'$ is the simplified (interpretable) representation of instance $x$, $\phi_0$ is the base value (the average prediction), and the $\phi_j$ are the per-feature attributions (Arsenault et al., 2025, p. 265:20; Gün & Kartal, 2025, pp. 111–112). Because Kernel SHAP is model-agnostic but assumes feature independence, while TreeSHAP exploits tree structure for efficient computation, the popularity of XGBoost in financial time series forecasting makes TreeSHAP a natural choice in that domain; overall, SHAP is the most prevalent explainability method in the financial forecasting literature, a popularity attributed to its robust mathematical foundation and ease of use (Arsenault et al., 2025, pp. 265:20, 265:29).

LIME (Local Interpretable Model-agnostic Explanations) uses a different approach: it explains an individual prediction by fitting an interpretable local surrogate — typically a sparse linear model — on perturbed samples in the vicinity of the instance (Molnar, 2022, p. 13; Kraft, 2025, p. 17). Formally, the explanation is obtained by minimizing a locality-weighted fidelity loss plus a complexity penalty,

$$\xi(x) = \arg\min_{g \in G}\; \mathcal{L}(f, g, \pi_x) + \Omega(g), \tag{2.8}$$

where $f$ is the black-box model, $g$ is an interpretable model from the class $G$, $\pi_x$ is a proximity kernel weighting perturbed instances by their distance to the instance $x$ being explained, $\mathcal{L}$ measures how faithfully $g$ reproduces $f$ locally, and $\Omega(g)$ penalizes the complexity of the explanation (Hoang et al., 2026, pp. 198–199). Unlike SHAP, LIME cannot straightforwardly explain a model globally, and in the time series setting it is hard to define the region around the prediction, especially when segmenting and replacing the time series (Arsenault et al., 2025, pp. 265:21–22). Both caveats are consequential for this thesis: the locality kernel and the perturbation scheme are design choices that affect the resulting attributions, and hence their stability and faithfulness (Section 2.4).

### 2.3.4 Attention Weights as Explanations and Their Reliability

Because attention weights are produced inside the model during the forward pass, they are often presented as intrinsic explanations: they are intuitive, model-internal, and in architectures such as the TFT they form part of the advertised interpretability (Habiba et al., 2026, pp. 8–9, 90). Classifying attention as ante-hoc or post-hoc is itself nuanced: the weights are computed within the model (suggesting ante-hoc status), yet the attention distribution may be an unintended by-product of optimization rather than a designed explanation, arguing for a post-hoc reading (Spinnato, 2023, p. 23). In the taxonomy of XAI for time series, attentions are built into the model as an ante-hoc mechanism, whereas attribution methods operate post-hoc on fixed trained models (Theissler et al., 2022).

However, substantial criticisms concern the reliability of attention weights. Empirical comparisons reviewed in the financial forecasting literature find that attention weights contain some useful information about the feature importance, but the information may not be enough to correctly rank the features, that attention-based rankings show low correlation with gradient-based and leave-one-out rankings, and that if a reliable interpretation has to be plausible and faithful, then attention weights would not be reliable (Arsenault et al., 2025, p. 265:18). Reported results diverge across contexts, definitions, and testing methods, so the reliability of the interpretation of complex models must be challenged, and their use must be done with caution (Arsenault et al., 2025, p. 265:18); comparatively, attention weights are not always faithful and vary by attention head (Habiba et al., 2026, p. 90).

These criticisms do not make attention weights useless, but they require empirical evaluation: attention weights from the TFT are analysed as one intrinsic attribution signal among others, subject to the same faithfulness and stability evaluation as post-hoc attributions (RQ2, RQ3), rather than accepted as explanations by construction.

## 2.4 Evaluating Explanations

An explanation is not trustworthy simply because a method can produce one. Explanations must be evaluated along criteria that the literature converges on — comprehensibility, fidelity, accuracy, stability, and completeness — even though these theoretical concepts … are difficult to quantify (Arsenault et al., 2025, p. 265:24). This thesis operationalizes three of them: faithfulness (this section, 2.4.1), stability (2.4.2), and robustness to the known pitfalls of explanation methods on financial data (2.4.3).

Evaluation is difficult because there is no ground truth against which an explanation can be compared, which complicates the study of interpretability itself (Molnar, 2022, p. 9), and because gold-standard time series datasets with annotated discriminative regions do not yet exist (Theissler et al., 2022).

### 2.4.1 Faithfulness

Faithfulness describes how accurate an explanation fits a model's behavior towards the prediction score; the same construct circulates under the names perturbation analysis, ablation study, trustworthiness, or fidelity (Theissler et al., 2022, p. 100715). For surrogate-based explanations, fidelity has a direct quantitative reading: the degree to which the interpretable model reproduces the black-box's outputs — measured as agreement of predicted labels in classification and, for regression, as the mean squared or mean absolute error between surrogate and black-box predictions, a quantity that is admittedly rarely assessed in regression settings (Spinnato, 2023, pp. 38–39). The standard experimental protocol is perturbation-based: an explanation is computed, the features or time points identified as most relevant are altered to non-informative values, and the resulting degradation of the prediction is observed; a large degradation indicates that the explanation identified genuinely load-bearing inputs (Theissler et al., 2022, p. 100715).

Deletion-and insertion-type procedures instantiate this protocol by progressively removing (respectively re-adding) features in order of attributed importance and tracking the prediction curve, with area-based summaries such as AOPCR and ablation thresholds such as APT proposed as quantitative faithfulness measures for time series prediction (Arsenault et al., 2025, p. 265:25).

Two caveats are especially important for financial time series. First, perturbation requires a non-information holding baseline value, and in time series such values are not easy to identify — zero, for instance, may itself carry meaning and alter the semantics of an observation — so perturbation strategies must be chosen deliberately and reported as part of the evaluation design (Theissler et al., 2022, p. 100715; Spinnato, 2023, p. 39). Second, the protocol inherits the tunable parameters of the perturbation scheme, which might lead to different results when being used by different researchers (Theissler et al., 2022).

A further structural caveat distinguishes explanation quality from model quality: an interpretation can only be as good as its underlying model, since an overfit or underfit model distorts apparent feature effects, and different methods answer different questions — Shapley-based measures expose the features a model mechanistically uses, whereas test-set permutation importance reflects what contributes to generalization (Molnar, 2022, pp. 5–6 of the pitfalls chapter).

### 2.4.2 Stability of Explanations

Stability denotes the consistency of an explanation when applied to similar instances (Arsenault et al., 2025, p. 265:24). The temporal dimension is central to this thesis: explanations computed from models retrained on different windows, or queried in different market regimes, should agree if they reflect durable structure rather than sample-specific noise. Empirical studies support this concern. Explanation methods applied to time series models may suffer from instability, lack of consistency, and temporal misalignment, with SHAP and LIME characterized as computationally expensive, unstable over time (Habiba et al., 2026, pp. 89–90). In applied economic modelling, overall variable importance has been found fairly stable across country groups, while the magnitudes of feature impacts still differ over time (Fourné, 2026, p. 44) — a pattern consistent with stable rankings but unstable attributions.

Even for a fixed dataset and model class, different explainers can produce highly variant explanations for the same underlying ML model, suggesting different parts of the input data to be the potential cause for a given prediction (Theissler, 2024, §3.2). And the training process itself is an ignored source of uncertainty, since standard methods quantify only the Monte Carlo approximation error of the explanation given a fixed model (Molnar, 2022, pp. 29–30). Architectural choices compound the problem: for rectified networks, increasing the number of hidden units provably increases a bound on the variance of the model's Jacobian, and hence reduces the stability of sensitivity-based interpretations (Dixon et al., 2020, pp. 173–174).

In non-stationary environments, feature relevance itself drifts — features relevant today (e.g., volatility indicators) may lose significance tomorrow (Habiba et al., 2026, p. 156). Stability therefore determines whether attributions can inform decisions across regimes. It is not merely a desirable presentational property; its quantification under structural breaks and regime shifts is the subject of RQ2.

### 2.4.3 Known Pitfalls: Dependent Features, Extrapolation, and Multiple Comparisons

The literature identifies three recurring pitfalls that directly threaten explanation validity in financial time series. The first concerns feature dependence. Perturbation-based methods — PDP, PFI, LIME, and Shapley-based attributions alike — manipulate features individually; when features are correlated, the intervened data have a different distribution than the training data, and might even lie outside the convex hull of the data and represent physically impossible entities (Molnar, 2022, p. 4). The model is then evaluated in areas of the feature space with few or no observed real data points, where model uncertainty can be expected to be very high, which can produce misleading interpretations (Molnar, 2022, p. 9 of the pitfalls chapter). Financial feature sets — lagged returns, rolling volatility estimates, overlapping technical indicators — are strongly dependent by construction, and financial datasets are highly multicollinear because of shared risk factors (López de Prado, 2020, pp. 76–77).

Conditional variants that resample from conditional rather than marginal distributions mitigate extrapolation but change the estimand itself: in the extreme case of duplicated features, conditional importance approaches zero for both, and it would be a pitfall to conclude that both are irrelevant (Molnar, 2022, pp. 24–25). The financial literature documents the same phenomenon as substitution effects: when two features share predictive information, impurity-based importance is split between them, and permutation-based importance may judge both unimportant even if they are critical; clustering or orthogonalization of features is the proposed, imperfect remedy (López de Prado, 2020, pp. 81, 84–85).

The second pitfall is extrapolation beyond the training distribution. This problem is related to feature dependence but is conceptually distinct: even marginally valid perturbations can place the model in regions where its behaviour is unconstrained by data, so the explanation reports properties of an untested function rather than of the fitted relationship (Molnar, 2022, p. 23). The third pitfall is multiple comparisons. Simultaneously testing the importance of many features at a fixed significance level virtually guarantees false positives — with 50 features tested at $\alpha = 0.05$, the probability of at least one spuriously significant feature is approximately 0.92 under the global null — and remedies such as Bonferroni-type corrections, knockoff procedures, or resampling-based significance calibration exist but are rarely applied (Molnar, 2022, pp. 20–21 of the pitfalls chapter).

The same logic is familiar from data mining practice, where querying the data from enough angles makes a chance finding near-certain (Bruce et al., 2020, pp. 112–115), and from financial backtesting, where repeated trials on the same series inflate apparent performance — selection bias under multiple testing (López de Prado, 2020, pp. 105–107). Explanation evaluation comparing attributions across many features, models, markets, and regimes is exposed to exactly this risk, which the experimental protocol of Chapter 4 addresses.

Sections 2.3 and 2.4 therefore establish two premises for the empirical analysis: post-hoc explanation methods offer a principled, model-agnostic toolbox for opening black-box forecasting models, but their outputs are approximations whose faithfulness, stability, and robustness to dependent financial features cannot be assumed. The literature review in Chapter 3 situates these foundations within empirical evidence on explainable machine learning for financial forecasting, and derives the three research gaps that the experiments of Chapters 4 and 5 address.

# 3. Literature Review

This chapter reviews and synthesizes the literature at the intersection of financial time series forecasting, interpretable machine learning, and explainable artificial intelligence (XAI). Consistent with the delimitation of this thesis, the review covers supervised machine learning (ML) and deep learning (DL) approaches to daily-to-monthly stock price and return forecasting, and treats autoregressive integrated moving average (ARIMA) and generalized autoregressive conditional heteroscedasticity (GARCH) models as classical benchmarks rather than as objects of explanation. The chapter proceeds thematically rather than chronologically. Section 3.1 traces the move from linear benchmarks to ML/DL predictors and the evaluation-rigor literature that accompanies it.

Section 3.2 examines interpretability in financial ML, including the contested accuracy–interpretability trade-off and the economic meaning of feature importance. Section 3.3 reviews XAI methods as applied in finance and in time series analysis more broadly, together with emerging benchmarking standards. Section 3.4 synthesizes evidence on the stability, faithfulness, and economic plausibility of explanations. Section 3.5 consolidates the review into three research gaps and maps them to the research questions of this thesis.

## 3.1 Machine Learning for Financial Market Prediction

### 3.1.1 From Linear Benchmarks to Machine Learning and Deep Learning

The point of departure for any forecasting study on equity data is the near-unpredictability implied by asset-pricing theory. Under risk neutrality, prices follow martingales and returns are unpredictable; more generally, predictability is possible only if investors are risk averse and conditional moments of returns and discount factors vary over time (Cochrane, 2005, pp. 21–22). Empirically, daily returns of market indices and individual stocks exhibit high excess kurtosis and a mean close to zero, so the exploitable signal is small relative to noise (Tsay, 2005, p. 19), and financial series are relatively short, serially correlated, multicollinear and possibly nonstationary (López de Prado, 2020, p. 105).

Consequently, the random walk is the canonical benchmark in forecasting exercises (Hull, 2021, p. 109), and prediction tasks on financial time series are notoriously difficult, owing to their high noise content and generally accepted semi-strong market efficiency (Fischer & Krauss, 2018, p. 654; see Section 1.1.1). Against this baseline, classical linear models such as ARIMA and GARCH remain the reference points that any ML or DL predictor must beat out of sample (Tsay, 2005; Medeiros, 2022).

Large-scale comparative studies report that flexible ML and DL models can indeed outperform linear benchmarks. In a cross-sectional horse race over roughly 30,000 U.S. stocks and 94 characteristics, Gu et al. (2020) find that trees and neural networks achieve the best out-of-sample performance, trace their gains to nonlinear predictor interactions missed by other methods, and report that ML-based long–short portfolios attain an annualized out-of-sample Sharpe ratio of 0.77 against 0.51 for leading regression-based strategies (pp. 2223–2226). For time-series directional prediction, Fischer and Krauss (2018, p. 654) show that long short-term memory (LSTM) networks applied to S&P 500 constituents (1992–2015) outperform memory-free classifiers (random forest, a deep neural network, logistic regression), earning daily returns of 0.46 percent and a Sharpe ratio of 5.8 before transaction costs.

These results motivate the inclusion of black-box architectures — gradient-boosted trees and recurrent or transformer networks — in the model set of this thesis, and they exemplify the shift, described by Dixon et al. (2020, pp. 16–17), from classical fit statistics toward out-of-sample predictive performance as the governing criterion in financial modeling.

The superiority of flexible models is, however, far from uniform, and the same literature contains instructive negative and mixed results. Tsay's (2005, pp. 180–183) Examples 4.7–4.8 remain a canonical cautionary case: for monthly IBM log returns, a 3-2-1 feed-forward network attains an out-of-sample mean squared forecast error (MSFE) of 91.74, essentially identical to an AR(1) model (91.70) and a random walk with drift (91.85), while an 8-4-1 network fails to outperform the random-walk benchmark in direction prediction; moreover, repeated estimations of the same architecture produce MSFE values ranging from 89.46 to 93.65, revealing substantial estimation instability. Similarly, James et al. (2013, p. 162) report that a logistic model for daily S&P 500 direction, evaluated on a strict temporal holdout, attains a test error rate of 52 percent — worse than random guessing.

Reviewing more recent work, Mao et al. (2026, pp. 2–3) document context-dependent findings: LSTMs outperform ARIMA for monthly market indices in some studies, whereas ARIMA is superior for daily cryptocurrency prices in others, and hybrid ARIMA–LSTM constructions attempt to combine both strengths. At the level of forecasting competitions, evidence summarized by Molnar (2022, pp. 57–58) indicates that statistical models can consistently outperform ML approaches, including LSTMs, in time series forecasting tasks. Taken together, these strands suggest that black-box superiority in financial forecasting is real in some large-scale settings but fragile, and that it depends on the horizon, the market, and the evaluation design — a dependence with direct consequences for how the accuracy–interpretability trade-off (Section 3.2) and the stability of explanations (Section 3.4) should be studied.

### 3.1.2 Overfitting, Backtest Overfitting, and Evaluation Rigor

A second strand of the literature concerns the conditions under which reported predictive gains can be trusted at all. López de Prado (2020, pp. 7–9) distinguishes two types of overfitting: training-set overfitting, in which a specification so flexible that it explains not only the signal, but also the noise is chosen, and test-set overfitting, in which repeated evaluation on the same test set induces a selection bias. Two clarifications from this literature are central to the framing of this thesis. First, overfitting risk is independent of interpretability: intrinsically interpretable models such as decision trees and polynomial classifiers overfit noisy training data as readily as neural networks do (Kubat, 2015, pp. 81–83, 114–115; James et al., 2013, p. 320).

The difficulty posed by black-box architectures is thus not excessive overfitting but opaque failure: an interpretation can only be as good as its underlying model (Molnar, 2022, p. 6 of the embedded pitfalls chapter), and a black box offers no direct way to inspect what has been learned. Second, because financial datasets exhibit low signal-to-noise ratios, false discoveries are mechanically abundant in this field (López de Prado, 2020, p. 106; see Section 2.1.3), and the same flexibility that produces genuine gains in one regime can manufacture illusory gains in a backtest.

The backtest-overfitting literature quantifies this danger. López de Prado (2020, p. 105) defines selection bias under multiple testing (SBuMT): it is easy to overfit a backtest by conducting multiple historical simulations and reporting the best, so that most published discoveries in finance are likely false, due to test set overfitting (p. 8). His false strategy theorem shows that the expected maximum Sharpe ratio across trials is strictly positive even when every strategy has a true Sharpe ratio of zero (pp. 108–109), motivating multiplicity-adjusted criteria such as the deflated Sharpe ratio and combinatorial purged cross-validation (pp. 113–124). The same logic appears in the general statistical literature as alpha inflation: with enough queries of the same data, something will emerge as 'significant' just by chance (Bruce et al., 2020, pp. 112–115), and selecting comparisons after inspecting the data constitutes data snooping (Montgomery, 2017, pp. 92–93).

Empirically, Bauer (2025, Chapter 3) reports that models selected by a model confidence set (MCS) procedure can show significantly worse out-of-sample than in-sample performance, i.e., exhibit backtest overfitting even under a disciplined selection protocol. This body of work matters for the present thesis beyond its predictive implications: if reported accuracy differences between intrinsically interpretable models and black-box architectures may partly reflect selection bias, then any evaluation of the accuracy–interpretability trade-off must itself be multiplicity-aware.

In response, a methodological consensus on evaluation rigor has formed. Gu et al. (2020, pp. 2244–2248) employ strict chronological train/validation/test splitting with hyperparameter tuning on the validation period only, comparing models with a modified Diebold–Mariano (DM) test; Medeiros (2022, pp. 111–120) systematizes forecast comparison via the DM test, the Li–Liao–Quaedvlieg test, and the MCS under rolling or expanding estimation windows; and Mao et al. (2026) validate their multi-model comparisons with the Bonferroni–Dunn procedure. For time-ordered data, Dixon et al. (2020, pp. 213–214) prescribe walk-forward (sliding-window) cross-validation to avoid look-ahead bias, and Green (2026, p. 82) warns that hyperparameter tuning is itself a form of learning through which information about the validation data can leak into the model.

The parsimony principle articulated in the forecasting literature — that simpler models are usually preferable out of sample because complexity enlarges the scope for data mining (Diebold, 2007) — supplies the evaluative prior against which flexibility must justify itself. What this literature provides is a mature apparatus for comparing predictive accuracy; what it does not provide is any comparable apparatus for comparing the explanations that models produce. This asymmetry is taken up in Sections 3.3 and 3.4.

## 3.2 Interpretability in Financial Machine Learning

### 3.2.1 The Accuracy–Interpretability Trade-off in Finance

The claim that predictive accuracy and interpretability stand in tension is a standard tenet of statistical learning, which holds that interpretability decreases as a method's flexibility increases (James et al., 2013, p. 25; see Section 2.3.2). Notably, the canonical textbook statement uses stock prediction itself as the example in which interpretability is supposedly dispensable — yet immediately qualifies that even then we will often obtain more accurate predictions using a less flexible method, owing to overfitting in highly flexible methods (James et al., 2013, p. 26). Biecek and Burzykowski (2021, pp. 6–7) formalize the underlying desiderata as three laws of model explanation — validation, justification, and speculation of predictions — and observe that compliance can be achieved either with interpretable-by-design models or with post-hoc approximation tools, where transparency is acknowledged to carry a possible performance cost (p. 7).

Molnar (2022, pp. 3, 10) attributes the tension to structural restrictions: interpretable models constrain the hypothesis space, so when the best solution lies outside that space, flexibility and interpretability diverge. Terminologically, this literature distinguishes black-box from glass-box models, while noting that interpretability itself lacks a conclusive mathematical definition (Biecek & Burzykowski, 2021, pp. 8–10; Molnar, 2022, p. 9).

In finance, the trade-off is not merely technical but institutional. Dixon et al. (2020, p. 167) report that interpretability is a prominent issue for practitioners in deciding whether to use neural networks in favor of other machine learning and statistical methods … sometimes even if the latter's predictive accuracy is inferior, and Hull (2021, p. 135) notes that interpretability can be a legal requirement in credit modeling. Review chapters in recent finance volumes restate the trade-off as a central obstacle: highly interpretable models such as decision trees or linear regression are said to lack the predictive power of more complex models like deep neural networks, with Rudin's argument invoked that interpretable models should be favored in high-stakes decisions (Hoang et al., 2026, p. 207).

Similar formulations appear in Saratchandran (2025, p. 309) and, for a gradient-boosting application with SHapley Additive exPlanations (SHAP), in Dorak and Yakar (2025, pp. 111–112). These accounts converge on the diagnosis but are largely narrative: they assert, rather than measure, the performance cost of transparency in financial applications.

A counter-literature questions whether the trade-off is as steep as commonly assumed. Molnar (2022, pp. 57–58) warns against using an opaque, complex ML model when an interpretable model would have been sufficient, citing benchmark evidence in which simple models achieve the bulk of the predictive power of complex ones, statistical models outperform ML in time series forecasting competitions, and ensembles fail to beat logistic regression in clinical and credit-scoring domains; his design recommendation is that complex models should only be favored if the additional performance gain is both significant and relevant. In a small-sample financial application, Supsermpol et al. (2026, p. 99) likewise find that random forests outperform both logistic regression and more complex alternatives because complex algorithms struggle with overfitting on limited data.

In their survey of XAI for financial time series forecasting, Arsenault et al. (2025, pp. 265:29–265:30) present both positions: the high-stakes argument for intrinsically interpretable models and the rejoinder that matching black-box accuracy with an interpretable design often takes more time and effort, so that a trade-off may hold in certain cases. The disagreement is partly definitional (what counts as interpretable) and partly empirical (which dataset, horizon, and market). For stock price and return forecasting specifically, the magnitude of the trade-off — and whether post-hoc explanation methods can relax it — remains an open, task-dependent empirical question, which this thesis addresses under RQ1.

### 3.2.2 Feature Importance and Economic Meaning

If models are to be interpreted, the dominant currency of interpretation in financial ML is feature importance, and here the literature offers both a methodological program and a set of pointed warnings. López de Prado (2020, pp. 75–77) argues that the classical instrument of variable selection, the p-value, suffers from four flaws — strong distributional assumptions, fragility under multicollinearity, answering the wrong probability question, and in-sample significance that need not imply out-of-sample forecasting value — and demonstrates in a controlled experiment that p-values misrank informative, redundant, and noise features in ways that [this] casts a doubt over decades of empirical research in Finance (p. 75).

As substitutes he develops mean decrease impurity (MDI) and mean decrease accuracy (MDA), the latter sometimes also known as permutation importance (p. 81), which evaluate importance in cross-validated out-of-sample experiments rather than in-sample (pp. 77–82). Crucially, he cautions that MDI is an in-sample measure, so that a variable significant for explanatory purposes in-sample may nonetheless be irrelevant for forecasting out of sample (p. 80; see Section 2.2.2) — an early statement of the faithfulness problem for importance measures.

A second warning concerns correlated features. Because financial predictors share common risk factors, substitution effects arise when two features share predictive information and can bias the results from feature importance methods: MDI halves the importance of duplicate features, while MDA may deem two identical but critical features unimportant (López de Prado, 2020, p. 84); his preferred remedy is clustered feature importance computed on groups of related predictors (pp. 84–85). The same phenomenon is analyzed from the ML side by Molnar (2022, pp. 23–25), who shows that permutation- and perturbation-based importance and effect measures extrapolate into regions of the feature space with little or no data support when features are dependent, creating a tension between staying true to the data (by not extrapolating), and staying true to the model.

Since the lagged returns, rolling volatility measures, and technical indicators used in stock forecasting are strongly dependent by construction, these substitution and extrapolation effects are not edge cases but the expected operating condition for any feature-attribution method applied to financial time series.

The deepest caution is epistemological: importance is a measurement, not a mechanism. Gu et al. (2020, pp. 2258–2259) operationalize variable importance through predictive R² losses from nullifying a predictor and through squared partial derivatives, explicitly framing the interpretive goal as modest; yet they emphasize that these improved predictions are only measurements. The measurements do not tell us about economic mechanisms or equilibria (p. 2229). The econometrics literature concurs that predictive estimators should not be read as structural quantities: direct interpretation of ML estimates stretches the method beyond its intended use — prediction (Chan et al., 2022, §3.3.2), and ensemble variable-importance scores do not have any theoretical basis and as such cannot be used with confidence (Chan et al., 2022, pp. 70–76). From the modeling side, Dixon et al. (2020, pp. 168, 186) add that black-box effect measures such as partial dependence plots ignore interactions and offer no one-to-one correspondence with regression coefficients, limiting the theoretical insight they afford into how architecture choices shape interpretations.

Finally, López de Prado (2020, p. 9) reminds readers that a backtest may insinuate that a strategy is profitable, but [does] not tell us why, because historical simulation is not a controlled experiment. Relative to this literature, the present thesis adopts López de Prado's feature-importance program but extends it in two directions he does not pursue: it evaluates post-hoc attribution methods such as SHAP and Local Interpretable Model-agnostic Explanations (LIME) — which he mentions only in passing (2020, p. 77) — and it tests their temporal stability and faithfulness rather than their ranking properties alone.

## 3.3 Explainable AI for Financial Time Series

### 3.3.1 XAI Methods Applied in Finance

The XAI literature distinguishes two routes to interpretability. Arsenault et al. (2025, pp. 265:2–265:4) separate interpretable models, designed in such a way that users can intuitively understand [their] inner workings, from explainable models, defined as black-box models enhanced with specific explainability methods, such as Shapley additive explanations (SHAP) or Local Interpretable Model-agnostic Explanations (LIME), while cautioning that the two terms are often used interchangeably in the literature. Molnar (2022, p. 3) partitions the field into three strands — inherently interpretable models, interpretability-enhancing modifications of complex models, and post-hoc interpretation methods — and characterizes model-agnostic methods as those that probe the prediction function without access to model internals (p. 11).

Following this taxonomy, and to avoid narrowing the field prematurely, this thesis treats SHAP and LIME as prominent examples within a broader post-hoc explanation toolbox that also includes permutation feature importance (PFI), partial dependence plots (PDP) and accumulated local effects (ALE), and counterfactual explanations (Molnar, 2022, pp. 15–21), alongside model-internal signals such as global feature importance and attention weights. Shapley-based attributions rest on the axiomatic foundation of efficiency, dummy, symmetry, and additivity, for which Shapley values are the unique solution (Molnar, 2022, p. 28; Nguyen, 2026, pp. 7–8).

Applications of these methods in finance are documented in several recent surveys and edited volumes, with SHAP emerging as the de facto standard. Arsenault et al. (2025, p. 265:29) find SHAP to be the most prevalent among explainability methods in financial time series forecasting, a popularity they attribute to its mathematical foundation and ease of use, and they note that TreeSHAP is particularly useful given the popularity of extreme gradient boosting (XGBoost) in this domain (p. 265:20). Hoang et al. (2026, pp. 198–199, 205–206) describe LIME's local-surrogate formulation and SHAP's dual local–global capability, and argue that for stock market prediction such techniques enhance transparency, allowing traders and analysts to validate AI-driven insights; they additionally cover counterfactual explanations as an auditing mechanism (§3.3).

Worked finance applications remain predominantly classification-oriented and cross-sectional: Dorak and Yakar (2025, pp. 101, 111–112) apply SHAP to a gradient-boosting model of dividend propensity for European firms, and Sivasamy (2025, pp. 220–224) reports a credit-default case study in which gradient boosting (AUC 0.92) improves over a logistic baseline (AUC 0.82) with global feature importances attached. In a derivatives-calibration context, Green (2026, pp. 539–540) summarizes evidence that local methods such as LIME, layer-wise relevance propagation (LRP), and DeepLIFT can be less effective for deep networks, and that global game-theoretic methods yield more reliable and consistent insights, while also reporting the caution that local interpretability may not align with financial intuition.

Three limitations of this applied literature are consistent across sources. First, its evidentiary base is thin: the finance-oriented chapters are narrative reviews that assert the benefits of XAI without evaluating explanations empirically (Hoang et al., 2026; Saratchandran, 2025, pp. 306–309), and even the broad technical survey of Habiba et al. (2026, pp. 8–9) concludes that post-hoc tools such as SHAP remain partial, advocating that interpretability be embedded as a first-class architectural constraint (as in the Temporal Fusion Transformer (TFT) or N-BEATS). Second, post-hoc explanations carry an intrinsic fidelity caveat: because the explanation is a separate artifact not tied to the model itself, its fidelity can be questioned (Arsenault et al., 2025, p. 265:19; see Section 1.2).

Third, the methods were largely developed for independent, identically distributed (i.i.d.) tabular data: LIME's locality is hard to define … especially when segmenting and replacing the time series (Arsenault et al., 2025, pp. 265:21–265:22), and model-agnostic techniques in general may not be suitable for … models in which the model output depends also on [a] sequence of previous inputs (Biecek & Burzykowski, 2021, p. 11). Each limitation foreshadows one of the research gaps consolidated in Section 3.5.

### 3.3.2 XAI for Time Series: Classification-Side Evidence

The most systematic treatment of XAI for temporal data comes from the classification side. Theissler et al. (2022, p. 100700) provide the first extensive review of XAI for time series classification, organizing methods into time-point-based, subsequence-based, and instance-based explanations and characterizing them along the dimensions of ante-hoc versus post-hoc, model-agnostic versus model-specific, and local versus global. Two of their findings are directly relevant here. Methodologically, they identify faithfulness analysis as the most prominent and widely used quantitative evaluation, in which the parts of a sample deemed relevant by an explanation are perturbed to non-informative values and the resulting change in prediction quality is observed (Theissler et al., 2022, p. 100715).

However, they emphasize that time series lack an obvious non-informative baseline — zero could have a specific meaning altering the class membership — so perturbation design itself becomes a methodological choice with multiple proposed alternatives. Substantively, they conclude that there is not one best explanation method to use for all models, but rather a heavy dependence on the applied model's approach, and that subsequence- and instance-based methods for multivariate series remain scarce despite their practical relevance (Theissler et al., 2022).

The same review is pointed about the state of evaluation practice. Quantitative assessment is quite frequently done using author-selected examples, showing that the approach is plausible, and where a quantitative criterion is used it is typically predictive accuracy — yet accuracy … can not assess the methods' interpretability (Theissler et al., 2022). The authors call for gold-standard datasets with annotations of the truly discriminative temporal regions, noting that such datasets are not available yet, and observe that perturbation-based evaluation has a number of parameters itself, which might lead to different results when being used by different researchers.

Complementary work deepens these concerns: Theissler (2024, §3.1) argues that for most time series the pure localization of influential inputs is not a sufficient explanation (the non-inherent semantics problem), and Spinnato (2023, pp. 38–39) notes that fidelity — the agreement between a surrogate or explanation and the black-box model — is routinely assessed for classification but rarely assessed for regression, where it could nonetheless be computed as the mean squared or absolute error between surrogate and black-box outputs.

For the present thesis, this classification-side evidence plays a dual role. On the one hand, it supplies evaluation protocols — perturbation-based faithfulness tests, surrogate-fidelity scores — that can be transferred to forecasting, and Spinnato's (2023, pp. 38–39) regression-fidelity formulation makes that transfer explicit. On the other hand, the transfer must be argued rather than assumed: forecasting is a regression task on continuous, overlapping, non-stationary observations, not a classification task on discrete labeled series, and the perturbation-baseline problem identified by Theissler et al. (2022) is arguably aggravated for price and return series, where masked segments, zeroed returns, or interpolated values all encode economic meaning.

Kraft (2025, p. 17) adds a further caution from the clinical domain: approximate SHAP computations may misattribute or miss critical features, and such faithfulness concerns reflect a broader challenge with post-hoc methods rather than a defect of any single algorithm. The classification literature thus demonstrates that quantitative explanation evaluation is feasible while simultaneously showing that the financial forecasting setting remains essentially unserved by it.

### 3.3.3 Benchmarking and Evaluation Standards: FinTSB and Monash

A precondition for cumulative progress in any empirical field is shared benchmarking infrastructure, and two recent contributions define the state of the art for the forecasting side of this thesis. Hu et al. (2026) introduce FinTSB, a benchmark for financial time series forecasting motivated by three diagnosed failures of prior evaluation practice: a diversity gap (failure to cover the full spectrum of stock movement patterns), a standardization deficit (absence of unified assessment protocols that would permit cross-study comparison), and a real-world mismatch (neglect of market structure factors such as transaction costs, which inflates reported performance).

FinTSB characterizes its datasets by Augmented Dickey–Fuller (ADF) tests for unit roots to select series with pronounced non-stationarity and low signal-to-noise, enforces temporal ordering to eliminate data leakage, and evaluates performance consistency across market regimes (Hu et al., 2026). Complementarily, the Monash Time Series Forecasting Archive (Godahewa et al., 2021) provides twenty public datasets spanning domains and frequencies, with baseline results reported across eight error metrics, and demonstrates the potential of global models trained across collections of series.

These benchmarks converge with the evaluation-rigor literature of Section 3.1.2 on the importance of leakage-aware temporal splitting, multi-metric reporting, and regime-aware assessment — evaluation priorities that this thesis examines while disclosing the executed design’s remaining leakage and comparability limitations. Their shared blind spot is equally instructive: neither benchmark contains any provision for evaluating explanations. Predictive quality is standardized; interpretability is not. This observation aligns with the conclusion of Arsenault et al. (2025, p. 265:30) that most XAI models have not been tested with the same benchmarks, leaving the selection among explanation approaches challenging, and with the call of Theissler et al. (2022) for rigorous, quantitative, and automatic evaluation protocols.

The experimental design of this thesis can therefore be described as layering an explainability-evaluation track — stability, faithfulness, and computational overhead metrics — onto the financial and general-purpose forecasting benchmarks that the field has recently consolidated.

## 3.4 Stability, Faithfulness, and Economic Plausibility of Explanations

### 3.4.1 The Explanation Instability and Variance Problem

A growing body of evidence indicates that explanations are not stable objects. Theissler (2024, §3.2) articulates the variance problem: well-trained models of comparable performance should reasonably yield comparable explanations, yet experimental studies show that different XAI methods can produce highly divergent explanations of the same underlying ML model, implicating different parts of the input data as the potential cause of a given prediction (reporting results of Bodria et al., as cited in Theissler, 2024; see Section 2.4.2). For time series models specifically, Habiba et al. (2026, p. 90, Table 3.9) catalog the weaknesses of the main explanation families: attention weights are not always faithful and vary by attention head, saliency maps are sensitive to noise and lack temporal scope, and SHAP/LIME are computationally expensive and unstable over time.

The same authors note that in non-stationary series features relevant today (e.g., volatility indicators) may lose significance tomorrow (Habiba et al., 2026, p. 156, Table 5.2), so that explanation instability in finance is not only a property of the explainer but a plausible consequence of the data-generating process itself.

The literature identifies several distinct mechanisms that generate instability, and financial time series activate all of them. First, feature dependence: when predictors are correlated, perturbation-based methods extrapolate into unrealistic regions of the feature space, producing misleading or fragile attributions (Molnar, 2022, pp. 23–25), and substitution effects split or suppress importance across related features (López de Prado, 2020, p. 84). Second, estimation variance: standard explanation practice treats the trained model as fixed and ignores the uncertainty in the training process itself (Molnar, 2022, pp. 29–30), while Dixon et al. (2020, pp. 173–174) prove for a shallow network class that increasing the number of hidden units increases the variance bound of the model's Jacobian — formally linking model capacity to the instability of sensitivity-based interpretations.

Third, regime shifts: structural breaks undermine unit-root inference and, more broadly, any global summary of a time-varying relationship (Kirchgässner et al., 2013, pp. 178–179), and cross-validation cannot repair models under distribution shift since that distribution comes from the future (Dixon et al., 2020, pp. 39–40). The temporal decay documented at the model level — Fischer and Krauss (2018, p. 654) find LSTM profitability fluctuating around zero after 2010 as the earlier edge was arbitraged away — implies that any attribution extracted from such a model is indexed to a particular market regime.

Direct evidence on attribution stability over time is nonetheless scarce and incidental. In an applied Shapley analysis of a fiscal-risk index, Fourné (2026, p. 44) reports that overall variable importance is fairly stable across country groups, while the magnitudes of individual variables' impacts differ over time; in a small-sample corporate-finance application, Supsermpol et al. (2026, p. 99) observe the relative importance of predictors shifting over time as firms mature.

Both remarks are by-products of studies with other primary aims, and neither quantifies stability formally. To date, no study in the reviewed corpus systematically measures the temporal stability of post-hoc feature attributions for stock price or return forecasting models across regimes such as bull and bear markets or crisis periods. This silence motivates RQ2.

### 3.4.2 Faithfulness versus Plausibility: The Attention-Is-Not-Explanation Debate

Even where explanations are stable, they may fail to reflect the model they purport to describe. The evaluation literature distinguishes faithfulness — how accurately an explanation fits the model's behavior toward its predictions, a notion appearing under the synonyms perturbation analysis, ablation study, trustworthiness, or fidelity (Theissler et al., 2022, p. 100715) — from plausibility, the degree to which an explanation is convincing to humans. Arsenault et al. (2025, p. 265:24) adopt the related criteria of comprehensibility, fidelity, accuracy, stability, and completeness, while cautioning that these theoretical concepts … are difficult to quantify.

Operationalizations exist: perturbation of explanation-identified regions with observation of the prediction change (Theissler et al., 2022), area-over-the-perturbation-curve and ablation-threshold measures for time series feature importance (Arsenault et al., 2025, p. 265:25), and surrogate fidelity computed as the error between surrogate and black-box outputs (Spinnato, 2023, pp. 38–39). Yet Arsenault et al. (2025, p. 265:30) conclude that few studies, if any, rigorously tested the faithfulness and the reliability of their explainability methods in financial time series forecasting, and that the absence of quantitative interpretability metrics marks a domain-wide deficit.

The debate over attention weights as explanations crystallizes the faithfulness–plausibility distinction and bears directly on the transformer architectures included in this thesis. Attention is attractive because it is model-internal and temporally adaptive: weights are recomputed at each time step, allowing the model to track changing temporal relationships (Habiba et al., 2026, pp. 81–82). However, Arsenault et al. (2025, p. 265:18) summarize comparative studies finding that attention-based importance shows a low correlation with the one obtained from other methods, such as gradient-based and leave-one-out methods, concluding that if a reliable interpretation has to be plausible and faithful, then attention weights would not be reliable, and that the reliability of attention depends on the context, the definition and the testing method.

Spinnato (2023, Chapter 4) adds a taxonomic nuance: attention can be regarded as ante-hoc because weights are computed inside the model, yet as post-hoc insofar as the attention distribution may be an unintended consequence of the model's [optimization]. The methodological moral is that model-internality is no guarantee of faithfulness — an intrinsic explanation and a post-hoc explanation must both be tested, not trusted by construction.

More broadly, the plausibility of an explanation is systematically easier to achieve and easier to display than its faithfulness, and the reviewed literature repeatedly warns against conflating the two. Approximate attribution methods can misattribute or miss critical features in safety-critical settings (Kraft, 2025, p. 17); post-hoc techniques may not fully capture the complex interactions within deep models (Saratchandran, 2025, p. 309); and partial-dependence-style black-box measures obscure how architecture choices shape the interpretation itself (Dixon et al., 2020, p. 186). Underlying these specific findings is a structural obstacle: for IML, there is no ground truth to which an explanation can be compared to (Molnar, 2022, p. 9), which complicates any claim that an explanation is correct rather than merely agreeable.

Theissler (2024, §3.4) reports that user-based evaluations are avoided in the majority of XAI tool papers, so even plausibility is rarely measured. For financial forecasting, where an unfaithful but plausible attribution can misallocate capital or mask model failure, this gap between the two properties is precisely what RQ3's faithfulness evaluation addresses.

### 3.4.3 Statistical Drivers versus Genuine Economic Drivers

The final strand of the review asks whether explanations, even faithful ones, identify economically meaningful structure. The baseline position of the empirical asset-pricing literature is restrictive: machine learning improves prediction, but predictive measurements do not by themselves identify economic mechanisms or equilibria, and recovering mechanisms requires the researcher to add structure — to build a hypothesized mechanism into the estimation problem (Gu et al., 2020, p. 2229). The interpretable-ML literature reaches the analogous conclusion from the other direction: most interpretation methods describe the model rather than the real world, and inferring real-world effects from them requires additional strong assumptions such as model unbiasedness (Molnar, 2022, pp. 28–29).

Both statements imply that a faithful attribution is, at best, evidence about the model's mechanics; the step from model mechanics to economic drivers is a separate inferential act that the explanation method itself cannot legitimate.

Financial data make this step especially hazardous. Where the signal-to-noise ratio is low and multiple tests are run, spurious findings are mechanically generated (López de Prado, 2020, p. 106; Bruce et al., 2020, pp. 112–115), and in-sample significance does not imply out-of-sample forecasting value (López de Prado, 2020, p. 75). Asset-pricing theory adds a subtler ambiguity: return predictability, where genuine, may reflect time-varying risk premia rather than exploitable inefficiency (Cochrane, 2005, pp. 21–22), so a feature that an attribution method flags as predictive — say, a volatility indicator — may encode compensation for risk, transient mispricing, or pure noise, three interpretations with very different economic content.

The causal-inference critique generalizes the point: predictive algorithms devoid of causal models can only fit a function to data, interpreting it through a model-blind lens (Gill, 2020, pp. 767–768). This thesis accordingly treats the alignment of attributions with economically interpretable signals as a separate validation dimension rather than as a by-product of predictive accuracy.

The literature also demonstrates what such economic-plausibility validation can look like, albeit imperfectly. Gu et al. (2020, p. 2223) show that diverse ML methods agree on the same set of dominant predictive signals, a set that includes variations on momentum, liquidity, and volatility — establishing these signal families as an economically grounded reference against which attributions can be checked. Fischer and Krauss (2018, p. 654) attempt a direct post-hoc rationalization, distilling their LSTM's behavior into a rules-based short-term reversal strategy; yet the distilled rules recover only about half of the network's returns, indicating that the rationalization is incomplete and that a substantial share of the model's logic resists economic redescription.

Green (2026, pp. 539–540) reports the related finding that local explanations of a deep calibration model do not align well with financial intuition, and von Zahn et al. (2025) show experimentally that explanations change users' confidence and delegation behavior specifically when they reveal divergence between the model's logic and the user's own — a reminder that the practical value of an explanation lies precisely in exposing economically unexpected model behavior rather than in confirming prior beliefs. No reviewed study, however, systematically tests whether post-hoc attributions of stock-forecasting models capture these established economic drivers or instead track spurious, regime-specific correlations; this is the third gap addressed by RQ3.

## 3.5 Synthesis and Research Gaps

Read together, the reviewed literatures exhibit three points of broad agreement and three unresolved tensions. There is agreement, first, that flexible ML/DL models can outperform linear and random-walk benchmarks in equity prediction under favorable conditions (Gu et al., 2020; Fischer & Krauss, 2018), while the conditions themselves — horizon, market, sample size, evaluation design — moderate or even reverse the ranking (Tsay, 2005, pp. 180–183; Mao et al., 2026; Molnar, 2022, pp. 57–58). Second, there is agreement that reported gains require multiplicity-aware, leakage-free evaluation to be credible (López de Prado, 2020, pp. 105–124; Medeiros, 2022; Dixon et al., 2020, pp. 213–214). Third, there is agreement that interpretability is institutionally necessary in finance and achievable through intrinsically interpretable models or post-hoc explanation methods (Biecek & Burzykowski, 2021, pp. 6–7; Arsenault et al., 2025; Hoang et al., 2026).

Against this consensus stand three tensions: whether the accuracy–interpretability trade-off is steep or negligible in practice (Section 3.2.1); whether attention and other model-internal signals constitute reliable explanations (Section 3.4.2); and whether predictive importance can ever be read as economic mechanism (Section 3.4.3). Each tension is asserted far more often than it is measured, and each is intensified — not resolved — by the temporal structure of financial data.

From these tensions, three research gaps can be stated precisely; Table 3.1 summarizes them together with their evidentiary basis and their mapping to the research questions of this thesis.

Table 3.1 Research gaps established by the literature review and their mapping to research questions

| Gap | Statement | Principal evidence | Research question |
|---|---|---|---|
| Gap 1: Integrated evaluation and domain-specific metrics | No framework jointly evaluates predictive accuracy and explanation quality for financial time series forecasting; benchmarks standardize accuracy only, and quantitative interpretability metrics are absent | Arsenault et al. (2025, p. 265:30); Theissler et al. (2022); Hu et al. (2026); Godahewa et al. (2021) | RQ1 |
| Gap 2: Temporal instability and regime robustness | Explanation evaluation inherits i.i.d. assumptions; attribution variance across methods, retraining, and market regimes is documented conceptually but never quantified for financial forecasting | Theissler (2024); Habiba et al. (2026, Table 3.9); Molnar (2022, pp. 23–30); Dixon et al. (2020, pp. 39–40) | RQ2 |
| Gap 3: Economic reliability and faithfulness | Post-hoc explanations are rarely tested for faithfulness, and attributions are not validated against genuine economic drivers rather than spurious correlations | Arsenault et al. (2025, pp. 265:19, 265:30); Gu et al. (2020, p. 2229); Molnar (2022, pp. 28–29); López de Prado (2020, pp. 75–84) | RQ3 |

*Note.* The table consolidates the synthesis of Sections 3.1–3.4; principal evidence lists the works that document each gap rather than works that close it.

Table 3.1 makes explicit that the three gaps are complementary rather than redundant. Gap 1 concerns the measurement infrastructure: because accuracy and interpretability are evaluated with disjoint toolkits — and because, as Theissler et al. (2022) observe, accuracy can not assess the methods' interpretability — the accuracy–interpretability trade-off posited in RQ1 cannot even be quantified without a joint evaluation design. The domain-agnostic complexity measures proposed by Molnar (2022, pp. 9–10) and the fidelity/stability criteria cataloged by Arsenault et al. (2025, p. 265:24) exist as concepts but have not been instantiated as a finance-specific, benchmark-integrated protocol. Gap 2 concerns the temporal dimension that standard protocols omit: the variance problem identified by Theissler (2024), the over-time instability attributed to SHAP and LIME by Habiba et al. (2026, Table 3.9), and the dependence- and regime-driven fragility analyzed by Molnar (2022) and Dixon et al. (2020) jointly imply that an attribution measured once, on one sample, says little about its robustness to the non-stationarity, structural breaks, and regime shifts that financial series exhibit (Tsay, 2005, pp. 98–99; Kirchgässner et al., 2013, pp. 178–179) — the subject of RQ2.

Gap 3 concerns validity in the strong sense: faithfulness testing is rare (Arsenault et al., 2025, p. 265:30), explanations describe models rather than the economy (Molnar, 2022, pp. 28–29), and predictive importance does not identify economic mechanisms (Gu et al., 2020, p. 2229) — the subject of RQ3.

The thesis is positioned against the five closest works as follows. Arsenault et al. (2025) survey XAI for financial time series forecasting and diagnose the fidelity and stability deficits, but as a survey they evaluate nothing experimentally; this thesis provides a bounded experimental response to that agenda while retaining important implementation limitations. Theissler et al. (2022) provide the faithfulness-evaluation protocols, but for time series classification; this thesis transfers and adapts them to financial forecasting as a regression task with regime structure. Hu et al. (2026) deliver the financial forecasting benchmark with leakage control and regime-aware evaluation, but no explainability track; this thesis evaluates explanation diagnostics alongside forecast metrics and uses one executed Monash-related Sunspot series as a limited cross-domain robustness block (Godahewa et al., 2021).

López de Prado (2020) establishes the rigorous feature-importance program and the backtest-overfitting critique, but treats Shapley-based and surrogate explanations only in passing (p. 77) and does not examine temporal attribution stability; this thesis extends his program to the post-hoc toolbox and to the time dimension. Molnar (2022) provides the deepest methodological analysis of post-hoc interpretation — its axioms, pitfalls, and limits under dependent features — but is domain-agnostic; this thesis instantiates and stress-tests that analysis where its assumptions are most strained: on serially dependent, low-signal financial time series.

In sum, the literature demonstrates that accurate stock prediction with ML/DL is possible but fragile, that explanation methods exist in abundance, and that neither their stability, their faithfulness, nor their economic content has been established for financial time series forecasting. The three gaps are therefore not marginal omissions but direct consequences of how the constituent fields have developed: financial ML optimized accuracy, XAI optimized plausibility, and time series evaluation optimized predictive rigor, with no subfield responsible for their intersection. The methodology of this thesis, presented in Chapter 4, is designed around precisely this intersection.

# 4. Methodology

This chapter describes the study as it was actually executed. It therefore differs from the broader design in the initial proposal. The final methodology covers four financial indices and one Sunspot series, two forecast horizons, four lagged predictors, seven forecasting models, and a defined set of explanation diagnostics. No models were retrained and no experimental settings were changed during thesis finalization. Where the implementation restricts the meaning of a result, that restriction is stated at the point at which the procedure is defined.

## 4.1 Research Design

The study uses a comparative, retrospective experimental design. Model family, dataset, forecast horizon, and walk-forward scheme define the principal evaluation dimensions. The response variables are forecast error, directional agreement, R², attribution stability, and explanation-faithfulness diagnostics. The design connects directly to the research questions: predictive and explanation outcomes inform RQ1, consecutive-fold and regime-stratified attributions inform RQ2, and additivity, intervention, and cross-channel diagnostics inform RQ3.

The design is comparative but not a fully matched factorial experiment. Linear Regression, ARIMA, and the tree models have six folds per scheme and 252 test observations per fold. LSTM and TFT have four folds per scheme and 231 evaluated test observations after sequence construction. Their rolling origins also differ. ARIMA’s raw-series indices are misaligned with the lag-trimmed frame used to construct its folds. These properties prevent a single matched seven-way performance ranking. The analysis consequently treats model–series–horizon summaries as descriptive aggregates and does not estimate factorial main effects or interaction effects.

Replication takes three forms. First, rolling and expanding windows provide repeated temporal evaluations. Second, Random Forest, XGBoost, and LightGBM average predictions from two seeds, 20261017 and 20261018. Third, the same model configuration is applied to four financial indices and the normalized Sunspot input. Linear Regression, ARIMA, LSTM, and TFT use one seed. Only Random Forest has a complete exported two-seed RMSE sensitivity table, so stochastic variability cannot be compared comprehensively across model families.

The unit of aggregation must also be distinguished from the unit of observation. Forecast metrics are calculated within each completed fold and then summarized without weighting across folds and schemes. Financial pooled displays take an unweighted mean of the four series-level means. Explanation-stability counts are correlation rows, not independent forecast observations. Faithfulness success fractions are explained-fold outcomes. These levels are not pooled into a single significance test.

Figure 4.1 summarizes the executed pipeline and its audit layer. The pipeline begins with frozen data inputs, constructs the lagged frames and temporal folds, trains the seven models, and exports forecast and explanation tables. The final audit checks artifact integrity and implementation validity before interpreting the results.

![Figure 4.1. Executed forecasting, explanation, and evidence-audit pipeline.](figures/Figure_4_1_Executed_Pipeline.png)

*Note.* Own illustration from the executed notebook code. Static test blocks contain 252 observations; deep test blocks contain 231 after sequence construction. The operational split subtracts purge and embargo before each test block.

## 4.2 Data

### 4.2.1 Data Sources and Sample Coverage

The primary data are daily adjusted-price observations for the S&P 500, DAX, NASDAQ Composite, and Dow Jones Industrial Average. The supplied snapshot records Yahoo Finance as the source for the indices. The non-financial block is the normalized Sunspot file used by all seven notebooks and associated in the supplied materials with the Monash Time Series Forecasting Archive (Godahewa et al., 2021). The executed study does not include the broader FinTSB collection described in the proposal, although FinTSB remains relevant to the literature on leakage-aware financial evaluation (Hu et al., 2026).

Table 4.1 reports the verified dimensions and dates. Raw rows differ from execution rows for the financial series because the first return observation is undefined. The modeling frame then loses 252 observations at the start through lag construction and five at the end through target construction. No duplicate dates, missing adjusted closes, nonpositive index prices, return missing values after transformation, or modeling-frame missing values were found.

**Table 4.1**

*Verified dataset structure and temporal coverage.*

| Series | Raw rows | Execution rows | Frame rows | Raw period | Frame period |
|---|---:|---:|---:|---:|---:|
| S&P 500 | 9,067 | 9,066 | 8,809 | 1990-01-02–2025-12-31 | 1991-01-02–2025-12-23 |
| DAX | 9,107 | 9,106 | 8,849 | 1990-01-02–2025-12-30 | 1991-01-09–2025-12-18 |
| NASDAQ Composite | 9,067 | 9,066 | 8,809 | 1990-01-02–2025-12-31 | 1991-01-02–2025-12-23 |
| DJIA | 8,561 | 8,560 | 8,303 | 1992-01-02–2025-12-31 | 1992-12-31–2025-12-23 |
| Sunspot | 11,109 | 11,109 | 10,852 | 1990-01-01–2020-05-31 | 1990-09-10–2020-05-26 |

*Note.* Execution rows are return rows for financial indices and native-value rows for Sunspot. Frame rows remain after lag and target construction. All listed duplicate-date, missing-value, and nonpositive-price checks were zero where applicable. Source: `dataset_structure_audit.csv`, reconciled against the supplied raw and execution-input files.

The two index-data representations were reconciled field by field. Dates and row counts agree, and shared Open, High, Low, Close, Adjusted Close, and Volume fields match exactly or to floating-point precision. Their byte hashes differ because the snapshot adds adjusted-close and adjusted-open columns and uses a lowercase date field. This is a schema and serialization difference, not a substantive numerical conflict.

The Sunspot provenance is materially different. The snapshot CSV reproduces the supplied raw TSF values, but the normalized notebook input has different values on 10,957 of its 11,109 rows. The maximum absolute difference is 274, and the correlation between the normalized input and aligned raw series is approximately 0.406. No supplied artifact documents the transformation. All seven notebooks nevertheless use the same normalized file, confirmed by its provenance hash. The executed Sunspot results are therefore internally consistent with the notebook input but cannot be described as numerically traceable to the supplied raw archive values.

### 4.2.2 Variables and Forecast Horizons

For a financial index, let $P_t$ be the adjusted closing price at time $t$. The one-session log return is

$$
r_t = \ln P_t - \ln P_{t-1}. \tag{4.1}

$$

For horizons $h \in \{1,5\}$, the supervised target is the cumulative future log return,

$$
y_{t,h}=\sum_{i=1}^{h}r_{t+i}=\ln P_{t+h}-\ln P_t. \tag{4.2}

$$

This is a regression task. Directional accuracy is calculated from the sign of the predicted and realized cumulative return, but no separate classification model is estimated. At $h=5$, adjacent targets overlap, which affects the interpretation of directional agreement and any prospective error-independence assumption.

For Sunspot, the input column is the native value $v_t$, and the target is $y_{t,h}=v_{t+h}$. Financial and Sunspot errors consequently have different units and meanings. They are displayed separately and are never averaged into a common error statistic.

## 4.3 Temporal Preprocessing and Feature Engineering

### 4.3.1 Lag Construction and Modeling Frames

Every model receives four predictors: `lag_1`, `lag_5`, `lag_21`, and `lag_252`. For financial series, each predictor is a lagged log return, $x_{t,\ell}=r_{t-\ell}$. For Sunspot, each is a lagged native value, $x_{t,\ell}=v_{t-\ell}$. These lags approximate previous-session, one-week, one-month, and one-year information in the financial setting, but the thesis does not interpret them as independent economic mechanisms.

The executed frame function constructs all lags and both targets before trimming the largest lag and target horizon and dropping remaining missing values. It does not add rolling volatility, momentum, drawdown, macroeconomic, liquidity, or other exogenous variables. It also does not execute the forward imputation described in the initial proposal. The verified modeling frames instead rely on complete rows after deterministic lag and target construction.

Linear, tree, and deep tabular features are standardized using statistics fitted on the training portion of each fold. The same fitted scaler is applied to test and explanation inputs. ARIMA is univariate and does not use the four-lag predictor vector; its implementation issue is described separately in Section 4.4.1.

### 4.3.2 Walk-Forward Splits and Validation Tails

The common nominal settings are a minimum training window of 500 observations, a test span of 252, purge 5, embargo 5, and rolling and expanding schemes. Both schemes are established approaches to temporal forecast evaluation (Ballarin, 2024; Medeiros, 2022, pp. 111–120). For a frame of length $n$, rolling fold $k$ starts its test block at $n-K\times252+k\times252$. Expanding fold $k$ starts at $500+5+5+k\times252$. Training ends ten observations before the test start. Rolling training retains at most the most recent 500 observations before that gap; expanding training starts at the beginning of the frame.

The code subtracts purge plus embargo before the test block. The operational design therefore creates a ten-observation pre-test gap; it does not implement a separate post-test embargo. Describing the configuration simply as purge 5 and embargo 5 without this qualification would misrepresent the executed split.

The static models use $K=6$, and the deep models use $K=4$. Deep inputs are sequences of 21 consecutive rows. Sequence construction removes the first 21 targets from each training and test slice, leaving 231 evaluated test targets in a full deep test block. This difference changes both the number and position of evaluated observations. Deep/static RMSE differences are therefore reported as exported descriptive results, not matched performance premiums.

Boosted trees and neural networks use the final 10% of training rows or sequences for early stopping, with a minimum of 50. There is no additional purge between the fitting and early-stopping partitions. Models are not refit on the validation tail after the stopping point is selected. The neural networks restore the best validation state; the boosted trees retain their early-stopped fit. This internal validation design is less strict than the outer pre-test gap and is included in the validity assessment.

Table 4.2 summarizes the resulting coverage and explains why the forecast and explanation row counts differ across model groups.

**Table 4.2**

*Executed fold design, forecast coverage, and explanation coverage.*

| Model group | Schemes | Folds per scheme | Training window | Test targets per full fold | Forecast rows per model | Explained units |
|---|---|---:|---|---:|---:|---:|
| Linear Regression and tree models | Rolling and expanding | 6 | 500 rolling maximum; expanding from frame start | 252 | 120 | 80 |
| ARIMA | Rolling and expanding | 6 | Same nominal fold design; raw-array alignment defective | 252 | 120 | 20 native units |
| LSTM and simplified TFT | Rolling and expanding | 4 | 500 rolling maximum; expanding from frame start | 231 after lookback 21 | 80 per model | 60 per model |

*Note.* Each fold has a nominal 252-observation test span and a ten-observation pre-test gap formed by purge 5 plus embargo 5. Forecast rows combine five series, two horizons, two schemes, and the stated folds. Deep test targets differ in number and position; ARIMA timestamps are misaligned. Source: executed notebook code and consolidated `forecast_results.csv` files.

## 4.4 Models

Table 4.3 lists the executed model settings. The configurations are fixed by the saved notebook code and are not reconstructed from the earlier methodology narrative.

**Table 4.3**

*Executed forecasting and explanation configurations.*

| Model | Executed forecast configuration | Replication and folds | Explanation channel and coverage qualification |
|---|---|---|---|
| Linear Regression | OLS on four standardized lags; no penalty or tuning search | One seed; six folds | Exact coefficients/SHAP; PFI; ALE; weighted local ridge surrogate; interventions |
| ARIMA | AIC selection from (1,0,0), (2,0,0), (1,0,1), (2,0,1), (3,0,0), (1,1,0), (2,1,0); sequential updates without refitting each origin | One seed; six folds | Native orders and parameter vectors; raw/frame alignment defective |
| Random Forest | 100 trees; maximum depth 10; minimum leaf 20 | Two-seed forecast average; six folds | TreeSHAP/PFI/local surrogate/interventions on first-seed constituent |
| XGBoost | At most 200 rounds; depth 6; learning rate 0.05; row/column sample 0.8; L1 0.1; L2 1.0; 25-round early stopping | Two-seed forecast average; six folds | TreeSHAP/PFI/local surrogate/interventions on last-seed constituent |
| LightGBM | Matching fixed boosted-tree settings; callback-based 25-round early stopping | Two-seed forecast average; six folds | TreeSHAP/PFI/local surrogate/interventions on last-seed constituent |
| LSTM | Lookback 21; hidden size 64; two layers; dropout 0.1; batch 64; Adam 0.001; at most 25 epochs; patience 8; best state restored | One seed; four folds | KernelSHAP over 84 flattened inputs, aggregated to four lags; companion diagnostics; probe defects |
| Simplified TFT | Variable-selection softmax, LSTM encoder, four-head attention, linear output; same core optimization settings as LSTM | One seed; four folds | KernelSHAP plus native variable-selection/attention signals; probe defects; not full PyTorch Forecasting TFT |

*Note.* Base seed 20261017; tree second seed 20261018. Boosted-tree and neural early stopping use the final 10% of training rows/sequences (minimum 50), without an additional inner purge or refit on the validation tail. Source: the seven preserved notebooks; no setting was changed during manuscript finalization.

### 4.4.1 Linear Regression and ARIMA

Linear Regression is estimated by ordinary least squares on the standardized lag features. Its fitted coefficients provide a direct description of the linear prediction rule, making it the intrinsically interpretable reference. No ridge penalty or hyperparameter search is executed.

ARIMA selects one order per fold by AIC from seven candidates: (1,0,0), (2,0,0), (1,0,1), (2,0,1), (3,0,0), (1,1,0), and (2,1,0). It then produces multi-origin forecasts through sequential state updates without refitting at every test origin. The candidate distribution is reported as a native-model result in Chapter 5.

A critical alignment defect affects ARIMA. Fold indices are created from the lag-trimmed frame, but the ARIMA training and update arrays are sliced from the untrimmed return or value series. The same integer positions therefore do not identify the same timestamps used by the other models and targets. ARIMA’s exported numbers remain documented observations, but they are excluded from matched model-comparison conclusions.

### 4.4.2 Random Forest, XGBoost, and LightGBM

Random Forest uses 100 trees, maximum depth 10, and minimum leaf size 20. XGBoost and LightGBM use up to 200 boosting rounds, depth 6, learning rate 0.05, row and column subsampling of 0.8, L1 regularization 0.1, L2 regularization 1.0, and 25-round early stopping. XGBoost passes early stopping through the estimator; LightGBM uses callbacks. These settings describe the executed models rather than a tuned search grid.

For each of these models, the forecast is the average of predictions from two seeds. Explanations do not cover the averaged predictor in the same way. Random Forest explains the first-seed model, whereas XGBoost and LightGBM explain the model from the last seed. Explanation results therefore characterize one constituent rather than the complete two-seed forecasting ensemble.

The exported completion audits list 240 seed-level planned units for each tree model and 120 completed ensemble forecast rows. This is a denominator distinction, not 120 missing forecast experiments. The completed forecast grain is the two-seed ensemble row.

### 4.4.3 LSTM and Simplified TFT

The LSTM uses lookback 21, hidden size 64, two recurrent layers, dropout 0.1, batch size 64, Adam with learning rate 0.001, at most 25 epochs, and validation patience 8. The best validation state is restored before forecasting.

The executed TFT is a simplified adapter rather than the full PyTorch Forecasting Temporal Fusion Transformer. It combines a variable-selection softmax over the four inputs, an LSTM encoder, four-head multi-head attention, and a linear output. It uses the same lookback, hidden size, batch size, optimizer, learning rate, epoch cap, and patience as the LSTM. Native variable-selection and attention signals are exported, but these signals are evaluated rather than assumed to be faithful explanations.

## 4.5 Explainability Diagnostics

### 4.5.1 Post-hoc and Model-Specific Attribution Channels

Linear Regression uses exact coefficient-based attributions and linear SHAP consistency checks. Tree models use TreeSHAP on the explained constituent. Deep models use KernelSHAP on flattened sequence inputs and then aggregate the 21-by-4 attributions into four lag-level values. The local surrogate is a weighted ridge model fitted on synthetic normal perturbations in the style of LIME; it is not a claim that a particular external LIME package was executed. Permutation feature importance supplies an alternative global channel, and the Linear archive includes an accumulated-local-effects figure. ARIMA exports native orders and parameter vectors, while TFT exports variable-selection and attention-related signals.

The explanation budget covers four folds per scheme for static models and three for deep models. Static explanations use 40 probe positions and deep explanations use 30. The resulting explained-unit totals are 80 for Linear and each tree model, 60 for LSTM and TFT, and 20 reported native ARIMA units. Forecast coverage and explanation coverage are therefore not identical.

Deep explanation probes have a critical construction defect. Probe rows are spaced across the full frame, filtered to positions at least 21, and then passed to the sequence constructor as if they were contiguous histories. The sequences formed from these rows do not represent 21 consecutive sessions. In addition, assigning `pr[:len(phi)]` does not correctly identify the sequence outputs after the lookback reduction. Deep stability, attribution, and surrogate diagnostics are consequently exploratory.

### 4.5.2 Explanation-Model Coverage and Economic Scope

The distinction between the forecast model and the explained model is essential. For the tree ensembles, SHAP values describe one seed’s predictor while the forecast table describes a two-seed average. Additivity for a constituent does not automatically establish additivity for the averaged predictor. For TFT, native weights and KernelSHAP are different channels and need not agree.

All attribution channels operate on lagged prices or values. No macroeconomic, liquidity, volatility, accounting, or causal variables are supplied to the models. The explanation methods can therefore describe statistical reliance on past observations. They cannot identify genuine macroeconomic drivers, distinguish risk compensation from mispricing, or exclude spurious correlations. Economic plausibility is assessed as a limitation of identification, not inferred from feature rankings.

## 4.6 Evaluation Framework

### 4.6.1 Predictive Metrics

For $N$ evaluated observations, forecast errors are summarized by

$$
\operatorname{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2}, \qquad \operatorname{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat y_i|. \tag{4.3}

$$

Directional accuracy is the proportion of observations for which predicted and realized targets have the same sign. It is exported for financial series and is missing by design for Sunspot. R² is the fold-level scikit-learn value,

$$
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y_{\mathrm{test}})^2}, \tag{4.4}

$$

where $\bar y_{\mathrm{test}}$ is the realized mean within the test block. This reference uses test-block information and is therefore an oracle descriptive comparator, not an implementable ex ante forecast. Negative R² indicates that the forecasts perform worse than that particular reference in squared error; it does not replace a valid benchmark test.

The primary model–series–horizon table reports unweighted means across completed fold rows. Variability is retained through standard deviations, medians, quartiles, minima, maxima, and fold-level negative-R² counts in the machine-readable evidence table. Financial pooled means are second-stage unweighted means of the four series means. No unit-bearing RMSE or MAE value is pooled across financial and Sunspot data.

### 4.6.2 Stability and Faithfulness Metrics

For each probe, stability is the Spearman correlation between its four attribution values in consecutive folds. The calculation is repeated across fold pairs, schemes, horizons, series, and regime groups, then summarized by median and count. It is not a correlation of two global four-feature importance vectors. With only four coordinates and possible ties, the statistic is coarse and discrete. It measures local rank consistency, not attribution-magnitude stability or the stability of a global ranking.

SHAP additivity is checked through the maximum absolute residual between the model output and the SHAP base value plus summed attributions. Near-zero residuals support numerical local-additivity consistency for the explained model. They do not by themselves prove intervention faithfulness or economic validity.

Deletion and insertion diagnostics compare perturbation-curve areas under model-attribution orderings with the mean of 32 random orderings. A fold-level success indicator records whether the model ordering performs better than that random-control mean. For deep models, the model ordering has four aggregated lags while the random controls permute all 84 flattened sequence inputs. This mismatch makes the deep success fractions unsuitable as like-for-like faithfulness estimates. Zero-baseline perturbations can also create inputs unlike naturally observed financial histories.

### 4.6.3 Regime-Shift Analysis

Financial regime labels use 63-session rolling statistics of log returns. Bull observations require a rolling mean above its full-series 60th percentile and rolling standard deviation below its 80th percentile. Bear observations have rolling mean below the 40th percentile. Crisis observations have rolling standard deviation above the 95th percentile and override the preceding labels. Remaining observations are labeled boundary. Sunspot is labeled NA.

These are data-driven strata, not independently dated historical bull, bear, or crisis periods. Because thresholds use the full series, they incorporate future distributional information and do not support a strictly prospective regime-classification interpretation. Small groups, especially crisis, further limit generalization. The analysis addresses stability across these executed strata, not robustness to every structural break or historical crisis.

### 4.6.4 Statistical Comparison and Unavailable Procedures

Every notebook exports Diebold–Mariano statistics intended to compare model loss with zero-return or persistence and training-mean references. The implementation sets the model loss within each fold to a constant equal to the fold RMSE squared. It does not use the observation-level squared errors $(y_i-\hat y_i)^2$. The resulting test input is invalid for the intended equal-predictive-loss comparison. All exported Diebold–Mariano statistics and p-values are therefore excluded from inferential conclusions.

Observation-level prediction files were not supplied, so valid tests cannot be reconstructed from the exports alone. Recomputing them would require re-execution, which was not authorized. The same restriction applies to a seven-way Model Confidence Set, corrected multiple-comparison procedures, combinatorial purged cross-validation, Sharpe-ratio and deflated-Sharpe analysis, and comprehensive explanation-overhead comparisons. These procedures were planned or narrated but are not supported by complete executed artifacts. Chapter 5 reports their status rather than presenting placeholders as results.

## 4.7 Reproducibility and Implementation Provenance

The notebooks record Python 3.13.5 on macOS ARM and use 20261017 as the base seed. All seven original notebooks were parsed without execution or modification. Their model archives passed CRC checks and safe-path inspection. Every internal model-archive checksum matched, as did all 18 hashes in the raw-data input manifest. The final package preserves the original `.ipynb` files byte for byte.

These checks establish artifact integrity and provenance consistency. They do not establish independent computational reproducibility. The supplied materials do not include a complete dependency lock, a detailed hardware specification, observation-level predictions for all models, or an independent rerun. Notebook download code and local execution paths may also depend on external data availability and software versions. The README therefore separates execution instructions from verified reproduction claims.

The evidence hierarchy used throughout the thesis is as follows: executed notebook code establishes configurations; exported result tables supply numerical observations; transparent aggregations derived from those tables are labeled as descriptive; model evidence statements act as locators rather than independent proof; and implementation validity is assessed before any numerical result is interpreted. This hierarchy allows negative findings and defects to remain visible instead of being replaced by a more favorable narrative. Appendix C documents the seven notebooks and their configurations, and Appendix D records provenance and reproducibility checks.

# 5. Results

This chapter reports the verified outputs of the seven executed studies. The results are descriptive unless an executed procedure explicitly supports a stronger statement. Financial pooled values are unweighted means of the four index-level means. Static model cells contain 12 fold–scheme rows and deep model cells contain eight. Tree forecast rows already represent two-seed prediction averages. Financial and Sunspot errors are reported in their own units and are never pooled.

All 760 exported forecast rows have complete status. Mean R² is negative in every model–series–horizon cell, and negative findings are retained throughout. ARIMA’s alignment defect, the deep/static evaluation mismatch, deep explanation-probe defects, and the invalid Diebold–Mariano implementation restrict interpretation. Observation-level predictions were not supplied, so the chapter does not introduce reconstructed forecast plots or alternative statistical tests.

## 5.1 Predictive Performance (RQ1)

### 5.1.1 Financial Forecast Errors

Table 5.1 and Figure 5.1 summarize financial predictive performance. At h1, TFT has the lowest pooled RMSE (0.009804), followed by LSTM (0.009864). XGBoost and LightGBM follow at 0.011392 and 0.011398, with Linear Regression at 0.011476 and Random Forest at 0.011529. At h5, TFT again has the lowest pooled RMSE (0.021999), followed by LSTM (0.022124). LightGBM, ARIMA, XGBoost, Linear Regression, and Random Forest lie between 0.024738 and 0.025023.

The deep models also have the lowest pooled MAE: 0.007288 for TFT and 0.007335 for LSTM at h1, and 0.016917 and 0.017063 at h5. These are consistent descriptive error patterns across the four indices. They are not a matched estimate of a deep-learning premium because LSTM and TFT use fewer folds, different rolling origins, and 231-observation test windows.

**Table 5.1**

*Descriptive predictive summary across four financial indices.*

| Model | h1 RMSE | h1 MAE | h1 DA | h1 R² | h5 RMSE | h5 MAE | h5 DA | h5 R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear | 0.011476 | 0.008366 | 0.518188 | -0.025129 | 0.024787 | 0.018750 | 0.566799 | -0.039142 |
| ARIMA* | 0.011574 | 0.008446 | 0.511161 | -0.054154 | 0.024738 | 0.018809 | 0.553985 | -0.034170 |
| Random Forest | 0.011529 | 0.008426 | 0.505704 | -0.042062 | 0.025023 | 0.018950 | 0.551587 | -0.059547 |
| XGBoost | 0.011392 | 0.008287 | 0.525050 | -0.012616 | 0.024764 | 0.018758 | 0.558614 | -0.034385 |
| LightGBM | 0.011398 | 0.008296 | 0.524140 | -0.013765 | 0.024744 | 0.018762 | 0.553158 | -0.033900 |
| LSTM** | 0.009864 | 0.007335 | 0.529491 | -0.031577 | 0.022124 | 0.017063 | 0.554113 | -0.103841 |
| TFT** | 0.009804 | 0.007288 | 0.526650 | -0.021248 | 0.021999 | 0.016917 | 0.557630 | -0.084182 |

*Note.* Values are unweighted means of four series-level means; RMSE and MAE use cumulative log-return units. *ARIMA has an index-alignment defect. **Deep models have four folds and 231 test observations per fold rather than six folds and 252 observations. No column implies statistical superiority. Source: `financial_model_horizon_summary.csv`, derived from all seven `forecast_results.csv` files.

![Figure 5.1. Descriptive financial RMSE by model and horizon.](figures/Figure_5_1_Financial_RMSE.png)

*Note.* Unweighted means of four series-level means. ARIMA is aligned defectively; deep models have non-matched origins and test lengths. Source: aggregations of all seven `forecast_results.csv` files.

Among the static models with valid frame alignment, XGBoost has the lowest pooled h1 RMSE and LightGBM the lowest pooled h5 RMSE by very small margins. The h1 difference between XGBoost and Linear Regression is 0.000084 in pooled RMSE; at h5, LightGBM is 0.000043 below Linear Regression. These differences are descriptive and should not be interpreted as economically or statistically material without valid matched tests. Random Forest has the highest pooled RMSE among the non-ARIMA static models at both horizons.

Figure 5.2 shows that the broad pattern occurs in each financial series rather than being driven by one pooled index. LSTM and TFT occupy the lower exported RMSE region for DAX, DJIA, NASDAQ, and S&P 500 at both horizons. The tree and linear models remain close to one another. ARIMA is shown for completeness but is not used for matched comparisons.

![Figure 5.2. Financial RMSE profiles by executed series, model, and horizon.](figures/Figure_5_2_Series_RMSE.png)

*Note.* Each point is a mean across executed folds and schemes within a series. Financial cumulative log-return units only. Source: `empirical_evidence_master_table.csv`.

**Table 5.2**

*Series-level financial RMSE means by model and horizon.*

| Model | Series | h1 RMSE | h5 RMSE |
|---|---|---:|---:|
| Linear | S&P 500 | 0.010277 | 0.021537 |
| Linear | DAX | 0.011788 | 0.026140 |
| Linear | NASDAQ Composite | 0.012917 | 0.028363 |
| Linear | DJIA | 0.010921 | 0.023106 |
| ARIMA | S&P 500 | 0.010350 | 0.021417 |
| ARIMA | DAX | 0.011821 | 0.026246 |
| ARIMA | NASDAQ Composite | 0.013162 | 0.028255 |
| ARIMA | DJIA | 0.010964 | 0.023033 |
| Random Forest | S&P 500 | 0.010292 | 0.021790 |
| Random Forest | DAX | 0.011904 | 0.026326 |
| Random Forest | NASDAQ Composite | 0.012990 | 0.028766 |
| Random Forest | DJIA | 0.010929 | 0.023212 |
| XGBoost | S&P 500 | 0.010155 | 0.021494 |
| XGBoost | DAX | 0.011810 | 0.026184 |
| XGBoost | NASDAQ Composite | 0.012808 | 0.028337 |
| XGBoost | DJIA | 0.010796 | 0.023040 |
| LightGBM | S&P 500 | 0.010160 | 0.021514 |
| LightGBM | DAX | 0.011815 | 0.026111 |
| LightGBM | NASDAQ Composite | 0.012823 | 0.028291 |
| LightGBM | DJIA | 0.010793 | 0.023061 |
| LSTM | S&P 500 | 0.008648 | 0.019020 |
| LSTM | DAX | 0.009716 | 0.021774 |
| LSTM | NASDAQ Composite | 0.011520 | 0.025935 |
| LSTM | DJIA | 0.009572 | 0.021769 |
| TFT | S&P 500 | 0.008580 | 0.018911 |
| TFT | DAX | 0.009705 | 0.021922 |
| TFT | NASDAQ Composite | 0.011457 | 0.025835 |
| TFT | DJIA | 0.009474 | 0.021327 |

*Note.* Each value is an unweighted mean across completed rolling and expanding fold rows. ARIMA is defective for matched comparison; deep models use non-matched origins and test lengths. Source: `empirical_evidence_master_table.csv`.

Table 5.2 gives the underlying series-level RMSE means. For example, TFT’s h1 means range from 0.008580 on the S&P 500 to 0.011457 on NASDAQ, while its h5 means range from 0.018911 to 0.025835. Linear Regression’s corresponding ranges are 0.010277–0.012917 at h1 and 0.021537–0.028363 at h5. The similar ordering across indices strengthens the descriptive robustness of the financial pattern, but it does not remove the evaluation-design mismatch.

### 5.1.2 Directional Accuracy and R²

Directional accuracy does not reproduce the RMSE ordering. LSTM has the highest pooled h1 DA at 0.529491, followed by TFT at 0.526650 and XGBoost at 0.525050; XGBoost is highest among the static models. Linear Regression records 0.518188 at h1. At h5, Linear Regression’s 0.566799 exceeds all other pooled values. XGBoost and TFT record 0.558614 and 0.557630, respectively, and LSTM records 0.554113.

Across all financial model–series–horizon cells, mean DA ranges from 0.491402 to 0.584987. Linear Regression’s overall financial mean across both horizons is 0.542493386243. Some cells therefore exceed 0.50, but 0.50 is not automatically the correct benchmark when target signs are imbalanced. Five-session targets overlap, and the available artifacts contain no valid directional-skill test. DA cannot establish trading profitability, market inefficiency, or forecast superiority.

The R² results are uniformly weak. Mean R² is negative in all 70 model–series–horizon cells, and 671 of 760 exported fold rows are negative. In the financial pooled table, h1 means range from −0.012616 for XGBoost to −0.054154 for ARIMA; h5 means range from −0.033900 for LightGBM to −0.103841 for LSTM. Thus, lower RMSE for deep models does not imply a better R² ranking: their predictions still compare unfavorably with the realized test-block mean used by the fold-level calculation.

Figure 5.3 displays the financial fold-level distributions. The negative median tendency persists across model families and horizons, with visible variability between folds. The test-block mean is an oracle descriptive reference, so these results should be read as frequent failure to outperform that reference rather than as a complete ex ante benchmark evaluation.

![Figure 5.3. Financial fold-level R² relative to the realized test-block mean.](figures/Figure_5_7_Financial_R2.png)

*Note.* Boxes combine rolling and expanding folds and hide outliers for readability. The reference uses test-block information and is not an implementable forecast. Source: consolidated `forecast_results.csv` rows.

### 5.1.3 Sunspot Robustness Block

The financial RMSE ordering reverses on the normalized Sunspot input. Linear Regression has the lowest mean RMSE by a substantial descriptive margin: 13.139662 at h1 and 13.108861 at h5. TFT is next at 20.605463 and 20.355241. Random Forest records 23.210541 and 23.106027, LightGBM 27.486915 and 27.401196, XGBoost 30.331053 and 29.110530, and ARIMA 31.655317 and 32.105515. LSTM is descriptively worst at 90.360062 and 90.260971, with MAE values of 89.412689 and 89.314367.

Figure 5.4 keeps these values in native Sunspot-number units. It demonstrates that no model is universally superior across the executed datasets. In particular, the model with the poorest financial error profile is not the poorest on Sunspot, and the deep architecture with the strongest financial RMSE is not the strongest on the robustness block.

![Figure 5.4. Sunspot RMSE on the normalized notebook input.](figures/Figure_5_3_Sunspot_RMSE.png)

*Note.* Native Sunspot-number units. The normalized input differs from the supplied raw archive on 10,957 of 11,109 rows, so external provenance remains unresolved. Deep windows are non-matched and ARIMA is aligned defectively.

The result is valid as a description of what the notebooks executed on their shared normalized file. It is not a verified benchmark result for the raw Monash series because the transformation between the raw TSF values and the notebook input is undocumented. This distinction limits external comparison but does not justify replacing the exported values with a more favorable narrative.

### 5.1.4 Status of Statistical Comparisons

No statistically supported model ranking is reported. The exported Diebold–Mariano tables use a repeated fold-level RMSE² constant as model loss rather than observation-level squared forecast error. Their statistics and p-values are invalid for the intended predictive-superiority tests and are excluded. A Model Confidence Set, valid corrected multiple comparisons, combinatorial purged cross-validation, economic backtests, Sharpe ratios, deflated Sharpe ratios, and comprehensive computational-overhead outcomes were not executed or exported as complete artifacts. Table 5.3 records the status and interpretive consequence of each procedure.

**Table 5.3**

*Availability and permitted interpretation of statistical and extended evaluations.*

| Procedure or evidence family | Status | Reason or scope | Interpretive consequence |
|---|---|---|---|
| Diebold–Mariano statistics and p-values | Exported but invalid | Model loss is constant RMSE² within each fold rather than observation-level error | Excluded from all inferential conclusions |
| Seven-way Model Confidence Set | Not available | No complete executed artifact | No superior model set claimed |
| Corrected predictive-superiority and multiplicity analysis | Not available | Observation-level predictions absent; no valid executed correction | All model rankings remain descriptive |
| Combinatorial purged cross-validation | Not executed as a complete artifact | Only the documented rolling/expanding design is available | No CPCV result reported |
| Economic backtest and Sharpe/deflated-Sharpe analysis | Not available | No transaction-cost, portfolio, or risk-adjusted result files | No profitability or deployment claim |
| Comprehensive runtime, memory, and explanation overhead | Not available | Isolated printed runtimes do not form a complete comparison | No computational-cost ranking |
| Random Forest two-seed sensitivity | Available for RF only | 96 financial paired fold rows | RF-specific forecast sensitivity only; no other-model generalization |

*Note.* “Not available” identifies missing executed evidence, not a zero or unfavorable experimental result. Source: notebook and archive audit.

Consequently, terms such as best, lowest, or highest in this chapter refer only to the stated descriptive aggregation. They do not denote statistical membership in a superior model set. The absence of valid inference is itself a result of the audit and constrains the answers to all three research questions.

## 5.2 Post-hoc and Native Explanation Outcomes (RQ1 and RQ3)

### 5.2.1 Distribution of Top-Ranked SHAP Features

The exported attributions do not identify one universally dominant lag. Linear Regression’s most frequent top-ranked feature is `lag_1`, occurring in 33 of 80 explained fold units. Random Forest also has `lag_1` most often, in 29 of 80 units. XGBoost’s modal top feature is `lag_252` in 30 of 80 units, and LightGBM’s is `lag_21` in 29 of 80. LSTM and TFT have `lag_1` as their modal top feature in 20 of 60 and 19 of 60 units, respectively.

Figure 5.5 shows the full distribution rather than only the modal category. For example, Random Forest ranks `lag_252` first in 22 units and `lag_1` first in 29, so describing the model as uniformly driven by a single long lag would be inaccurate. LSTM’s first-place counts are divided among `lag_1` (20), `lag_21` (19), `lag_252` (15), and `lag_5` (6). These distributions show model- and fold-specific reliance rather than a stable global economic hierarchy.

![Figure 5.5. Distribution of top-ranked mean absolute SHAP features.](figures/Figure_5_6_Top_SHAP_Features.png)

*Note.* Counts cover 80 explained units for static models and 60 for deep models. Deep rankings are exploratory; tree explanations cover one constituent of each two-seed forecast ensemble. Source: `shap_importance.csv` aggregations.

Raw SHAP magnitudes are not averaged across financial and Sunspot observations because the targets and input scales differ. The rank-based display avoids that unit-mixing problem. It still does not solve the deep-probe defect or the constituent-versus-ensemble coverage limitation.

### 5.2.2 Native ARIMA and TFT Signals

Across 120 ARIMA forecast rows, the selected orders are (1,0,0) in 40 units, (3,0,0) in 36, (1,0,1) in 20, (2,0,0) in 18, and (2,0,1) in 6. No exported row selects (1,1,0) or (2,1,0). These counts document model-mechanics variation across folds. They do not rescue ARIMA forecast comparability because the alignment defect remains.

TFT’s mean native variable-selection weights are 0.233000 for `lag_1`, 0.290524 for `lag_5`, 0.229747 for `lag_21`, and 0.246729 for `lag_252`. The weights sum to one and are relatively diffuse, with `lag_5` largest on average. This differs from TFT’s modal post-hoc top feature, `lag_1`, illustrating that native and post-hoc channels need not produce the same account.

Native weights are not treated as validated explanations merely because they are produced inside the architecture. Their mean native-versus-post-hoc Spearman agreement is only 0.330 across 60 units, and the deep-probe defects affect the comparison. The appropriate conclusion is limited agreement between channels, not proof that either channel is faithful or economically meaningful.

## 5.3 Temporal Stability Across Regimes (RQ2)

### 5.3.1 Consecutive-Fold Attribution Stability

Table 5.4 and Figure 5.6 report median per-probe Spearman correlations. Linear Regression and Random Forest have median 0.8 in bear, boundary, bull, crisis, and Sunspot/NA groups. XGBoost and LightGBM have median 0.4 in all four financial groups and 0.8 on Sunspot. These results directly contradict a blanket claim that explanations universally degrade in bear or crisis regimes.

**Table 5.4**

*Median per-probe consecutive-fold attribution correlation by regime stratum.*

| Model and channel | Bear | Boundary | Bull | Crisis | Sunspot/NA |
|---|---:|---:|---:|---:|---:|
| Linear — exact coefficient/SHAP | 0.800000 | 0.800000 | 0.800000 | 0.800000 | 0.800000 |
| Random Forest — TreeSHAP | 0.800000 | 0.800000 | 0.800000 | 0.800000 | 0.800000 |
| XGBoost — TreeSHAP | 0.400000 | 0.400000 | 0.400000 | 0.400000 | 0.800000 |
| LightGBM — TreeSHAP | 0.400000 | 0.400000 | 0.400000 | 0.400000 | 0.800000 |
| LSTM — KernelSHAP* | 0.400000 | 0.400000 | 0.400000 | 0.200000 | 0.400000 |
| TFT — KernelSHAP* | 0.200000 | 0.400000 | 0.316228 | 0.300000 | 0.258114 |
| TFT — native variable selection* | 0.200000 | 0.100000 | 0.100000 | 0.400000 | 0.000000 |
| ARIMA — native vectors** | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

*Note.* Each statistic is a median of per-probe Spearman correlations over four attribution values across consecutive folds. Crisis counts are 16 for each deep channel and 72 for each static channel; these are correlation-row counts, not independent forecast samples. *Deep values are exploratory because probe construction is defective. **ARIMA values are reported for completeness but excluded from substantive interpretation because tied zeros make them structurally uninformative and its index alignment is defective. Source: `stability_summary_consolidated.csv`.

![Figure 5.6. Consecutive-fold attribution stability by data-driven regime stratum.](figures/Figure_5_4_Stability.png)

*Note.* Median per-probe correlation of four attribution values across consecutive folds. Financial labels use 63-session statistics and full-series quantiles. Deep diagnostics are exploratory; ARIMA values are structurally uninformative.

LSTM’s KernelSHAP medians are 0.4 in bear, boundary, bull, and Sunspot/NA, and 0.2 in crisis. TFT KernelSHAP ranges from 0.2 in bear to 0.4 in boundary, with 0.316228 in bull, 0.3 in crisis, and 0.258114 on Sunspot. TFT native variable-selection values are lower in boundary and bull (0.1) than in crisis (0.4), and zero on Sunspot. The native channel therefore does not show a simple crisis-only degradation pattern either.

These deep values cannot support strong robustness conclusions. The explanation sequences are constructed from noncontiguous probe rows, and output positions are assigned incorrectly after sequence construction. The numbers are reported because they were exported, but their principal value is to demonstrate the need for corrected diagnostics rather than to rank deep explanation reliability definitively.

ARIMA has median 1.0 in every exported group. This value is not evidence of exceptional robustness. The native four-lag vector effectively contains one nonzero AR entry and tied zeros, making the rank correlation structurally uninformative. ARIMA’s forecast alignment defect provides an additional reason to exclude these values from substantive stability rankings.

### 5.3.2 Interpretation of Regime Labels and Sample Counts

The stability counts are correlation rows, not independent forecast samples. The same probe can contribute to multiple fold-pair and scheme summaries, and the four financial labels derive from full-series rolling quantiles rather than independently dated historical events. Crisis has fewer rows than the other financial groups, with 16 rows for each deep channel and 72 for each static channel in the exported summary. Median differences involving crisis therefore require particular caution.

The results answer a narrow question: how consistent are four local attribution coordinates across consecutive fitted folds within the executed data-driven strata? They do not measure magnitude stability, global feature-ranking stability, robustness to unseen structural breaks, or explanation correctness. Linear and Random Forest are comparatively stable only under this operational definition and within the limits of their explained model coverage.

### 5.3.3 Random Forest Seed Sensitivity

Random Forest provides the only complete two-seed RMSE table. Across 96 financial fold rows, the absolute RMSE difference between seeds 20261017 and 20261018 has median 0.0000402570075981, mean 0.0000558789806078, and maximum 0.000334055821701. The median relative difference is 0.257748%. These small descriptive differences indicate limited sensitivity of RF point forecasts to the two executed seeds.

This finding concerns forecast RMSE, not attribution stability. It cannot be generalized to XGBoost, LightGBM, LSTM, or TFT because comparable complete seed-level tables were not supplied. Two seeds also provide only a narrow view of stochastic variability.

## 5.4 Faithfulness and Economic Plausibility (RQ3)

### 5.4.1 SHAP Additivity

Linear Regression, Random Forest, and LightGBM have near-zero maximum absolute SHAP additivity residuals: 2.84217×10⁻¹⁴, 1.42109×10⁻¹³, and 2.27374×10⁻¹³, respectively. These results support numerical local-additivity consistency for the explained models. Linear’s exact linear mechanism makes this outcome especially expected, while the tree results apply to the single explained constituent rather than the two-seed forecast average.

XGBoost does not show uniform near-zero additivity. Its maximum residual is 0.215091253662109, its median is 0.0000279866991664336, and its mean is 0.0153693891694365 across 80 explained units. The distribution combines many small values with materially larger residuals. The available artifacts do not establish the cause, so the result is reported without attributing it to a particular library or numerical mechanism.

### 5.4.2 Deletion and Insertion Diagnostics

Table 5.5 and Figure 5.7 compare intervention outcomes. Linear succeeds in 33 of 80 units for both deletion and insertion. Random Forest succeeds in 47 of 80 deletion units and 44 of 80 insertion units. XGBoost succeeds in 45 of 80 for both, and LightGBM in 37 of 80 deletion and 38 of 80 insertion units. These fractions are mixed rather than uniformly favorable.

**Table 5.5**

*SHAP additivity and intervention diagnostics.*

| Model | Maximum absolute SHAP residual | Deletion success | Insertion success |
|---|---:|---:|---:|
| Linear | 2.842171×10⁻¹⁴ | 33/80 (41.25%) | 33/80 (41.25%) |
| Random Forest | 1.421085×10⁻¹³ | 47/80 (58.75%) | 44/80 (55.00%) |
| XGBoost | 2.150913×10⁻¹ | 45/80 (56.25%) | 45/80 (56.25%) |
| LightGBM | 2.273737×10⁻¹³ | 37/80 (46.25%) | 38/80 (47.50%) |
| LSTM* | Not applicable | 8/60 (13.33%) | 2/60 (3.33%) |
| TFT* | Not applicable | 30/60 (50.00%) | 34/60 (56.67%) |

*Note.* Success means that the model-attribution ordering outperforms the mean of 32 random orderings in an explained unit. Tree additivity applies to the explained constituent. *Deep intervention fractions are not like-for-like because model orderings contain four aggregated lags while random controls permute 84 flattened inputs; deep probes are also defective. Source: `faithfulness_metric_summary.csv`.

![Figure 5.7. SHAP additivity and deletion/insertion faithfulness diagnostics.](figures/Figure_5_5_Faithfulness.png)

*Note.* Intervention success means that the model-attribution ordering beats the mean of 32 random controls. Deep comparisons are not like-for-like because model orderings contain four aggregated lags and random controls permute 84 inputs.

The contrast between additivity and intervention outcomes is substantively important. Linear has essentially exact additivity but succeeds against the random-control mean in only 41.25% of explained units. Random Forest has near-zero additivity and the highest static deletion success, 58.75%, but its insertion success is 55.00%. A numerical decomposition property therefore does not guarantee success under the executed perturbation diagnostic.

LSTM’s exported deletion and insertion fractions are 8/60 and 2/60; TFT’s are 30/60 and 34/60. These values should not be interpreted as valid comparative faithfulness estimates. The model ordering covers four aggregated lags, whereas random controls permute all 84 flattened sequence inputs. The control task is not equivalent, and deep-probe defects and zero-baseline perturbations add further limitations.

### 5.4.3 Cross-Channel Agreement

Mean PFI–SHAP Spearman agreement is −0.065 for Linear Regression, 0.140 for Random Forest, 0.1825 for XGBoost, 0.070641 for LightGBM, and 0.033333 for LSTM. These are weak average relationships across four-feature rankings. They show that two common importance channels frequently prioritize different lags. Because PFI and SHAP respond to different aspects of model behavior and can be affected by dependence and perturbation choices, weak agreement does not identify which method is correct. Table 5.6 adds the median and range for each available channel comparison.

**Table 5.6**

*Cross-channel Spearman rank-agreement summaries.*

| Model | Channel comparison | n | Mean ρ | Median ρ | Minimum | Maximum |
|---|---|---:|---:|---:|---:|---:|
| Linear | PFI vs. SHAP | 80 | -0.065000 | -0.200000 | -1.000000 | 1.000000 |
| Random Forest | PFI vs. SHAP | 80 | 0.140000 | 0.300000 | -1.000000 | 1.000000 |
| XGBoost | PFI vs. SHAP | 80 | 0.182500 | 0.300000 | -1.000000 | 1.000000 |
| LightGBM | PFI vs. SHAP | 80 | 0.070641 | 0.200000 | -1.000000 | 1.000000 |
| LSTM | PFI vs. SHAP | 60 | 0.033333 | 0.100000 | -1.000000 | 1.000000 |
| Linear | Local ridge surrogate vs. SHAP | 80 | 0.925000 | 1.000000 | 0.400000 | 1.000000 |
| TFT | Native selection vs. post-hoc* | 60 | 0.330000 | 0.400000 | -1.000000 | 1.000000 |

*Note.* Correlations summarize four-feature ranking agreement and do not identify which channel is faithful. *TFT native/post-hoc agreement is exploratory because of the deep-probe defects. Source: `faithfulness_metric_summary.csv`.

Linear Regression’s LIME-style local ridge surrogate and SHAP rankings have mean agreement 0.925, median 1.0, and range 0.4 to 1.0 across 80 units. This high agreement is plausible for an approximately linear local mechanism. It does not demonstrate equivalent surrogate reliability for black-box models. TFT native-versus-post-hoc agreement averages 0.330, median 0.4, and ranges from −1 to 1 across 60 units. Again, disagreement cannot identify the faithful channel.

### 5.4.4 Economic Plausibility

The empirical feature set contains only lagged prices or values. A high attribution to `lag_1`, `lag_5`, `lag_21`, or `lag_252` indicates statistical use of a past observation within a fitted model. It does not establish a macroeconomic, liquidity, volatility, risk-premium, or causal mechanism. No external economic variables or causal identification design were executed, and no human-expert or ground-truth explanation evaluation was supplied.

The evidence therefore does not establish that post-hoc explanations capture genuine economic drivers. It also cannot prove that every attribution is spurious. The defensible conclusion is non-identification: the experiment describes predictive reliance but cannot discriminate genuine economic mechanisms from spurious correlations. This boundary applies equally to exact linear attributions, TreeSHAP, KernelSHAP, PFI, ARIMA parameters, and TFT native weights. Appendix A reports all 70 forecast summary cells, and Appendix B provides supplementary attribution, native-signal, and seed-sensitivity tables.

# 6. Discussion

The results support a differentiated account of trustworthy financial forecasting. Flexible models occupy the lowest exported RMSE region on the financial indices, but this pattern reverses on Sunspot and is not accompanied by uniformly positive R², stable attributions, or faithful explanations. Post-hoc methods expose useful aspects of model behavior, yet their outputs vary across models, diagnostics, and temporal strata. The discussion therefore treats predictive accuracy, intrinsic interpretability, post-hoc explainability, stability, faithfulness, and economic meaning as related but distinct criteria.

## 6.1 RQ1: The Accuracy–Interpretability Trade-off and the Mitigating Role of Post-hoc Explanation

The financial results are directionally consistent with studies showing that flexible machine learning can improve on simpler prediction rules under favorable conditions (Fischer & Krauss, 2018; Gu et al., 2020). LSTM and TFT have the lowest pooled exported RMSE and MAE at both financial horizons, and the pattern appears in each of the four index datasets. This consistency makes the descriptive result more informative than an isolated best fold. It would nevertheless be inappropriate to call it a statistically established complexity premium. The deep models use different origins and test lengths, and the available predictive-superiority test is invalid.

The static-model results also illustrate why the trade-off cannot be read as a simple linear relationship between complexity and accuracy. XGBoost and LightGBM are only marginally below Linear Regression in pooled financial RMSE, while Random Forest is slightly above it. The differences among these models are small relative to the shared forecasting difficulty. This resembles the caution in the reviewed literature that flexible methods do not automatically outperform linear or classical time series models, especially when the signal is weak and the evaluation design matters (James et al., 2013; Molnar, 2022; Tsay, 2005).

The Sunspot block is the clearest challenge to a universal hierarchy. Linear Regression’s RMSE is 13.139662 at h1, compared with 20.605463 for TFT and 90.360062 for LSTM. The same model family that occupies the strongest financial position therefore occupies the weakest Sunspot position. Although the normalized Sunspot provenance restricts external interpretation, the executed result remains decisive against a claim of universal model superiority within this study. Model performance is conditional on the dataset, target definition, and evaluation window.

R² and directional accuracy reinforce that conclusion. Every model–series–horizon cell has negative mean R² relative to the realized test-block mean, and 671 of 760 fold rows are negative. At the same time, Linear Regression has the highest pooled h5 directional accuracy. A model can consequently rank favorably on one descriptive metric and unfavorably on another. Forecast evaluation cannot be reduced to selecting whichever number appears most favorable. The negative R² values are particularly important because they prevent the deep models’ lower RMSE from being translated into a broad claim of strong predictive skill.

Post-hoc explanations mitigate the transparency problem only in a limited sense. SHAP, local surrogates, PFI, and native signals provide a way to inspect which lagged inputs a model uses. They can reveal disagreement between methods, variation across folds, and differences between native and post-hoc accounts. These are useful diagnostic functions. They do not transform an LSTM or TFT into an intrinsically interpretable model, because the prediction rule remains too complex to comprehend directly. This distinction follows the conceptual separation between transparent models and black boxes paired with explanation methods (Arsenault et al., 2025; Biecek & Burzykowski, 2021; Molnar, 2022).

The mitigation is also incomplete because explanation quality is not uniform. Linear and Random Forest have higher exported attribution-stability medians than the boosted-tree and deep channels. Additivity is near zero for some models but not uniformly for XGBoost. Cross-channel agreement is often weak. Thus, the explanation layer introduces its own reliability questions rather than closing the trust gap automatically.

The answer to RQ1 is therefore **partial and conditional mitigation, not elimination of the accuracy–interpretability trade-off**. A practitioner may obtain diagnostic visibility from post-hoc methods, and the financial RMSE results may justify considering deep models in a further matched evaluation. The current evidence does not establish that this visibility compensates reliably for lost intrinsic transparency, nor that the descriptive financial error advantage generalizes across datasets or meets a statistical superiority standard.

## 6.2 RQ2: Temporal Stability and Robustness to Market Regimes

The theoretical background predicts that non-stationarity and regime change can alter model behavior and therefore explanations (Aggarwal, 2015; Dixon et al., 2020; Tsay, 2005). The executed results support the need to evaluate that possibility, but not a universal crisis-degradation rule. Linear Regression and Random Forest retain median per-probe correlation 0.8 in every exported group, including bear and crisis. Their stability under the operational metric is comparatively high, not proof of invariance under all structural breaks.

The contrast with XGBoost and LightGBM is informative. Both have financial medians of 0.4 but Sunspot medians of 0.8. Similar predictive performance among static models therefore coexists with different attribution persistence. This separation matters for RQ1 as well as RQ2: forecast accuracy cannot serve as a proxy for explanation reliability. A model selection process that reports only RMSE would miss this difference.

The deep channels are lower or more variable in several groups. LSTM’s crisis median is 0.2, TFT KernelSHAP is 0.2 in bear and 0.3 in crisis, and TFT native variable selection is 0.1 in boundary and bull but 0.4 in crisis. The native TFT result is especially inconsistent with the assumption that crisis observations necessarily produce the lowest stability. More broadly, the results suggest that model mechanics and explanation channels moderate temporal behavior. This is compatible with the reviewed concern that explanation variance is a distinct evaluation problem (Habiba et al., 2026; Theissler, 2024).

The deep-probe defects prevent a stronger substantive ranking. Noncontiguous probe sequences do not represent valid model inputs, and the assigned output positions are incorrect after sequence construction. It would therefore be misleading to conclude that LSTM or TFT explanations are inherently unstable on the basis of these numbers alone. The appropriate interpretation is that the exported diagnostics do not establish deep-explanation robustness and that corrected experiments are needed.

The metric itself narrows the conclusion. Spearman correlation over four attribution coordinates produces a small set of possible values and is sensitive to ties. It describes local rank consistency for each probe across consecutive folds, not the stability of attribution magnitudes or global feature rankings. ARIMA illustrates this problem clearly: its median 1.0 arises from a vector with one nonzero entry and tied zeros and is substantively uninformative. Numerical perfection can therefore coexist with a weak construct.

The regime labels impose a second boundary. They are rolling-statistic strata defined by full-series quantiles, not dated historical crises. Full-sample thresholds use future distributional information, and crisis groups contain fewer correlation rows. The results do not demonstrate what would happen under a prospective regime classifier or around a specific event such as a dated market crash. The RF two-seed results also measure forecast-RMSE sensitivity, not explanation stability, and cannot be generalized to other model families.

The answer to RQ2 is therefore **limited, model- and channel-specific temporal stability**. Linear Regression and Random Forest are comparatively stable in the exported per-probe display; boosted trees are less stable on financial strata; and deep robustness remains unresolved because of defective probes. Universal bear- or crisis-regime degradation is not observed. Robustness to independently dated crises, arbitrary structural breaks, and alternative stability constructs remains untested.

## 6.3 RQ3: Faithfulness and Economic Reliability

The faithfulness results show why no single diagnostic can stand for overall explanation quality. Linear Regression, Random Forest, and LightGBM have near-zero maximum SHAP additivity residuals. This is meaningful evidence that their exported attribution decompositions satisfy numerical local additivity for the explained models. For Linear Regression, it is consistent with the exact linear mechanism. For the tree models, the claim applies to one constituent and not automatically to the two-seed forecast average.

XGBoost breaks any blanket exact-additivity claim. Its maximum residual is 0.215091253662109, despite a small median residual. Reporting only the median would hide the upper tail; reporting only the maximum would hide the many near-zero cases. The distribution therefore needs to be retained. Since the artifacts do not identify the cause, the thesis does not speculate about a software or approximation mechanism.

Intervention outcomes provide a different picture. Linear’s near-exact additivity coexists with deletion and insertion success of 33/80, while Random Forest achieves 47/80 and 44/80. If additivity were sufficient for practical faithfulness, these outcomes would be expected to align more closely. Instead, the results support the conceptual distinction between decomposing a prediction correctly and identifying a feature ordering that is useful under a specific perturbation design (Molnar, 2022; Theissler et al., 2022).

The deep intervention fractions cannot resolve this issue. Their four-lag model ordering and 84-input random controls make the comparison unequal. Low LSTM success or higher TFT success should not be interpreted as evidence that one deep model has more faithful explanations than the other. The result instead identifies a failed comparability condition in the diagnostic implementation.

Cross-channel agreement reinforces the same caution. PFI–SHAP correlations are weak on average, while Linear’s local ridge surrogate agrees strongly with SHAP. The latter is plausible for an approximately linear mechanism but does not validate surrogate explanations of complex models. TFT native and post-hoc weights agree only modestly on average. Disagreement does not reveal which channel is faithful; it reveals that the channels are not interchangeable.

The economic-driver question remains unanswered at a more fundamental level. The models use only lagged prices or values. Predictive importance cannot identify whether a lag reflects momentum, reversal, time-varying risk compensation, market microstructure, or noise. The authorized literature warns that explanation methods describe models rather than the world and that predictive estimates should not be read as structural mechanisms (Chan et al., 2022; Gill, 2020; Gu et al., 2020; Molnar, 2022). The executed design offers no causal variables or identification strategy that could overcome this limitation.

The answer to RQ3 is therefore **narrow, diagnostic-specific faithfulness for some static models, mixed intervention and cross-channel evidence, and no established economic-driver identification**. The thesis cannot conclude that the explanations capture genuine economic drivers, but it also cannot prove that all highlighted associations are spurious. It establishes non-identification rather than a causal verdict.

## 6.4 Practical and Regulatory Implications

The findings argue against a model-selection rule based on a single accuracy table. A defensible selection process should report the forecast target, fold coverage, benchmark definition, explanation channel, and diagnostic validity alongside RMSE. In this study, TFT might appear preferable from financial RMSE alone, Linear Regression from Sunspot RMSE or h5 directional accuracy, and Random Forest from parts of the stability and intervention evidence. These are not interchangeable definitions of best. The appropriate choice depends on which requirement is binding and whether the comparison is valid.

For model governance, post-hoc explanations are most useful as diagnostic inputs rather than as automatic approval evidence. They can prompt questions about changing feature reliance, disagreement between channels, or sensitivity to perturbations. They should not be used to certify that a black box is transparent, temporally robust, or economically justified without separate evidence. This caution is relevant to financial environments in which traceability and human oversight matter (Biecek & Burzykowski, 2021; Hoang et al., 2026; von Zahn et al., 2025).

The results also imply that explanation reports should include negative findings. A near-zero additivity residual can be reported, but it should be accompanied by intervention outcomes, seed coverage, and the fact that the explained model may differ from the forecast ensemble. A regime-stability median should be accompanied by the four-coordinate metric definition, full-sample regime thresholds, and group counts. Such reporting is consistent with the broader regulatory emphasis on traceability, although this thesis does not conduct a legal compliance assessment (Buttaboni & Floridi, 2026).

No computational-overhead conclusion can be drawn. A complete runtime, memory, and explanation-cost comparison was not exported, so claims that a particular model offers the best accuracy–interpretability–cost balance would be unsupported. Likewise, directional accuracy does not justify deployment or trading recommendations in the absence of transaction costs, a valid economic backtest, and risk analysis.

## 6.5 Threats to Validity

**Internal validity** is threatened most directly by implementation defects. ARIMA’s misalignment prevents matched interpretation, deep probe sequences undermine explanation validity, and the constant-loss Diebold–Mariano implementation prevents statistical inference. Different deep/static test windows also confound model-family comparisons with evaluation coverage. These are not merely reporting limitations; they restrict which causal and comparative statements the design can support.

**Construct validity** is limited because several measures do not capture the full concepts named in the research questions. Four-value Spearman correlation is only one form of local attribution stability. Additivity is only one form of numerical consistency. Deletion/insertion success depends on perturbation baselines and comparable controls. Data-driven regime strata are not equivalent to historical structural breaks. Most importantly, predictive attribution over four lagged inputs is not a measure of genuine economic driving mechanisms.

**Statistical conclusion validity** is weak by necessity. The available DM p-values are invalid, no valid seven-way Model Confidence Set exists, and observation-level predictions are unavailable for reconstruction. Multiple models, datasets, horizons, schemes, and diagnostics also create selection opportunities. The thesis responds by withholding significance claims and treating all rankings as descriptive rather than by selecting favorable tests.

**External validity** is restricted to the executed daily index data, normalized Sunspot input, four-lag feature set, and fixed configurations. The results do not establish performance for individual equities, other frequencies, alternative features, full TFT implementations, other market periods, or live trading. The unresolved Sunspot transformation further limits comparison with published Monash benchmarks. Chapter 7 sets out these limitations and their implications in full.

## 6.6 Research-Question Answer Status

The three research questions receive bounded answers rather than uniformly complete resolutions. Table 6.1 distinguishes the substantive answer supported by the exports from the evidence that would be required for a stronger conclusion. The classifications concern evidentiary completeness, not whether the thesis discusses each question.

**Table 6.1**

*Answer status and remaining evidentiary gaps for the three research questions.*

| Research question | Answer status | Evidence-bounded answer | Principal unresolved evidence |
|---|---|---|---|
| RQ1: accuracy–interpretability trade-off | **Partially answered** | Post-hoc methods provide partial and conditional diagnostic mitigation, but they do not eliminate the trade-off or make black-box models intrinsically interpretable. | Common forecast origins and target timestamps; valid predictive-superiority inference; explanations of the exact two-seed predictors; corrected deep diagnostics; comprehensive explanation overhead. |
| RQ2: temporal stability and regime robustness | **Partially answered** | Exported stability is limited and specific to the model and explanation channel. Universal bear- or crisis-regime degradation is not observed. | Valid contiguous deep probes and output mapping; informative ARIMA stability constructs; prospective and independently dated regimes; broader stability measures and replication. |
| RQ3: faithfulness and economic drivers | **Partially answered overall; economic-driver component not answered** | Numerical local-additivity consistency is supported for some explained static predictors, while intervention and cross-channel evidence is mixed. Genuine economic drivers are not identified. | Like-for-like deep interventions; full-predictor explanation coverage; resolved XGBoost residual mechanism; theoretically motivated economic variables and an explicit causal identification strategy. |

This classification prevents two opposite misreadings. The first would treat the presence of several XAI metrics as a complete validation of explanation trustworthiness. The second would treat implementation defects as evidence that no explanation has any diagnostic value. The exports support narrower conclusions: some numerical properties are verified, some descriptive patterns are informative, and several central constructs remain unresolved. A complete answer to RQ1 and RQ2 requires corrected and matched experiments. A complete answer to the economic component of RQ3 additionally requires a research design that goes beyond lag-only predictive attribution.

# 7. Limitations

This chapter consolidates the restrictions that determine how far the findings can be generalized. Some limitations arise from the deliberately narrow forecasting design, while others arise from defects discovered during the artifact audit. The distinction matters. A narrow but correctly implemented design can support bounded conclusions within its scope; a defective alignment, test, or explanation pipeline can invalidate the interpretation that its output was intended to support. Neither type of limitation is repaired retrospectively in this thesis. The original notebooks and exported values are preserved, and the conclusions are restricted accordingly.

## 7.1 Data-Related Limitations

### 7.1.1 Sample Composition and Market Coverage

The financial evidence covers four daily market indices: the S&P 500, DAX, NASDAQ Composite, and Dow Jones Industrial Average. These series provide broad market coverage across the United States and Germany, but they are not representative of every financial instrument or market. Index returns differ from individual equity returns in diversification, liquidity, volatility, and microstructure. The study therefore cannot establish how the same models and explanations would behave for individual stocks, less liquid securities, or markets with different trading institutions.

It also does not evaluate intraday data or other sampling frequencies. References to one-week, one-month, or one-year lags are interpretive labels for five, 21, and 252 trading sessions, not separate forecast designs at weekly, monthly, or yearly frequency.

The observation periods are long, but length does not eliminate dependence or guarantee coverage of every relevant market condition. The financial samples end in December 2025 and begin between 1990 and 1992, depending on the index. Observations close in time can share volatility conditions and overlapping five-session targets. Consequently, the number of exported rows should not be interpreted as a count of independent experimental replications. The rolling and expanding schemes supply temporal repetition, but their folds remain related through shared history and, for expanding windows, shared training observations.

The use of adjusted closing prices is appropriate for a consistent return series, but it does not reproduce all information available to an investor at each historical date. The supplied snapshots establish the values used by the notebooks, not a point-in-time database with vintage control. Corporate-action handling, data-provider revisions, and index-construction rules remain outside the experimental design. The thesis therefore evaluates forecasting on frozen provider snapshots rather than the complete information environment faced by a historical market participant.

### 7.1.2 Restricted Feature Set and Target Definition

Only four lagged price or value measurements enter the tabular models. This restriction supports comparability and makes attribution outputs easy to display, but it excludes many variables that could be relevant to financial forecasting and economic interpretation. There are no explicit measures of volatility, volume, liquidity, macroeconomic conditions, financing costs, or market sentiment. The models also receive no causal design that could separate predictable variation from structural economic mechanisms. Even an exact attribution to `lag_1` therefore identifies reliance on a previous return, not the reason that return predicts the target.

The financial target is a cumulative future log return over one or five sessions. It is not a price-level target, a trading return after costs, or a risk-adjusted investment outcome. At the five-session horizon, adjacent targets share four daily returns. This overlap limits independence assumptions and affects the interpretation of directional accuracy. A favorable sign agreement does not establish economically exploitable predictability, because magnitude, transaction costs, turnover, timing, and risk constraints are not included.

Sunspot serves as an executed non-financial robustness block rather than as a second financial market. Its target is a future native value, and its error units are not comparable with cumulative log-return errors. The reversal of model rankings on Sunspot is informative about dataset dependence within the experiment, but it cannot be interpreted as evidence about financial market behavior alone. Financial and Sunspot values are consequently kept separate throughout the thesis.

### 7.1.3 Unresolved Sunspot Provenance

The most serious data-provenance limitation concerns Sunspot. The supplied snapshot CSV reproduces the raw TSF values, but the normalized notebook input differs on 10,957 of 11,109 rows despite having the same dates and row count. The maximum absolute difference is 274 and the aligned correlation is approximately 0.406. No supplied transformation record explains these differences. All seven notebooks used the same normalized input, so the exported results are internally consistent with that file; however, they cannot be presented as verified forecasts of the supplied raw Monash series.

This discrepancy cannot be resolved by selecting whichever version appears more plausible. Replacing the notebook input with the raw archive would change the experiment and require retraining, which was not authorized. The thesis therefore reports the executed values and limits their external interpretation. Sunspot comparisons with published benchmark results, or claims about the physical Sunspot process, require a documented reconciliation of the input transformation before they can be made.

### 7.1.4 Data Quality and Missing Evidence

The audit found no duplicate dates, missing adjusted closes, nonpositive index prices, or missing values in the final modeling frames. These checks reduce concern about simple structural data errors, but they do not prove economic correctness or independence from provider revisions. The two financial data representations have matching dates and shared numerical fields, despite schema and serialization differences. This reconciliation supports the integrity of the index inputs. It does not compensate for the separate Sunspot conflict.

Observation-level predictions and realized targets were not exported as complete analysis files. Fold-level metrics are sufficient to reproduce the reported descriptive summaries, but not to reconstruct alternative forecast tests, error distributions at the observation level, or matched pairwise comparisons. This absence is a substantive evidence boundary rather than a documentation inconvenience. New inferential results cannot be recovered without authorized re-execution or additional original artifacts.

## 7.2 Methodological Limitations

### 7.2.1 Non-Matched Model Evaluation

The seven models do not share a fully matched evaluation design. Linear Regression, ARIMA, and the tree models use six folds per scheme and 252 test observations per fold. LSTM and TFT use four folds per scheme and 231 test observations after sequence construction. Deep rolling origins also differ from static rolling origins. As a result, a difference between pooled deep and static RMSE combines model behavior with differences in evaluation periods and test composition.

This limitation directly restricts RQ1. The financial tables support the observation that LSTM and TFT occupy the lowest exported RMSE region, but they do not identify a causal effect of architectural complexity or a statistically validated forecasting premium. A matched comparison would require the same origins, target timestamps, and evaluated observations for every included model. The thesis does not simulate such a comparison by discarding rows or adjusting exported values, because doing so would not recreate the missing observation-level errors.

Within the static models, rolling and expanding results are summarized without weighting. Financial pooled values are then unweighted means of four series-level means. These choices make the aggregation transparent and prevent long series from dominating solely because they contain more observations. They also mean that the pooled numbers are descriptive displays rather than estimates from a hierarchical sampling model. No confidence interval should be inferred from the number of cells alone.

### 7.2.2 ARIMA Index Misalignment

ARIMA has an additional implementation defect. Fold indices are constructed on a lag-trimmed modeling frame, but training and forecast-update arrays are sliced from the untrimmed return or value series. The same integer positions therefore refer to different timestamps across the fold definition and the ARIMA data array. This defect affects both the training segment and sequential forecast updates.

The consequence is stronger than a general caution about comparability. ARIMA’s exported metrics cannot be treated as results on the same targets used by the other models, even where fold sizes appear similar. Its values remain in the tables for completeness and audit transparency, but they are excluded from substantive matched rankings. The selected ARIMA orders still document what the executed fitting routine selected on its actual slices; they do not repair forecast-target alignment or provide a valid classical benchmark for the other models.

### 7.2.3 Temporal Splits and Inner Validation

The outer split uses a ten-observation pre-test gap formed by subtracting the nominal purge and embargo settings together. It does not implement a separate post-test embargo. This operational definition is documented precisely because calling it a symmetric leakage-prevention scheme would be inaccurate. The gap reduces direct adjacency between training and test rows, but it does not prove that all forms of temporal leakage or dependence are eliminated.

Boosted trees and neural networks use the final 10% of training observations or sequences for early stopping, with no additional purge between fitting and validation partitions. At the five-session horizon, target overlap near this internal boundary remains a concern. Models are not refit on the validation tail after early stopping. These choices affect both the fitted model and the interpretation of validation performance. They do not invalidate the outer test exports automatically, but they prevent the design from being described as comprehensively leakage-free.

The fixed model settings and reduced fold budgets also limit the optimization evidence. The notebooks do not execute the broad temporal hyperparameter searches described in the earlier methodology narrative. The results consequently evaluate particular configurations, not the best achievable configuration within each model family. A weak result should not be interpreted as proof that a model class can never succeed, just as a favorable result should not be generalized beyond the executed settings.

### 7.2.4 Invalid and Unavailable Statistical Inference

The exported Diebold–Mariano procedure constructs the model loss as a constant equal to the fold RMSE squared within each fold. It therefore does not use the observation-level squared forecast errors required for the intended equal-predictive-loss comparison. The resulting statistics and p-values are invalid for that purpose and are excluded from all inferential conclusions. Reporting them alongside valid metrics would create a false impression of statistical confirmation.

No valid seven-way Model Confidence Set, corrected predictive-superiority procedure, or multiplicity-adjusted ranking is available. The study compares many models, series, horizons, schemes, and explanation diagnostics, which increases the opportunity for selective interpretation. The appropriate response is not to choose the most favorable unadjusted test but to withhold significance claims. Descriptive lowest or highest values are always tied to a specified aggregation and never presented as statistically superior model classes.

The same restriction applies to economic and computational evaluation. No transaction-cost backtest, Sharpe-ratio analysis, deflated-Sharpe analysis, or comprehensive runtime and memory comparison was exported. Directional accuracy cannot substitute for investment performance, and isolated printed runtimes cannot support a full explanation-overhead ranking. These unexecuted analyses remain unavailable rather than appearing as empty placeholders or invented results.

### 7.2.5 Limited Stochastic Replication

Random Forest, XGBoost, and LightGBM forecasts average two seeds, while Linear Regression, ARIMA, LSTM, and TFT use one. Only Random Forest includes a complete two-seed RMSE sensitivity table. Its small observed seed differences support a narrow statement about RF point forecasts under the two executed seeds. They do not establish stochastic robustness for boosted trees or neural networks, and they do not measure seed sensitivity of explanations.

Two seeds would in any case provide limited coverage of training variability. Initialization, data ordering, early stopping, and software-level nondeterminism may affect flexible models. Because comparable per-seed evidence is unavailable for most models, stochastic uncertainty remains an unresolved component of the comparative results. The thesis therefore avoids treating small descriptive differences between models as stable rankings across reruns.

## 7.3 Explainability-Related Limitations

### 7.3.1 Explanation Coverage Differs from Forecast Coverage

Explanations are not computed for every forecast observation or every fitted fold. Static explanation tables cover 80 explained units per model and deep tables cover 60, whereas forecast coverage is broader. For tree models, the forecast is a two-seed average but the explanation addresses one constituent: the first seed for Random Forest and the last seed for XGBoost and LightGBM. An explanation can therefore be internally consistent for the explained constituent without being a complete account of the predictor that generated the exported forecast metrics.

This distinction limits all three research questions. For RQ1, the availability of an attribution plot does not imply complete transparency of the forecast ensemble. For RQ2, stability of one constituent does not establish stability of the two-seed averaged prediction rule. For RQ3, additivity for a constituent cannot automatically be transferred to the ensemble. These properties would need to be evaluated on the exact predictor used for forecasting before a stronger coverage claim could be made.

### 7.3.2 Defective Deep Explanation Probes

The deep explanation workflow selects spaced probe rows and passes them to a sequence constructor as though they formed contiguous 21-session histories. The resulting sequences do not represent the temporal inputs on which valid deep explanations should be conditioned. The subsequent assignment of probe positions also fails to account correctly for the lookback reduction in sequence outputs. These defects affect deep SHAP attributions, local-surrogate comparisons, temporal stability, and downstream intervention diagnostics.

Accordingly, the exported LSTM and TFT explanation values are exploratory artifacts, not validated estimates of explanation quality. Their lower or variable stability cannot be used to prove that deep explanations are inherently unstable. Conversely, apparently favorable TFT intervention outcomes cannot establish superior faithfulness. The defensible conclusion is that the current artifacts do not resolve deep-explanation reliability. Corrected contiguous probes and position mapping would require authorized re-execution; the thesis does not alter the saved results to approximate that correction.

### 7.3.3 Narrow and Imperfect Stability Construct

The stability statistic is a Spearman correlation between four attribution values for each probe across consecutive folds. It measures local rank consistency, not the stability of a global feature-importance vector, attribution magnitude, forecast behavior, or economic interpretation. Four coordinates produce a coarse set of possible correlation values, and ties can strongly influence the result. Repeated probes and fold pairs also mean that summary counts are not independent sample sizes.

ARIMA demonstrates the construct problem. Its median correlation of 1.0 arises from a native vector with effectively one nonzero AR entry and tied zeros. This numerical regularity does not demonstrate meaningful robustness of lag attributions, and ARIMA’s alignment defect further limits interpretation. More generally, a high correlation does not show that the highlighted feature is correct, stable in magnitude, or useful to a human reviewer.

Regime groups add another limitation. Bull, bear, crisis, and boundary labels derive from 63-session rolling statistics and full-series quantiles. Crisis overrides the other labels, and Sunspot is not assigned a financial regime. These groups are data-driven strata rather than independently dated historical events. Full-series thresholds use future distributional information, and crisis groups contain fewer correlation rows. The analysis therefore cannot establish prospective regime detection, robustness around a specific historical crash, or stability under every form of structural break.

### 7.3.4 Faithfulness Diagnostics Are Not Interchangeable

Near-zero SHAP additivity establishes a numerical decomposition property for the explained model. It does not establish that the attributed features are causally responsible for predictions, that perturbing them gives a valid counterfactual, or that the explanation is useful to a person. The mixed deletion and insertion outcomes show why these constructs must remain separate. Linear Regression has essentially exact additivity but limited intervention success under the executed random-control comparison.

Intervention diagnostics also depend on design choices. Zero-baseline replacement can create feature combinations or sequences unlike observed data, especially when lagged values are dependent. Success against the mean of 32 random orderings is a relative explained-fold indicator, not an absolute measure of truth. The deep implementation introduces a further mismatch: model-attribution orderings cover four aggregated lags while random controls permute all 84 flattened sequence inputs. The resulting deep success fractions are not like-for-like and cannot be compared directly with static outcomes or between deep models.

Cross-channel disagreement has similar interpretive limits. PFI and SHAP answer different questions and may respond differently to dependence and perturbation. Weak average rank agreement does not reveal which method is faithful. High Linear surrogate–SHAP agreement is compatible with an approximately linear mechanism but does not validate local surrogates for complex architectures. TFT native and post-hoc disagreement likewise identifies non-equivalence between channels, not a winner.

### 7.3.5 Economic and Human Validity Remain Untested

No diagnostic in the supplied artifacts establishes genuine economic drivers. The feature set contains only lagged prices or values, and the study includes no causal identification, external mechanism variables, or ground-truth economic explanations. It is therefore impossible to determine whether an attribution reflects a durable economic mechanism, a statistical proxy, a regime-specific association, or noise. The thesis concludes non-identification rather than claiming either confirmed economic meaning or universal spuriousness.

Human usefulness is also unevaluated. No expert review, user study, decision task, or comprehension measure was supplied. Stability and numerical consistency may be relevant to trustworthy use, but they do not show that analysts understand an explanation correctly or make better decisions with it. Regulatory implications in Chapter 6 are therefore governance principles suggested by the evidence, not evidence of legal compliance or operational approval.

## 7.4 Generalizability and External Validity

The findings are bounded by the executed datasets, sample periods, four-lag information set, forecast targets, fixed configurations, software environment, and simplified TFT adapter. The adapter should not be equated with every implementation of the Temporal Fusion Transformer architecture. Similarly, results for one LSTM configuration do not characterize all sequence models, and results for fixed tree settings do not establish optimal tree performance. Model-family language in the thesis always refers to the implemented representatives rather than an exhaustive evaluation of every possible specification.

The descriptive financial RMSE pattern is consistent across the four indices, which strengthens its internal empirical interest. It does not overcome non-matched deep/static windows, negative R² results, or unavailable inferential tests. The Sunspot reversal further demonstrates that the ranking is dataset dependent. Neither pattern should be extrapolated to other assets, frequencies, forecast targets, market periods, or live decision environments without new evaluation.

Source-level literature verification is also restricted. The original manuscript supplies 47 bibliography entries, but the accompanying RDF contains only the Tsay book record and no full texts for the other works. The final thesis therefore preserves the authorized citations and bibliographic details without claiming independent verification against every original publication. Direct quotations were removed where the source text was unavailable for verification. The journal name in the supplied Hu et al. entry remains explicitly marked as inferred rather than being presented as independently confirmed. These restrictions concern bibliographic and claim-level verification; they do not authorize adding replacement sources.

Finally, artifact integrity is not equivalent to independent computational reproducibility. The archives, manifests, input hashes, notebook structures, and saved outputs were verified, and no saved execution errors were found. However, no independent rerun was performed. A complete dependency lock, detailed hardware specification, and some execution-level evidence are unavailable. External downloads and local paths may also affect future execution. The final package therefore supports inspection and transparent re-execution instructions, but it does not certify that another environment will reproduce every exported value exactly.

These limitations do not make the study uninformative. They define the contribution as an evidence-audited, descriptive evaluation with explicit validity boundaries. The strongest conclusions concern the observed model- and diagnostic-specific patterns and the absence of support for universal claims. Broader statistical, causal, operational, and generalization claims remain subjects for separately designed and authorized research rather than conclusions of the present thesis.

# 8. Conclusion

## 8.1 Summary of the Research

This thesis examined whether explainable machine learning can support more trustworthy financial time series forecasting without treating predictive accuracy and explanation quality as interchangeable. It retained the three original research questions and evaluated seven executed forecasting approaches: Linear Regression, ARIMA, Random Forest, XGBoost, LightGBM, LSTM, and a simplified Temporal Fusion Transformer adapter. The empirical design covered four daily financial indices and one normalized Sunspot input, one- and five-session horizons, four lagged predictors, rolling and expanding evaluation, and a set of predictive and explanation diagnostics.

The study’s central argument is that trustworthiness is multidimensional. Forecast error, directional agreement, intrinsic interpretability, post-hoc visibility, temporal stability, numerical additivity, intervention response, and economic meaning measure different properties. A favorable value on one dimension cannot compensate automatically for weakness on another. This principle guided both the execution audit and the interpretation of results. Actual notebook code and exported tables took precedence over planned protocols and secondary narratives, and invalid or non-comparable evidence was retained as a limitation rather than converted into a favorable conclusion.

The predictive results were mixed. LSTM and TFT occupied the lowest descriptive financial RMSE and MAE region at both horizons, but their evaluation windows were not matched to those of the static models. Linear Regression performed best on the executed Sunspot input and recorded the highest pooled five-session directional accuracy. Mean R² was negative in every model–series–horizon cell, and no valid predictive-superiority inference was available. No universally superior model therefore emerged from the evidence.

The explanation findings were equally differentiated. Linear Regression and Random Forest showed comparatively high exported per-probe attribution-stability medians, whereas boosted-tree and deep channels were less stable or more variable under the operational metric. Additivity was near zero for the explained Linear, Random Forest, and LightGBM models, but not uniformly for XGBoost. Deletion and insertion results were mixed, cross-channel agreement was often weak, and deep explanation diagnostics were compromised by implementation defects. Most importantly, attributions over lagged prices or values could not identify genuine economic drivers.

## 8.2 Answers to the Research Questions

### 8.2.1 Answer to RQ1: The Accuracy–Interpretability Trade-off and Post-hoc Explanations

The first research question was:

> In financial time series forecasting, to what extent do post-hoc explanation methods (e.g., SHAP, LIME) mitigate the accuracy–interpretability trade-off between intrinsically interpretable models and black-box architectures (e.g., LSTMs, Temporal Fusion Transformers)?

**Answer status: RQ1 is partially answered. Post-hoc explanation methods mitigate the trade-off partially and conditionally; they do not eliminate it.** SHAP, local surrogate models, permutation importance, and native diagnostic signals make aspects of model behavior inspectable. They can reveal which lagged inputs receive attention or attribution, whether explanation channels disagree, and whether attributions change across folds. This diagnostic visibility is useful, particularly when a complex model is being considered alongside a simpler reference.

The mitigation remains incomplete for three reasons. First, attaching an explanation does not make the underlying LSTM or TFT intrinsically interpretable. The prediction mechanism remains distinct from the explanation produced about it. Second, the descriptive financial RMSE advantage of the deep models is not a statistically established or fully matched complexity premium, and the ranking reverses on Sunspot. Third, explanation quality is itself uncertain: stability, additivity, intervention outcomes, and cross-channel agreement vary by model and diagnostic.

The defensible conclusion is therefore not that explained black boxes are equivalent to transparent models, nor that simpler models always forecast better. Model choice depends on the dataset, evaluation comparability, required explanation property, and tolerance for unresolved uncertainty. Post-hoc XAI supplements model assessment; it does not remove the underlying trade-off.

### 8.2.2 Answer to RQ2: Temporal Stability and Regime Robustness of Attributions

The second research question was:

> To what extent are feature attributions from post-hoc XAI methods temporally stable and robust to non-stationarity, structural breaks, and regime shifts (e.g., bull vs. bear markets, crisis periods) in financial time series?

**Answer status: RQ2 is partially answered. The exported attributions are temporally stable only to a limited, model- and explanation-channel-specific extent.** Linear Regression and Random Forest have median consecutive-fold per-probe correlations of 0.8 in every exported regime group. XGBoost and LightGBM have financial medians of 0.4 but Sunspot medians of 0.8, showing that similar forecast performance can coexist with different attribution persistence. LSTM and TFT outputs are lower or more variable in several groups, but their defective probe construction prevents a firm conclusion about deep-explanation robustness.

The evidence does not support universal degradation in bear or crisis groups. Some channels retain their exported median in those groups, and TFT native variable selection is not least stable in crisis. This finding challenges a simple account in which adverse market conditions necessarily destabilize every explanation. Model mechanics, the explanation channel, and the operational metric all moderate the observed result.

The answer remains partial because the metric correlates only four attribution coordinates and does not measure magnitude stability or global-ranking stability. Regime labels are rolling-statistic strata based on full-series quantiles rather than independently dated historical events. ARIMA’s perfect medians are structurally uninformative, and deep-probe defects limit interpretation. Robustness to actual historical crises, unseen structural breaks, and prospective regime classification therefore remains untested.

### 8.2.3 Answer to RQ3: Faithfulness and Financial Reliability

The third research question was:

> To what extent are post-hoc explanations faithful to underlying model mechanics, and do they capture genuine economic drivers rather than spurious correlations?

**Answer status: RQ3 is partially answered overall, while its economic-driver component is not answered. Post-hoc explanations are faithful only in narrow, diagnostic-specific respects, and the study cannot establish that they capture genuine economic drivers.** Near-zero SHAP additivity residuals support numerical local-additivity consistency for the explained Linear Regression, Random Forest, and LightGBM models. XGBoost’s residual distribution prevents a uniform exact-additivity claim. For all tree models, the property applies to the explained constituent rather than automatically to the two-seed forecast ensemble.

Additivity does not settle practical or causal faithfulness. Deletion and insertion outcomes are mixed, including for Linear Regression despite its exact linear decomposition. PFI and SHAP rankings often agree weakly, while high surrogate–SHAP agreement for Linear Regression does not generalize to black-box models. Deep intervention comparisons are not like-for-like, and native TFT weights cannot be accepted as faithful merely because they originate within the architecture.

The economic component is unidentified by design. Lagged prices and values can reveal statistical reliance on past observations, but they cannot distinguish a genuine economic mechanism from a proxy, regime-specific association, or spurious correlation. The thesis therefore reaches a cautious negative conclusion: economic-driver identification is not established, and the artifacts also cannot prove that all attributions are spurious. Predictive explanation and causal economic explanation remain separate tasks.

## 8.3 Scientific and Practical Contributions

The first contribution is an integrated evaluation perspective that keeps forecasting and explanation constructs separate. Rather than treating XAI output as a decorative companion to RMSE, the thesis evaluates accuracy, directional agreement, R², attribution content, temporal stability, numerical consistency, intervention response, and cross-channel agreement as distinct forms of evidence. This structure provides a clear answer to the thesis problem: trustworthy forecasting cannot be inferred from a single performance score or a plausible attribution plot.

The second contribution is the comparative empirical synthesis across the seven executed models, two horizons, two temporal schemes, four financial indices, and the Sunspot robustness block. The results document both the lower descriptive financial errors of LSTM and TFT and the conditions that prevent a broad superiority claim. They also preserve the weak R² results, the Sunspot reversal, heterogeneous stability, nonzero XGBoost additivity residuals, and mixed intervention evidence. These negative and inconsistent findings are part of the study’s contribution because they show where simple narratives fail.

The third contribution is methodological and concerns evidence governance. The audit reconciled the actual notebook code with exported results, verified archive and input checksums, reconstructed the executed configurations, and established a claim-level evidence matrix. It identified the invalid Diebold–Mariano loss construction, ARIMA misalignment, non-matched deep/static evaluation, defective deep probes, unequal deep intervention controls, constituent-versus-ensemble explanation coverage, and unresolved Sunspot provenance. Making these defects visible improves the reliability of the final interpretation even though it reduces the strength of the conclusions.

The practical contribution is a model-assessment principle rather than a deployment recommendation. Financial model reports should identify the exact target, evaluated timestamps, benchmark definition, explanation coverage, and diagnostic validity. Near-zero additivity should not be reported without intervention evidence; a stability median should not be reported without its construct definition and group counts; and an attribution should not be described as an economic driver without an identification strategy. Such reporting supports traceability and human oversight but does not, by itself, certify legal compliance, investment value, or operational readiness.

## 8.4 Future Research

Future work should first correct the validity restrictions within the existing study rather than expand its scope prematurely. A matched evaluation would use common forecast origins and identical target timestamps for all seven models, retain observation-level predictions and realized targets, and apply valid predictive-superiority procedures with appropriate treatment of dependence and multiplicity. Correcting the ARIMA indexing and implementing the intended observation-level loss series would allow the classical and statistical comparisons to be interpreted as designed. These steps require separately authorized re-execution and were not performed in this thesis.

A second priority is explanation validity. Deep probes should contain genuine contiguous histories with correct sequence-output mapping, and model-attribution and random-control orderings should operate on the same input space. Explanations should be computed for the exact forecast predictor, including the complete two-seed ensemble where that ensemble generates the reported forecasts. More comprehensive seed replication would help distinguish model behavior from stochastic variation. Stability measures should include attribution magnitudes and clearly defined global rankings alongside local rank consistency.

A third priority is to strengthen regime and economic evaluation. Prospectively defined regime rules and independently dated event windows would provide a stronger test of robustness than full-sample quantile strata. Economic-driver research would require theoretically motivated variables and an explicit identification strategy rather than additional attribution plots alone. Human evaluation could then test whether explanations improve understanding or decisions, which numerical diagnostics cannot establish.

Finally, the Sunspot input transformation must be documented before that block can support external benchmark comparison. Independent reproduction should use a complete dependency specification, detailed hardware record, preserved observation-level outputs, and a controlled execution environment. Comprehensive runtime, memory, and explanation-overhead measurements would be needed before cost-related claims can be added. These are directions for future evidence collection, not claims completed by the present study.

In conclusion, explainable machine learning offers useful tools for inspecting financial forecasters, but trustworthy forecasting requires more than accurate predictions or visually convincing explanations. The strongest lesson of this thesis is methodological restraint: report what the executed evidence shows, preserve what it fails to show, and evaluate each dimension of trust on its own terms. Under that standard, post-hoc XAI is a valuable diagnostic component of financial model assessment, not a universal solution to the accuracy–interpretability trade-off, temporal instability, faithfulness, or economic uncertainty.

# References

Aggarwal, C. C. (2015). *Data mining: The textbook*. Springer. https://doi.org/10.1007/978-3-319-14142-8

Aggarwal, C. C. (2018). *Neural networks and deep learning: A textbook*. Springer.

Arsenault, P.-D., Wang, S., & Patenaude, J.-M. (2025). A survey of explainable artificial intelligence (XAI) in financial time series forecasting. *ACM Computing Surveys, 57*(10), Article 265. https://doi.org/10.1145/3729531

Ballarin, G. (2024). *Essays in time series econometrics and machine learning* [Doctoral dissertation, Universität Mannheim].

Bauer, L. (2025). *Forecast evaluation and combination with applications to financial risk* [Doctoral dissertation, Albert-Ludwigs-Universität Freiburg].

Biecek, P., & Burzykowski, T. (2021). *Explanatory model analysis: Explore, explain and examine predictive models*. CRC Press.

Brockwell, P. J., & Davis, R. A. (2016). *Introduction to time series and forecasting* (3rd ed.). Springer.

Bruce, P., Bruce, A., & Gedeck, P. (2020). *Practical statistics for data scientists: 50+ essential concepts using R and Python* (2nd ed.). O'Reilly Media.

Buttaboni, C., & Floridi, L. (2026). A regulatory taxonomy of AI opacity in the EU: Rethinking transparency, traceability, interpretability, and explainability. *AI and Ethics, 6*, Article 100. https://doi.org/10.1007/s43681-025-00940-0

Chan, F., Harris, M. N., Singh, R. B., & Yeo, W. B. E. (2022). Nonlinear econometric models with machine learning. In F. Chan & L. Mátyás (Eds.), *Econometrics with machine learning* (pp. 41–78). Springer. https://doi.org/10.1007/978-3-031-15149-1_2

Cochrane, J. H. (2005). *Asset pricing* (Rev. ed.). Princeton University Press.

Diebold, F. X. (2007). *Elements of forecasting* (4th ed.). Thomson South-Western.

Dixon, M. F., Halperin, I., & Bilokon, P. (2020). *Machine learning in finance: From theory to practice*. Springer. https://doi.org/10.1007/978-3-030-41068-1

Dorak, Ö., & Yakar, A. (2025). The impact of ESG factors on the propensity for dividends for European firms: A machine learning approach. In M. Gün & B. Kartal (Eds.), *Machine learning in finance: Trends, developments and business practices in the financial sector* (pp. 101–118). Springer.

Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research, 270*(2), 654–669. https://doi.org/10.1016/j.ejor.2017.11.054

Fourné, F. (2026). *Essays in macroeconomics: Forecasting and analyzing economic dynamics* [Doctoral dissertation, Ludwig-Maximilians-Universität München].

Gill, K. S. (2020). [Review of the book *The book of why: The new science of cause and effect*, by J. Pearl and D. Mackenzie]. *AI & Society, 35*, 767–768. https://doi.org/10.1007/s00146-020-00971-7

Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., & Montero-Manso, P. (2021). *Monash time series forecasting archive*. arXiv. https://doi.org/10.48550/arXiv.2105.06643

Green, A. (2026). *Deep learning in quantitative finance*. Wiley.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies, 33*(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009

Gün, M., & Kartal, B. (Eds.). (2025). *Machine learning in finance: Trends, developments and business practices in the financial sector*. Springer. https://doi.org/10.1007/978-3-031-83266-6

Habiba, M., Pearlmutter, B. A., & Maleki, M. (2026). *Recent trends in modelling the continuous time series using deep learning*. Springer. https://doi.org/10.1007/978-3-032-18022-3

Hoang, A., Phan, H., & Nguyen, V.-D. (2026). Explainable AI in finance: Enhancing transparency and interpretability of AI models in financial decision-making. In H. T. T. Nguyen, H. H. Phan, & V.-N. Huynh (Eds.), *Data science in finance and accounting* (pp. 193–212). Springer. https://doi.org/10.1007/978-3-032-06179-9_9

Hu, Y., Li, Y., Liu, P., Zhu, Y., Li, N., Dai, T., Xia, S., Cheng, D., & Jiang, C. (2026). FinTSB: A comprehensive and practical benchmark for financial time series forecasting. *Frontiers of Computer Science* [journal name inferred from DOI; not stated in file]. https://doi.org/10.1007/s11704-026-51064-5

Hull, I. (2021). *Machine learning for economics and finance in TensorFlow 2: Deep learning models for research and industry*. Apress. https://doi.org/10.1007/978-1-4842-6373-0

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: With applications in R*. Springer.

Kim, T. Y., Oh, H.-S., & Lim, Y. (2026). Forecasting high-dimensional non-normal time series using averaged quantile regression. *Journal of Time Series Econometrics, 18*(1), 49–61. https://doi.org/10.1515/jtse-2024-0014

Kirchgässner, G., Wolters, J., & Hassler, U. (2013). *Introduction to modern time series analysis* (2nd ed.). Springer.

Kraft, S. (2025). *Explainable AI for clinical decision support* [Doctoral dissertation, Eberhard Karls Universität Tübingen].

Kubat, M. (2015). *An introduction to machine learning*. Springer. https://doi.org/10.1007/978-3-319-20010-1

López de Prado, M. (2018). *Advances in financial machine learning*. Wiley.

López de Prado, M. (2020). *Machine learning for asset managers* (Elements in Quantitative Finance). Cambridge University Press. https://doi.org/10.1017/9781108883658

Mao, S.-Q., Meng, L.-X., Barbu, A., & Niu, X.-F. (2026). Is the financial market efficient? An advanced time series analysis. *Financial Innovation, 12*, Article 91. https://doi.org/10.1186/s40854-025-00856-y

Medeiros, M. C. (2022). Forecasting with machine learning methods. In F. Chan & L. Mátyás (Eds.), *Econometrics with machine learning* (pp. 111–146). Springer. https://doi.org/10.1007/978-3-031-15149-1_4

Molnar, C. (2022). *Model-agnostic interpretable machine learning* [Doctoral dissertation, Ludwig-Maximilians-Universität München].

Montgomery, D. C. (2017). *Design and analysis of experiments* (9th ed.). Wiley.

Nguyen, H. T. T. (2026). On explainable data science. In H. T. T. Nguyen, H. H. Phan, & V.-N. Huynh (Eds.), *Data science in finance and accounting* (pp. 1–14). Springer. https://doi.org/10.1007/978-3-032-06179-9_1

Saratchandran, D. V. (2025). Addressing compliance, bias, and transparency in AI systems. In S. K. Gupta, J. Rosak-Szyrocka, R. Rena, C.-S. Shieh, & G. Erkol Bayram (Eds.), *The impact of artificial intelligence on finance: Transforming financial technologies* (pp. 299–322). Springer. https://doi.org/10.1007/978-3-031-92916-8_15

Sert, M. F. (2025). A hybrid ARIMA-LSTM/GRU model for forecasting monthly trends in Turkey's gold and currency markets with a macro-economic data-driven approach. In M. Gün & B. Kartal (Eds.), *Machine learning in finance: Trends, developments and business practices in the financial sector* (pp. 35–52). Springer.

Sivasamy, S. (2025). Machine learning and predictive analytics in financial decision making. In S. K. Gupta, J. Rosak-Szyrocka, R. Rena, C.-S. Shieh, & G. Erkol Bayram (Eds.), *The impact of artificial intelligence on finance: Transforming financial technologies* (pp. 211–228). Springer. https://doi.org/10.1007/978-3-031-92916-8_11

Spinnato, F. (2023). *Explanation methods for sequential data models* [Doctoral dissertation, Scuola Normale Superiore, Pisa].

Stankiewicz, S. (2015). *Forecasting and econometric modelling of macroeconomic and financial time series* [Doctoral dissertation, University of Konstanz]. KOPS. http://nbn-resolving.de/urn:nbn:de:bsz:352-0-296747

Supsermpol, P., Huynh, V.-N., & Chiadamrong, N. (2026). Post-IPO performance prediction: A comparative study of logistic regression and machine learning techniques for Thai IPO firms. In H. T. T. Nguyen, H. H. Phan, & V.-N. Huynh (Eds.), *Data science in finance and accounting* (pp. 99–132). Springer. https://doi.org/10.1007/978-3-032-06179-9_6

Theissler, A. (2024). Explaining is not enough: Four open problems in explainable AI for AI with user interaction [Extended abstract]. *AAAI 2024 Spring Symposium on User-Aligned Assessment of Adaptive AI Systems*, Stanford University.

Theissler, A., Spinnato, F., Schlegel, U., & Guidotti, R. (2022). Explainable AI for time series classification: A review, taxonomy and research directions. *IEEE Access, 10*, 100700–100724. https://doi.org/10.1109/ACCESS.2022.3207765

Tsay, R. S. (2005). *Analysis of financial time series* (2nd ed.). Wiley.

von Zahn, M., Liebich, L., Jussupow, E., Hinz, O., & Bauer, K. (2025). *Knowing (not) to know: Explainable artificial intelligence and human metacognition* (SAFE Working Paper No. 464). Leibniz Institute for Financial Research SAFE.

# Appendices

## Appendix A: Complete Forecast Summary Cells

Table A.1 retains every model–series–horizon cell used in the empirical synthesis. Each row summarizes the completed rolling and expanding fold–scheme rows in the corresponding exported `forecast_results.csv`. RMSE and MAE use cumulative log-return units for financial series and native Sunspot-number units for Sunspot. DA is undefined for Sunspot and is shown as an em dash. R² uses the realized test-block mean as its reference. ARIMA’s raw/frame alignment is defective; deep models have non-matched origins and 231-observation test blocks. Displayed decimals are rounded for readability; the machine-readable evidence tables retain full exported precision.

**Table A.1**

*Verified descriptive forecast metrics for all 70 model–series–horizon cells.*

| Model | Series | h | Fold rows | Test n | RMSE mean | RMSE SD | MAE mean | DA mean | R² mean | Negative R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear | S&P 500 | 1 | 12 | 252 | 0.010277 | 0.004975 | 0.007384 | 0.520172 | -0.031136 | 11/12 |
| Linear | S&P 500 | 5 | 12 | 252 | 0.021537 | 0.009109 | 0.016264 | 0.584987 | -0.053170 | 11/12 |
| Linear | DAX | 1 | 12 | 252 | 0.011788 | 0.004678 | 0.008757 | 0.520503 | -0.008887 | 8/12 |
| Linear | DAX | 5 | 12 | 252 | 0.026140 | 0.011691 | 0.019616 | 0.551587 | -0.025063 | 12/12 |
| Linear | NASDAQ Composite | 1 | 12 | 252 | 0.012917 | 0.005110 | 0.009424 | 0.529101 | -0.029454 | 11/12 |
| Linear | NASDAQ Composite | 5 | 12 | 252 | 0.028363 | 0.009218 | 0.021703 | 0.567791 | -0.048821 | 11/12 |
| Linear | DJIA | 1 | 12 | 252 | 0.010921 | 0.004803 | 0.007900 | 0.502976 | -0.031037 | 11/12 |
| Linear | DJIA | 5 | 12 | 252 | 0.023106 | 0.008879 | 0.017417 | 0.562831 | -0.029515 | 9/12 |
| Linear | Sunspot | 1 | 12 | 252 | 13.139662 | 6.288355 | 10.741528 | — | -0.137958 | 7/12 |
| Linear | Sunspot | 5 | 12 | 252 | 13.108861 | 6.371943 | 10.751450 | — | -0.145780 | 7/12 |
| ARIMA | S&P 500 | 1 | 12 | 252 | 0.010350 | 0.004830 | 0.007417 | 0.512897 | -0.059463 | 12/12 |
| ARIMA | S&P 500 | 5 | 12 | 252 | 0.021417 | 0.009159 | 0.016258 | 0.584987 | -0.037582 | 12/12 |
| ARIMA | DAX | 1 | 12 | 252 | 0.011821 | 0.004684 | 0.008775 | 0.520172 | -0.015968 | 11/12 |
| ARIMA | DAX | 5 | 12 | 252 | 0.026246 | 0.011587 | 0.019743 | 0.524140 | -0.038317 | 11/12 |
| ARIMA | NASDAQ Composite | 1 | 12 | 252 | 0.013162 | 0.004897 | 0.009668 | 0.494378 | -0.085940 | 12/12 |
| ARIMA | NASDAQ Composite | 5 | 12 | 252 | 0.028255 | 0.009171 | 0.021865 | 0.532407 | -0.040086 | 12/12 |
| ARIMA | DJIA | 1 | 12 | 252 | 0.010964 | 0.004595 | 0.007924 | 0.517196 | -0.055245 | 12/12 |
| ARIMA | DJIA | 5 | 12 | 252 | 0.023033 | 0.008885 | 0.017370 | 0.574405 | -0.020697 | 9/12 |
| ARIMA | Sunspot | 1 | 12 | 252 | 31.655317 | 11.538604 | 29.821034 | — | -10.913093 | 12/12 |
| ARIMA | Sunspot | 5 | 12 | 252 | 32.105515 | 11.772600 | 30.272255 | — | -11.402161 | 12/12 |
| Random Forest | S&P 500 | 1 | 12 | 252 | 0.010292 | 0.004851 | 0.007414 | 0.502976 | -0.044975 | 9/12 |
| Random Forest | S&P 500 | 5 | 12 | 252 | 0.021790 | 0.009307 | 0.016500 | 0.554233 | -0.075483 | 10/12 |
| Random Forest | DAX | 1 | 12 | 252 | 0.011904 | 0.004650 | 0.008858 | 0.508267 | -0.033225 | 10/12 |
| Random Forest | DAX | 5 | 12 | 252 | 0.026326 | 0.011574 | 0.019732 | 0.544312 | -0.046063 | 10/12 |
| Random Forest | NASDAQ Composite | 1 | 12 | 252 | 0.012990 | 0.004992 | 0.009498 | 0.520172 | -0.047888 | 12/12 |
| Random Forest | NASDAQ Composite | 5 | 12 | 252 | 0.028766 | 0.009550 | 0.022036 | 0.559193 | -0.076156 | 12/12 |
| Random Forest | DJIA | 1 | 12 | 252 | 0.010929 | 0.004635 | 0.007935 | 0.491402 | -0.042159 | 11/12 |
| Random Forest | DJIA | 5 | 12 | 252 | 0.023212 | 0.008845 | 0.017532 | 0.548611 | -0.040487 | 10/12 |
| Random Forest | Sunspot | 1 | 12 | 252 | 23.210541 | 8.956741 | 20.412900 | — | -3.168120 | 12/12 |
| Random Forest | Sunspot | 5 | 12 | 252 | 23.106027 | 8.582587 | 20.287377 | — | -3.172917 | 12/12 |
| XGBoost | S&P 500 | 1 | 12 | 252 | 0.010155 | 0.004860 | 0.007252 | 0.526786 | -0.010626 | 10/12 |
| XGBoost | S&P 500 | 5 | 12 | 252 | 0.021494 | 0.009254 | 0.016277 | 0.568122 | -0.043251 | 11/12 |
| XGBoost | DAX | 1 | 12 | 252 | 0.011810 | 0.004697 | 0.008785 | 0.509921 | -0.012669 | 10/12 |
| XGBoost | DAX | 5 | 12 | 252 | 0.026184 | 0.011672 | 0.019596 | 0.548280 | -0.029297 | 9/12 |
| XGBoost | NASDAQ Composite | 1 | 12 | 252 | 0.012808 | 0.004947 | 0.009336 | 0.547619 | -0.017413 | 11/12 |
| XGBoost | NASDAQ Composite | 5 | 12 | 252 | 0.028337 | 0.009355 | 0.021730 | 0.554233 | -0.044162 | 12/12 |
| XGBoost | DJIA | 1 | 12 | 252 | 0.010796 | 0.004675 | 0.007776 | 0.515873 | -0.009757 | 10/12 |
| XGBoost | DJIA | 5 | 12 | 252 | 0.023040 | 0.008897 | 0.017430 | 0.563823 | -0.020829 | 9/12 |
| XGBoost | Sunspot | 1 | 12 | 252 | 30.331053 | 14.582567 | 27.754931 | — | -7.309703 | 12/12 |
| XGBoost | Sunspot | 5 | 12 | 252 | 29.110530 | 14.022597 | 26.490964 | — | -6.111804 | 12/12 |
| LightGBM | S&P 500 | 1 | 12 | 252 | 0.010160 | 0.004864 | 0.007263 | 0.529101 | -0.011222 | 10/12 |
| LightGBM | S&P 500 | 5 | 12 | 252 | 0.021514 | 0.009274 | 0.016313 | 0.568452 | -0.044326 | 11/12 |
| LightGBM | DAX | 1 | 12 | 252 | 0.011815 | 0.004683 | 0.008783 | 0.518188 | -0.014490 | 10/12 |
| LightGBM | DAX | 5 | 12 | 252 | 0.026111 | 0.011611 | 0.019557 | 0.540344 | -0.025042 | 10/12 |
| LightGBM | NASDAQ Composite | 1 | 12 | 252 | 0.012823 | 0.004937 | 0.009349 | 0.549934 | -0.020065 | 11/12 |
| LightGBM | NASDAQ Composite | 5 | 12 | 252 | 0.028291 | 0.009308 | 0.021708 | 0.555225 | -0.041007 | 12/12 |
| LightGBM | DJIA | 1 | 12 | 252 | 0.010793 | 0.004675 | 0.007787 | 0.499339 | -0.009281 | 9/12 |
| LightGBM | DJIA | 5 | 12 | 252 | 0.023061 | 0.008856 | 0.017471 | 0.548611 | -0.025225 | 9/12 |
| LightGBM | Sunspot | 1 | 12 | 252 | 27.486915 | 12.957976 | 24.921860 | — | -5.558392 | 12/12 |
| LightGBM | Sunspot | 5 | 12 | 252 | 27.401196 | 12.548013 | 24.799042 | — | -5.691949 | 12/12 |
| LSTM | S&P 500 | 1 | 8 | 231 | 0.008648 | 0.003722 | 0.006391 | 0.512446 | -0.037728 | 8/8 |
| LSTM | S&P 500 | 5 | 8 | 231 | 0.019020 | 0.007289 | 0.014595 | 0.565476 | -0.101187 | 6/8 |
| LSTM | DAX | 1 | 8 | 231 | 0.009716 | 0.002503 | 0.007382 | 0.525974 | -0.029089 | 6/8 |
| LSTM | DAX | 5 | 8 | 231 | 0.021774 | 0.005888 | 0.016739 | 0.530844 | -0.073533 | 7/8 |
| LSTM | NASDAQ Composite | 1 | 8 | 231 | 0.011520 | 0.004541 | 0.008519 | 0.560606 | -0.020320 | 6/8 |
| LSTM | NASDAQ Composite | 5 | 8 | 231 | 0.025935 | 0.007837 | 0.020080 | 0.563312 | -0.066090 | 8/8 |
| LSTM | DJIA | 1 | 8 | 231 | 0.009572 | 0.002903 | 0.007047 | 0.518939 | -0.039169 | 6/8 |
| LSTM | DJIA | 5 | 8 | 231 | 0.021769 | 0.005016 | 0.016839 | 0.556818 | -0.174556 | 7/8 |
| LSTM | Sunspot | 1 | 8 | 231 | 90.360062 | 75.831595 | 89.412689 | — | -67.605613 | 8/8 |
| LSTM | Sunspot | 5 | 8 | 231 | 90.260971 | 75.850316 | 89.314367 | — | -65.938777 | 8/8 |
| TFT | S&P 500 | 1 | 8 | 231 | 0.008580 | 0.003545 | 0.006327 | 0.502165 | -0.034894 | 5/8 |
| TFT | S&P 500 | 5 | 8 | 231 | 0.018911 | 0.007194 | 0.014462 | 0.571429 | -0.091645 | 6/8 |
| TFT | DAX | 1 | 8 | 231 | 0.009705 | 0.002564 | 0.007380 | 0.520563 | -0.023126 | 8/8 |
| TFT | DAX | 5 | 8 | 231 | 0.021922 | 0.005708 | 0.016898 | 0.545996 | -0.099726 | 7/8 |
| TFT | NASDAQ Composite | 1 | 8 | 231 | 0.011457 | 0.004513 | 0.008505 | 0.556277 | -0.009596 | 7/8 |
| TFT | NASDAQ Composite | 5 | 8 | 231 | 0.025835 | 0.008300 | 0.019899 | 0.574675 | -0.049256 | 7/8 |
| TFT | DJIA | 1 | 8 | 231 | 0.009474 | 0.002851 | 0.006941 | 0.527597 | -0.017376 | 6/8 |
| TFT | DJIA | 5 | 8 | 231 | 0.021327 | 0.005559 | 0.016411 | 0.538420 | -0.096102 | 7/8 |
| TFT | Sunspot | 1 | 8 | 231 | 20.605463 | 10.218561 | 18.323245 | — | -6.284478 | 8/8 |
| TFT | Sunspot | 5 | 8 | 231 | 20.355241 | 10.521255 | 18.057893 | — | -3.805122 | 7/8 |

*Note.* Fold rows are 12 for static models and eight for deep models. Test n is 252 for static models and 231 for deep models. Negative R² is reported as negative rows/complete rows. Source: `empirical_evidence_master_table.csv`; row-level provenance is preserved in `empirical_fold_results_consolidated.csv`.

## Appendix B: Supplementary Explanation Evidence

Table B.1 reports the complete top-rank distributions underlying Figure 5.5.

**Table B.1**

*Distribution of top-ranked mean absolute SHAP features.*

| Model | Feature | Top-rank count | Explained units | Share |
|---|---|---:|---:|---:|
| Linear | lag_1 | 33 | 80 | 41.25% |
| Linear | lag_5 | 14 | 80 | 17.50% |
| Linear | lag_21 | 23 | 80 | 28.75% |
| Linear | lag_252 | 10 | 80 | 12.50% |
| Random Forest | lag_1 | 29 | 80 | 36.25% |
| Random Forest | lag_5 | 13 | 80 | 16.25% |
| Random Forest | lag_21 | 16 | 80 | 20.00% |
| Random Forest | lag_252 | 22 | 80 | 27.50% |
| XGBoost | lag_1 | 10 | 80 | 12.50% |
| XGBoost | lag_5 | 17 | 80 | 21.25% |
| XGBoost | lag_21 | 23 | 80 | 28.75% |
| XGBoost | lag_252 | 30 | 80 | 37.50% |
| LightGBM | lag_1 | 14 | 80 | 17.50% |
| LightGBM | lag_5 | 13 | 80 | 16.25% |
| LightGBM | lag_21 | 29 | 80 | 36.25% |
| LightGBM | lag_252 | 24 | 80 | 30.00% |
| LSTM | lag_1 | 20 | 60 | 33.33% |
| LSTM | lag_5 | 6 | 60 | 10.00% |
| LSTM | lag_21 | 19 | 60 | 31.67% |
| LSTM | lag_252 | 15 | 60 | 25.00% |
| TFT | lag_1 | 19 | 60 | 31.67% |
| TFT | lag_5 | 10 | 60 | 16.67% |
| TFT | lag_21 | 17 | 60 | 28.33% |
| TFT | lag_252 | 14 | 60 | 23.33% |

*Note.* Deep rankings are exploratory. Tree explanations characterize one constituent of each two-seed forecast ensemble. Source: `dominant_shap_feature_counts.csv`.

Table B.2 documents the selected ARIMA orders without using them as evidence of valid forecast alignment.

**Table B.2**

*AIC-selected ARIMA orders across 120 forecast rows.*

| ARIMA order | Selected rows |
|---|---:|
| (1, 0, 0) | 40 |
| (2, 0, 0) | 18 |
| (1, 0, 1) | 20 |
| (2, 0, 1) | 6 |
| (3, 0, 0) | 36 |
| (1, 1, 0) | 0 |
| (2, 1, 0) | 0 |

*Note.* These counts document the executed fitting routine on its actual raw-array slices; they do not resolve ARIMA’s forecast-target misalignment. Source: `ARIMA_study/tables/native_params.csv`.

Table B.3 summarizes the native TFT variable-selection channel for comparison with the post-hoc rankings.

**Table B.3**

*TFT mean native variable-selection weights.*

| Feature | Mean native weight |
|---|---:|
| lag_1 | 0.263653 |
| lag_5 | 0.217046 |
| lag_21 | 0.229127 |
| lag_252 | 0.290175 |

*Note.* Means cover 240 feature rows representing 60 explained units × four features. Weights are native signals, not validated causal or faithful explanations. Source: `TFT_study/tables/shap_importance.csv`.

Table B.4 retains the RF-specific two-seed forecast sensitivity statistics.

**Table B.4**

*Random Forest forecast sensitivity to the two executed seeds.*

| Statistic | Verified value |
|---|---:|
| Paired financial fold rows | 96 |
| Median absolute RMSE difference | 0.0000402570075981 |
| Mean absolute RMSE difference | 0.0000558789806078 |
| Maximum absolute RMSE difference | 0.000334055821701 |
| Median relative RMSE difference | 0.257748% |

*Note.* Seeds are 20261017 and 20261018. Relative difference divides the absolute RMSE difference by the two-seed mean RMSE for the same fold. These statistics concern RF point forecasts only, not attribution stability or other models. Source: `RandomForest_study/tables/seed_stability.csv`.

## Appendix C: Notebook and Configuration Guide

The seven original notebooks are preserved byte for byte under `Supporting_Materials/Original_Notebooks/` in the revised package. Separate documentation-revised copies are provided under `Notebooks/`; their code cells, saved outputs, execution counts, and top-level metadata are identical to the originals. Both sets were parsed and audited without execution. Saved execution counts and output errors are structural observations from the `.ipynb` files, not a new independent rerun. Table C.1 identifies each original notebook’s structure and experimental role.

**Table C.1**

*Original notebook inventory and executed roles; documentation-revised copies retain the same code and outputs.*

| Notebook | Model | Cells | Code cells | Executed code cells | Saved errors | Executed role |
|---|---|---:|---:|---:|---:|---|
| 01_Linear_Regression.ipynb | Linear Regression | 19 | 10 | 8 | 0 | OLS; exact coefficients/SHAP; PFI; ALE; local ridge surrogate; interventions |
| 02_ARIMA.ipynb | ARIMA | 15 | 9 | 6 | 0 | AIC-selected ARIMA; native parameters; alignment defect retained |
| 03_Random_Forest.ipynb | Random Forest | 17 | 10 | 7 | 0 | 100 trees, depth 10, leaf 20; two-seed forecast average; first-seed explanation |
| 04_XGBoost.ipynb | XGBoost | 13 | 8 | 5 | 0 | Fixed boosted-tree configuration; two-seed forecast average; last-seed explanation |
| 05_LightGBM.ipynb | LightGBM | 13 | 8 | 5 | 0 | Matching fixed boosted-tree configuration; callback early stopping; last-seed explanation |
| 06_LSTM.ipynb | LSTM | 13 | 8 | 5 | 0 | Lookback 21; hidden 64; two layers; one seed; deep explanation defects retained |
| 07_Temporal_Fusion_Transformer.ipynb | Simplified TFT | 14 | 8 | 6 | 0 | Variable-selection softmax, LSTM encoder, four-head attention; one seed |

Table C.2 provides the file-identity checksums needed to verify their preservation.

**Table C.2**

*SHA-256 checksums of the preserved original notebooks.*

| Notebook | SHA-256 |
|---|---|
| 01_Linear_Regression.ipynb | `5f9d19ae3b1ce50294e928489dc96f6fab3b1626f437722e9738fe14a98abfaa` |
| 02_ARIMA.ipynb | `196907667f72e350378b80f6e6208e4a194306fe19dabf9a2244222a4a85e5c3` |
| 03_Random_Forest.ipynb | `d75783a0aff33072f3dc33d6491b1c44e24c96f7829aacd37050eeaa8c0dc1b1` |
| 04_XGBoost.ipynb | `be7dffa1fd7e2da40ef1081985e5f987c9a8161d658b0988e046208b301d800f` |
| 05_LightGBM.ipynb | `fb07fb32e4bc8ca49aa8921273c2561deec5b9752865aa62799c787360bb3c16` |
| 06_LSTM.ipynb | `54b6a6d438a29f05c24b67b50f69e0f66cb6efcdafad7b9ea25e3335d723c98e` |
| 07_Temporal_Fusion_Transformer.ipynb | `7bd31541a1a7c9546b140ab52ba84fe6e8ad977600868d356b527f63b8b956d1` |

*Note.* The hashes identify the exact supplied original files. They establish file identity and preservation, not computational reproducibility. A separate preservation report verifies the revised copies against these originals. Shared settings are seed 20261017, lags [1, 5, 21, 252], horizons [1, 5], test span 252, minimum training window 500, purge 5, embargo 5, rolling and expanding schemes, and `THESIS_FULL_BUDGET=False`.

## Appendix D: Provenance, Evidence References, and Reproducibility

The artifact audit used CRC and safe-path checks before extraction. Every checksum listed in the model-study manifests matched, as did all 18 hashes in the raw-data input manifest. Checksum agreement confirms the integrity and identity of supplied artifacts; it does not prove methodological validity or independent computational reproducibility. Table D.1 records the model-archive checks.

**Table D.1**

*Model-archive integrity and manifest verification.*

| Archive | Extracted files | Manifest entries verified | Archive SHA-256 |
|---|---:|---:|---|
| Linear_study.zip | 18 | 16 | `d4647f1cd6bbf163b93721647b4aa9453c0b251850622b37f7177d723876491d` |
| ARIMA_study.zip | 15 | 13 | `531b657d733f2e30dba3229d327ff1f183fe7cccecf7d25047c4bc154b77a8c4` |
| RandomForest_study.zip | 16 | 14 | `617887f7ee76751aabd93d4fa748f2af764794385bf54e693859439e3d97c953` |
| XGBoost_study.zip | 15 | 13 | `b8639eeb21ffda74ffd15fe622a972e990a732fe467795c65ea640477353b499` |
| LightGBM_study.zip | 15 | 13 | `03c293caeab9ad42e8f15f5d877fe39e3546d55d9096c0326e634df222e0bda6` |
| LSTM_study.zip | 15 | 13 | `16b003753cfcd5d9cec4e4d610fdd45890828a86ca24bdb946c46ed90ad141bd` |
| TFT_study.zip | 16 | 14 | `e1f0c9c2c0f30ecaffdc9478bfd3dad72fae17115e1f7a92d8b089d515a9fc04` |

Table D.2 maps the consolidated evidence files to their permitted thesis uses.

**Table D.2**

*Principal evidence files and their thesis use.*

| Evidence file | Verified content | Principal thesis use |
|---|---|---|
| `empirical_fold_results_consolidated.csv` | 760 forecast fold–scheme rows with source paths | Reproduction of forecast aggregates and negative-R² counts |
| `empirical_evidence_master_table.csv` | 70 model–series–horizon cells and 34 evidence fields | Tables 5.1, 5.2, A.1 and financial/Sunspot figures |
| `financial_model_horizon_summary.csv` | 14 model–horizon financial pooled rows | Table 5.1 and RQ1 synthesis |
| `stability_results_consolidated.csv` / `stability_summary_consolidated.csv` | Per-probe correlations and 40 regime/channel summaries | Table 5.4 and Figure 5.6 |
| `faithfulness_results_consolidated.csv` / `faithfulness_metric_summary.csv` | 3,981 metric rows and 55 summaries | Tables 5.5, 5.6 and Figure 5.7 |
| `shap_importance_consolidated.csv` / `dominant_shap_feature_counts.csv` | Attribution rows and 24 model–feature top-rank counts | Figure 5.5 and Table B.1 |
| `Research Question Evidence Matrix.csv` | 50 unique claim records: RQ1 14, RQ2 13, RQ3 23 | Claim-level provenance and limitations |
| `dataset_structure_audit.csv` | Five series structures, periods, and quality checks | Table 4.1 and data audit |

The notebooks record Python 3.13.5 on macOS ARM. A complete dependency lock, detailed hardware specification, complete observation-level prediction files, and an independent rerun are unavailable. The accompanying README gives execution order and package structure without claiming verified reproduction. The normalized Sunspot transformation remains undocumented. The supplied RDF covers only Tsay (2005); other bibliography metadata and source-level claims are retained from the authorized manuscript with the verification limitation disclosed in Chapter 7.

## Appendix E: Notation and Abbreviations

Table E.1 defines the mathematical notation used in the methodology and results.

**Table E.1**

*Principal mathematical notation.*

| Symbol | Meaning |
|---|---|
| $P_t$ | Adjusted closing price at session $t$ |
| $r_t = \ln P_t - \ln P_{t-1}$ | One-session log return |
| $h \in \{1,5\}$ | Executed forecast horizon in sessions |
| $y_{t,h}$ | Cumulative future log return for financial series; native future value for Sunspot |
| $v_t$ | Native Sunspot input value at time $t$ |
| $x_{t,\ell}$ | Lagged return or value at lag $\ell$ |
| $\ell \in \{1,5,21,252\}$ | Executed predictor lags |
| $\hat y_i$ | Model forecast for observation $i$ |
| $N$ | Number of evaluated observations in a fold |
| $K$ | Number of folds per scheme: six static, four deep |
| $\rho$ | Spearman rank-correlation coefficient |

Table E.2 defines the abbreviations used across the thesis and supporting artifacts.

**Table E.2**

*Abbreviations used in the thesis.*

| Abbreviation | Full form |
|---|---|
| AIC | Akaike Information Criterion |
| ALE | Accumulated Local Effects |
| APA | American Psychological Association |
| ARIMA | Autoregressive Integrated Moving Average |
| CPCV | Combinatorial Purged Cross-Validation |
| DA | Directional Accuracy |
| DJIA | Dow Jones Industrial Average |
| DM | Diebold–Mariano |
| LIME | Local Interpretable Model-agnostic Explanations |
| LSTM | Long Short-Term Memory |
| MAE | Mean Absolute Error |
| MCS | Model Confidence Set |
| OLS | Ordinary Least Squares |
| PFI | Permutation Feature Importance |
| R² | Coefficient of Determination |
| RMSE | Root Mean Squared Error |
| RQ | Research Question |
| SHAP | SHapley Additive exPlanations |
| TFT | Temporal Fusion Transformer |
| TSF | Time Series Forecasting file format |
| XAI | Explainable Artificial Intelligence |

*Note.* The LIME-style diagnostic in the executed notebooks is a weighted local ridge surrogate on synthetic normal perturbations; its name does not assert execution of a particular external LIME package. TFT refers to the simplified adapter documented in Chapter 4.
