# Multi-Horizon-Hedge-Fund-Forecasting
hedge-ts-xgb
XGBoost-based solution for the Kaggle Hedge Fund Time Series Forecasting competition — a community prediction challenge with 1,171 teams and a $10,000 prize pool.

Problem
Predict a continuous financial value (y_target) for each combination of entity (code, sub_code, sub_category) at four forecast horizons (1, 3, 10, 25 steps ahead). Submissions are scored using a weighted RMSE-inspired metric:

Score
=
1
−
clip
(
∑
w
i
(
y
i
−
y
^
i
)
2
∑
w
i
y
i
2
,
0
,
1
)

Higher is better (max = 1.0). The no-leakage constraint is strict: predictions at time t may only use data from ts_index ≤ t.

Dataset
File	Rows	Columns
train.parquet	~5.3M	94 (86 anonymized features + metadata)
test.parquet	~1.4M	92
Key columns: id, code, sub_code, sub_category, ts_index, horizon, weight, feature_a ... feature_ch.

Methodology
Feature Engineering
All features are computed using only past observations (no look-ahead). Train and test are concatenated before feature computation so lag features from late training rows naturally carry into test rows.

Feature Type	Details
Lags	y_target at offsets [1, 2, 4, 8, 13, 26] per entity group
Rolling mean/std	Windows [5, 10, 20, 40] on shift(1) of target
EWM mean	Exponentially weighted mean at spans [5, 15, 30] — captures recency
Target encoding	Bayesian expanding-mean encoding at 4 group levels, smoothed toward global mean
Cross-sectional rank	Percent rank of entity within each ts_index snapshot
Horizon ratio	Lag-1 value normalized by the cross-horizon average for the same entity and time
Model
Algorithm: XGBoost (XGBRegressor) trained separately per horizon (4 models)
Ensemble: 3 random seeds averaged per horizon to reduce variance
Validation: Temporal split at ts_index = 2800 (no random shuffling)
Early stopping: 80 rounds on weighted RMSE of the validation set
Key hyperparameters:

n_estimators     = 4000
learning_rate    = 0.04
max_depth        = 6
subsample        = 0.75
colsample_bytree = 0.65
min_child_weight = 15
reg_alpha        = 0.2
reg_lambda       = 1.5
No-Leakage Guarantee
All lags use shift(k) with k ≥ 1
Rolling and EWM features apply shift(1) before the window operation
Target encoding uses cumulative sum/count shifted by 1 row
Train/val split is strictly temporal
Results
Horizon	Val Score
1 (short-term)	~0.06
3 (medium-term)	~0.67
10 (long-term)	~0.86
25 (extra long-term)	~0.92
Overall average	~0.85
Note: horizon 1 is inherently harder to predict (lower signal-to-noise); longer horizons benefit more from the EWM and target encoding features.

Project Structure
hedge-ts-xgb/
├── ts_xgboost.ipynb     # Full pipeline: feature engineering + training + submission
├── submission.csv       # Final predictions
└── README.md
Requirements
xgboost >= 2.0
pandas >= 2.0
numpy
scikit-learn
GPU recommended (set device='cuda' in XGBoost params for ~5x speedup on Kaggle T4).

How to Run
Add the competition dataset to /kaggle/input/competitions/ts-forecasting/
Open ts_xgboost.ipynb in a Kaggle notebook (GPU accelerator recommended)
Run all cells — submission CSV is saved to /kaggle/working/submission.csv
