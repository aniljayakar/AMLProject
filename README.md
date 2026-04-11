# AML Transaction Monitoring

**5,078,345 synthetic transactions. 1 in 981 is illicit. The objective is to catch as many as possible without burying investigators in false positives.**

`5.1M transactions` &nbsp;|&nbsp; `0.102% laundering rate` &nbsp;|&nbsp; `65+ engineered features` &nbsp;|&nbsp; `Best F1: 0.57 (XGBoost, 100:1)`

---

## The Problem

Manual review of millions of daily transactions is not operationally viable. Rule-based systems produce too many alerts and miss pattern-based laundering. The question this project answers is narrow: for 5.1M rows of imbalanced synthetic transactions with temporal constraints, which model at which class-balance ratio produces the most useful alert queue for a fixed-headcount compliance team?

The dataset is IBM's IT-AML synthetic set, HI-Small subset. Ten days of transactions. Laundering rate of 0.102%. Eight distinct typologies including Fan-Out, Gather-Scatter, Cycle, and Bipartite patterns.

![aml_pipeline_architecture](https://github.com/user-attachments/assets/aa9d101a-3b06-460c-9910-e7eda9963776)

---

## Business Context

Money laundering costs the global economy an estimated $800 billion to $2 trillion annually. Financial institutions are legally required to detect and report suspicious activity, but manual review of millions of transactions is impossible at scale.

The core challenge in building an AML detection system is not accuracy. It is the **precision-recall tradeoff**:
- Too many false positives: operational teams are overwhelmed, legitimate customers are disrupted
- Too many false negatives: illicit transactions go undetected, creating regulatory and reputational risk

This project directly addresses that tradeoff through systematic comparison of models, resampling strategies, and interpretability tools.

---

## Results

This project covers the full detection pipeline: cloud ingestion, multi-method outlier diagnostics, typology-aware feature engineering across 65+ features, a systematic resampling sweep, and a LIME/SHAP explainability layer for SAR support.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| XGBoost (tuned, 100:1) | 0.66 | 0.51 | **0.57** | **0.99** |
| Random Forest (tuned, 100:1) | 0.85 | 0.37 | 0.52 | 0.95 |
| XGBoost (normal data) | 0.53 | 0.50 | 0.51 | 0.98 |
| Random Forest (normal data) | 0.89 | 0.33 | 0.48 | 0.97 |
| LightGBM | 0.02 | 0.53 | 0.03 | 0.80 |
| Decision Trees | 0.27 | 0.20 | 0.23 | 0.60 |
| SGD / SVM variants | <0.01 | <0.10 | <0.02 | N/A |

A model predicting every transaction as legitimate is 99.9% accurate. Accuracy is not the metric. F1 keeps both precision and recall visible simultaneously. Linear models fail unconditionally: the non-linear relationships in transaction network data cannot be captured by a hyperplane. LightGBM achieves high recall but near-zero precision, making it operationally useless at the default threshold.

---

## Pipeline Architecture

```
Raw Data (Azure ML Datastore)
        |
        v
Profiling + Datatype Corrections
[IQR + Z-score + Isolation Forest on amounts. No row removal. log1p transforms applied.]
        |
        v
Temporal Split  [Train: Sep 1-8  |  Test: Sep 9-18]
[train_stats propagated to test set. Global stat fallback for unseen accounts.]
        |
        v
Feature Engineering  [65+ features across 3 typology-aware categories]
   Velocity windows (24H/7D/14D rolling)
   Network centrality (DiGraph, 6 node metrics)
   Structuring indicators (NearThreshold, StructuringScore, IsPotentialSmurfing)
        |
        v
Resampling Sweep  [1:1 to 100:1 downsampling + 10x upsampling]
[sklearn.utils.resample. No SMOTE. Each ratio evaluated independently across all model families.]
        |
        v
Model Comparison  [7 families: DT, RF, XGBoost, LGBM, SGD, LR, SVM]
[RandomizedSearchCV + StratifiedKFold for XGBoost. Best: XGBoost 100:1.]
        |
        v
Explainability  [SHAP (global) + LIME (per-transaction)]
[Investigator audit trail for SAR filing.]
```

---

## Data Pipeline

### Ingestion

Data is loaded directly from an Azure ML `Datastore` using SDK v1:

```python
from azureml.core import Workspace, Dataset, Datastore

workspace = Workspace(subscription_id, resource_group, workspace_name)
datastore = Datastore.get(workspace, "finalproject")
dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'HI-Small_Trans.csv'))
df = dataset.to_pandas_dataframe()
```

This is a cloud-connected experimentation setup. All compute runs locally in the notebook kernel. There is no `Experiment`, `Run`, or `ScriptRunConfig` tracking. The `Workspace` and `Datastore` objects provide reproducible, version-controlled data access.

### Profiling and Integrity

**Datatype corrections** run before any computation. `Timestamp` is cast to `datetime64`. Bank identifiers and currencies go to `category`. Account IDs go to `string`. Amounts go to `float64`. On 5.1M rows, silent dtype coercions inside rolling window operations produce incorrect results. The corrections prevent that.

**Outlier diagnostics** are applied to `Amount Received` and `Amount Paid` using three methods in sequence:

- IQR method: flags values outside `Q1 - 1.5*IQR` / `Q3 + 1.5*IQR`
- Z-score: flags values with `|z| > 3`
- Isolation Forest: `contamination=0.05`, multivariate detection on both columns simultaneously

No rows are removed. Extreme transaction amounts carry the signal in AML data. `np.log1p` transforms are applied to both columns to compress the distribution without discarding data.

**Post-engineering nulls** are handled in `handle_null_values`: `TimeDifference` filled with column mean, `AvgTimeDifference` chained from `TimeDifference`, `RelativeFrequency` filled with 1. Filling `RelativeFrequency` with 1 is a deliberate assumption: the account is transacting at its own average pace. `StructuringScore` is recalculated after imputation to stay consistent.

---

## Temporal Split and Leakage Control

```
Training:  2022-09-01 to 2022-09-08
Test:      2022-09-09 to 2022-09-18
```

A random split is wrong here. Future transactions should not inform how past cases are scored. Two leakage controls are written directly into the feature functions.

**`train_stats` propagation.** `create_derived_features` stores `TotalAmountReceived` and `AvgTimeDifference` from the training set in a dict. The test set call receives that dict:

```python
train_df, train_stats = create_derived_features(train_df)
test_df, _ = create_derived_features(test_df, train_stats)
```

`RelativeAmount` on the test set is normalised by the training set's total volume. `AvgTimeDifference` on the test set uses training-period pacing, not the test period's own.

**Global statistics fallback.** `add_amount_received_features` accepts `global_mean`, `global_median`, and `global_std` from training. Accounts that appear in the test set but not the training set fall back to these values rather than computing their own statistics from test-period data.

**Known limitation.** `compute_pattern_based_features` is called on the full pre-split dataframe. The 7-day and 14-day rolling averages and the 24-hour velocity windows are computed before the Sep 8/9 boundary is enforced. Transactions at the cut point create marginal temporal leakage. On a 10-day dataset this is a small overlap, but it is leakage and would be the first thing corrected in a production rewrite.

---

## Feature Engineering

### Velocity and Burst Features

Time-aware rolling windows using `groupby().rolling(window='24H')` on a timestamp-indexed frame. Row-count rolling is not used anywhere in the codebase.

| Feature | Logic | Typology Target |
|---|---|---|
| `small_transactions_24h` | Count of transactions below $3,000 in a 24H window per account | Smurfing |
| `unique_banks_24h` | Distinct destination banks in a 24H window per account | Layering |
| `AvgAmountReceived_7days` | 7-day rolling mean per account | Baseline deviation |
| `AvgAmountReceived_14days` | 14-day rolling mean per account | Baseline deviation |
| `RapidTransactions` | Binary: inter-transaction gap < 5 minutes | Automated layering |
| `RelativeFrequency` | `TimeDifference / AvgTimeDifference` per account | Pace deviation from own history |
| `TimeSinceFirstTransaction` | Seconds since account's first transaction | Account age proxy |

### Network Features

A directed `networkx.DiGraph` is built from account-to-account edges. All node metrics are computed on the full pre-split graph and mapped back to transactions by account.

| Feature | Implementation | What It Captures |
|---|---|---|
| `DegreeCentrality` | `nx.degree_centrality(G)` | Accounts at the centre of transaction networks |
| `WeightedDegree` | `G.degree(weight='Amount Received')` | Degree scaled by transaction volume |
| `LocalClusteringCoefficient` | `nx.clustering(G)` | Tight sub-networks, circular flows |
| `AvgNeighborDegree` | `nx.average_neighbor_degree(G)` | Proximity to structurally central nodes |
| `EgoNetworkSize` | `len(nx.ego_graph(G, node))` per node | Size of 1-hop neighbourhood |
| `IsIsolate` | `nx.is_isolate(G, node)` per node | Anomalous singleton accounts |

Graph features are the single strongest predictor class in this model. An account that looks unremarkable in isolation often sits at the intersection of dozens of high-volume flows.

`EgoNetworkSize` is computed in a Python loop calling `nx.ego_graph` per node. This is the computational bottleneck. A direct in-degree plus out-degree count from the edge list would be faster and produce a correlated result without traversing the graph per node.

### Structuring and Threshold Features

| Feature | Logic | Typology Target |
|---|---|---|
| `NearThreshold` | Binary: $9,000 to $10,000 received | CTR avoidance |
| `Nearly_10K_Transactions` | Per-account count of near-threshold events | Repeated structuring |
| `IsPotentialSmurfing` | Binary: Amount Received < $1,000 | Sub-threshold micro-payments |
| `StructuringScore` | Equal-weight sum: `RelativeAmount + Amount Ratio + TimeDifference + NearThreshold + incoming_to_outgoing_ratio + amount_discrepancy` | Composite risk score |
| `IsPotentialStructuring` | Binary: `StructuringScore > 95th percentile` | High composite risk |

`StructuringScore` has a design flaw worth stating directly: all six component weights are 1 and `TimeDifference` is in raw seconds. It dominates the composite numerically. The score is directionally useful but not calibrated. Weighting the components against typology labels is the obvious next step.

### Bank Clustering

Roughly 40,000 unique banks are profiled on log-transformed transaction totals, network degree, clustering coefficient, and directional flow (sends-only, receives-only, or both). Features are standardised and clustered with K-Means at k=5, selected via elbow method. Cluster labels are ordinally encoded by risk level and joined to transactions as `From Bank Cluster` and `To Bank Cluster`. This converts a high-cardinality bank identifier into a model-usable ordinal risk proxy.

### Encoding

`Receiving Currency`, `Payment Currency`, and `Payment Format` are target-encoded using `category_encoders.TargetEncoder`. The encoder is fit on the training set and applied to both sets, so no label leakage occurs through the encoding step.

Cyclical time features are applied to hour-of-day and day-of-week using sin/cosine transforms. This preserves circular distance: 23:00 and 01:00 are treated as close, not maximally distant.

---

## Class Imbalance and Model Selection

### Resampling Strategy

No SMOTE. All resampling uses `sklearn.utils.resample`.

One upsampling configuration: minority class multiplied 10x with replacement. Seven downsampling configurations, majority class reduced without replacement:

| Ratio | Majority Class Samples |
|---|---|
| 1:1 | Equal to minority count |
| 10:1 | 10x minority count |
| 20:1 | 20x minority count |
| 30:1 | 30x minority count |
| 60:1 | 60x minority count |
| 80:1 | 80x minority count |
| 100:1 | 100x minority count |

Each ratio is evaluated independently across all model families. The purpose is to make the precision-recall trade-off explicit and configurable. In a compliance setting, the operating point is a resourcing decision, not a modelling decision. A team with capacity to review 500 alerts per day needs a model tuned to that budget.

### Hyperparameter Search

**Random Forest:** `RandomizedSearchCV`, `cv=3`, `n_iter=10`, `scoring=f1_macro`. The call does not explicitly pass `StratifiedKFold`, so fold-level class distributions are not guaranteed at a 0.1% positive rate. This is a weaker validation design than the XGBoost equivalent and is a gap in the experiment.

Parameters searched: `n_estimators` [50, 100, 150], `max_features` ['sqrt'], `max_depth` [10, 20, 30], `min_samples_split` [2, 5, 10], `min_samples_leaf` [1, 2, 4], `bootstrap` [True, False].

**XGBoost:** `RandomizedSearchCV` with `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` passed explicitly. `n_iter=10`, `scoring=f1_macro`. The explicit stratification keeps class proportions stable across folds at extreme imbalance ratios.

Parameters searched:
- `n_estimators`: [50, 100, 200]
- `learning_rate`: [0.01, 0.05, 0.1]
- `max_depth`: [3, 4, 5, 6]
- `subsample`: [0.8, 0.9, 1.0]
- `colsample_bytree`: [0.8, 0.9, 1.0]
- `gamma`: [0, 0.1]
- `alpha` (L1 regularisation): [0, 0.1, 0.5]
- `lambda` (L2 regularisation): [1, 1.5, 2]

The L1 and L2 regularisation terms in the search space directly control overfitting on the minority class in ways that depth and subsampling alone do not.

---

## Explainability and Investigator Audit Trail

A model score is not sufficient for SAR filing. A compliance officer needs to know why a transaction was flagged before making a defensible escalation decision.

**SHAP** (`shap.TreeExplainer`) is applied to the tuned Random Forest. SHAP values are additive and sum to the model output: attribution is mathematically exact, not an approximation. The summary plot ranks features by mean absolute SHAP value, giving a population-level view of which signals drive the model. A force plot for a single instance shows exactly which features pushed a specific score above threshold and by how much.

**LIME** (`LimeTabularExplainer`) produces per-transaction explanations:

```python
explainer = LimeTabularExplainer(
    X_train_np, feature_names=X_train.columns.tolist(),
    class_names=['Negative', 'Positive'], mode='classification'
)
exp = explainer.explain_instance(X_train_np[i], rf_best.predict_proba)
```

LIME perturbs the input around the target instance and fits a local linear approximation. The output is a ranked list of feature contributions specific to that transaction. For SAR filing, this is the operationally useful output: an investigator attaches it to the case record and documents exactly which signals drove the escalation decision.

The feature names map directly to typologies an investigator already recognises. `NearThreshold` is structuring. `unique_banks_24h` is layering. `IsPotentialSmurfing` is self-describing. That naming was deliberate.

---

## Known Limitations

- `compute_pattern_based_features` is computed pre-split. Rolling windows at the Sep 8/9 boundary produce marginal leakage.
- `StructuringScore` component weights are equal and untuned. `TimeDifference` in raw seconds dominates.
- `EgoNetworkSize` is computed in a node-by-node loop. Not scalable beyond this dataset size without rewriting.
- `cv=3` without explicit stratification in the Random Forest search is a weaker validation design than the XGBoost equivalent.
- All results are on synthetic data. Generalisation to real transaction flows is not established.

---

## Future Work

- Re-implement `compute_pattern_based_features` as a post-split function using train-derived baselines
- Calibrate `StructuringScore` weights against typology labels using logistic regression coefficients
- Replace the `EgoNetworkSize` loop with a vectorised in-degree/out-degree calculation
- Add `StratifiedKFold` explicitly to the Random Forest hyperparameter search
- Evaluate on Medium and Large IT-AML subsets for scale testing
- Threshold optimisation against a defined alert budget (precision-at-k) rather than global F1
- Package the feature pipeline as a reusable module with unit tests

---

## Setup

```bash
git clone https://github.com/aniljayakar/AMLProject.git
cd AMLProject
pip install -r requirements.txt
jupyter notebook
```

Dataset: IBM IT-AML, HI-Small subset. Download from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml). If running locally, replace the Azure ML data-loading cells with a `pd.read_csv` call pointing to the data directory.

---

*MSc Data Science, Middlesex University London, 2023. Distinction.*
