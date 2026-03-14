# Money Laundering Detection Using Machine Learning

**Can machine learning reliably detect financial crime in a dataset of 5 million transactions — where fewer than 1 in 1,000 is illicit?**

This project answers that question. Using IBM's synthetic AML dataset, I engineered 65+ features, compared 7 machine learning models across multiple resampling strategies, and identified the conditions under which fraud detection models are actually deployable in a real financial institution.

**Best result:** Tuned XGBoost achieved an F1 score of 0.57 and ROC-AUC of 0.99 on normal data — outperforming all other models while maintaining interpretability through SHAP values.

**Business recommendation:** XGBoost with graph-based and temporal features, deployed at a 100:1 downsampling ratio, offers the best balance between catching illicit transactions and minimising false positives — critical in production environments where flagging legitimate transactions has real customer and reputational cost.

---

## Key Results

| Model | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|
| Tuned XGBoost | 0.66 | 0.51 | **0.57** | **0.99** |
| Tuned Random Forest | 0.85 | 0.37 | 0.52 | 0.95 |
| XGBoost (normal data) | 0.53 | 0.50 | 0.51 | 0.98 |
| Random Forest (normal data) | 0.89 | 0.33 | 0.48 | 0.97 |
| LGBM | 0.02 | 0.53 | 0.03 | 0.80 |
| Decision Trees | 0.27 | 0.20 | 0.23 | 0.60 |
| SGD / SVM | <0.01 | <0.10 | <0.02 | N/A |

**Why F1 score matters here:** In highly imbalanced datasets, accuracy is misleading. A model that predicts every transaction as legitimate would be 99.9% accurate but useless. F1 score balances precision (are our fraud flags correct?) and recall (are we catching enough fraud?).

---

## What Made the Difference

Three things separated the high-performing models from the rest:

**1. Feature engineering over raw features**
Graph-based features — degree centrality, clustering coefficients, ego network size — were the strongest predictors. Money laundering is a network behaviour, not a single-transaction behaviour. Modelling the relationships between accounts, not just the transactions themselves, unlocked detection capability that raw features couldn't provide.

**2. Downsampling ratio matters more than upsampling**
Upsampling the minority class (SMOTE) improved recall but hurt precision, generating too many false positives. A 100:1 downsampling ratio on Random Forest and XGBoost produced the best precision-recall balance — a finding with direct implications for production deployment where false positive rates drive operational cost.

**3. Ensemble methods are non-negotiable**
Linear models (SGD, Logistic Regression) and SVMs failed on this problem. The non-linear relationships in financial transaction networks require ensemble methods. XGBoost consistently outperformed all alternatives.

---

## Business Context

Money laundering costs the global economy an estimated $800 billion to $2 trillion annually. Financial institutions are legally required to detect and report suspicious activity — but manual review of millions of transactions is impossible at scale.

The core challenge in building an AML detection system is not accuracy. It is the **precision-recall tradeoff**:
- Too many false positives: operational teams are overwhelmed, legitimate customers are disrupted
- Too many false negatives: illicit transactions go undetected, creating regulatory and reputational risk

This project directly addresses that tradeoff through systematic comparison of models, resampling strategies, and interpretability tools.

---

## Laundering Patterns Detected

The dataset contains 8 distinct money laundering typologies. XGBoost outperformed Random Forest on all of them:

| Laundering Pattern | Random Forest | XGBoost |
|---|---|---|
| Gather-Scatter | 27.5% | 39.4% |
| Scatter-Gather | 18.7% | 24.8% |
| Fan-Out | 22.8% | 34.9% |
| Fan-In | 14.8% | 22.4% |
| Stack | 16.9% | 20.6% |
| Random | 11.5% | 24.6% |
| Bipartite | 7.9% | 20.2% |
| Cycle | 10.2% | 15.7% |

---

## Technical Overview

### Dataset
IBM synthetic AML dataset — Small HI subset: 5,078,345 transactions over 10 days, with a laundering rate of approximately 1 in 981 transactions. Sourced from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).

### Feature Engineering (65 features engineered)
- **Graph-based:** Degree centrality, weighted degree, local clustering coefficient, ego network size, isolation check, average neighbour degree
- **Transaction-based:** Time since first transaction, small transactions in 24 hours, unique banks in 24 hours, rolling averages, near-threshold flags
- **Advanced:** Structuring score, IsPotentialSmurfing, IsPotentialStructuring, NearThreshold
- **Temporal:** Sin/cosine transformed hour and day-of-week features to capture cyclical patterns
- **Bank clustering:** K-Means clustering on ~40,000 unique banks, ordinally encoded by risk level

### Handling Class Imbalance
- Upsampling (minority class x10)
- Downsampling majority class at ratios: 1:1, 10:1, 20:1, 30:1, 60:1, 80:1, 100:1
- Compared impact on precision, recall, and F1 across all models

### Models Compared
Decision Trees, Random Forest, XGBoost, LightGBM, SGD (log loss + hinge loss), SVM (linear, RBF, polynomial, sigmoid kernels)

### Temporal Train/Test Split
- Training: 2022-09-01 to 2022-09-08
- Testing: 2022-09-09 to 2022-09-18
- Temporal split used (not random) to prevent data leakage and simulate real-world deployment

### Interpretability
SHAP and LIME used to explain individual predictions. Top features by importance: unique banks in 24 hours, log Amount Received, Structuring Score. This interpretability is critical in regulated environments where model decisions must be explainable to compliance teams and regulators.

---

## How to Run

```bash
git clone https://github.com/aniljayakar/AMLProject
cd AMLProject
pip install -r requirements.txt
jupyter notebook
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) and place it in the `/data` folder before running.

---

## Future Improvements
- Real-time scoring pipeline using the tuned XGBoost model
- Graph Neural Networks (GNNs) for deeper network-based detection
- Threshold optimisation based on operational cost of false positives vs false negatives
- Testing on Medium and Large dataset subsets for scalability validation

---

*MSc Data Science Dissertation — Middlesex University London, 2023. Distinction.*
