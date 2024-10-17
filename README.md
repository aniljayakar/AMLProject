# **Project Title: Feature Engineering and Imbalanced Data in Money Laundering Detection: A Comparative Study of ML Algorithms**

## **Table of Contents**
1. [Introduction](#introduction)
2. [Dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
3. [Feature Engineering](#feature-engineering)
4. [Modelling Flow](#modelling-flow)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results and Insights](#results-and-insights)
7. [Conclusion](#conclusion)
8. [How to Run the Project](#how-to-run-the-project)
9. [Future Improvements](#future-improvements)

---

## **Introduction**
This project focuses on detecting money laundering activities using machine learning models on the synthetic [IT-AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
 dataset developed by IBM. The study explores the importance of feature engineering and handling class imbalance to improve model performance in detecting illicit transactions(money laundering).

## Step 1: Data Collection, Preprocessing, and Exploratory Data Analysis

## **►Dataset**
The dataset used is the **synthetic [IT-AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
 dataset** developed by researchers at IBM . The dataset was sourced from Kaggle and consists of six subsets categorised by the illicit ratio of transactions: Group HI (higher illicit ratio) and Group LI (lower illicit ratio). 
 
|                            | SMALL              | MEDIUM             | LARGE              |
|----------------------------|--------------------|--------------------|--------------------|
| **Category**                | HI        | LI     | HI        | LI     | HI        | LI     |
| **Date Range (2022)**       | Sep 1-10  | Sep 1-10  | Sep 1-16  | Sep 1-16  | Aug 1 - Nov 5 | Aug 1 - Nov 5 |
| **# of Days Spanned**       | 10        | 10        | 16        | 16        | 97        | 97        |
| **# of Bank Accounts**      | 515K      | 705K      | 2077K     | 2028K     | 2116K     | 2064K     |
| **# of Transactions**       | 5M        | 7M        | 32M       | 31M       | 180M      | 176M      |
| **# of Laundering Transactions** | 5.1K  | 4.0K      | 35K       | 16K       | 223K      | 100K      |
| **Laundering Rate (1 per N Transactions)** | 981 | 1942     | 905       | 1948      | 807       | 1750      |


 
 This project primarily used the small dataset with a high illicit ratio, which contains 5,078,345 financial transactions spanning 10 days.It effectively addresses challenges such as overlap and labelling issues commonly found in real-world money laundering datasets, making it suitable for this research. The dataset was sourced from Kaggle.

Each transaction is represented by 11 attributes:

- **Timestamp**: Exact date and time of the transaction.
- **From Bank & Account / To Bank & Account**: Details of the sending and receiving accounts.
- **Amount Received & Amount Paid**: Monetary values of the transaction.
- **Receiving Currency & Payment Currency**: Types of currencies involved.
- **Payment Format**: The modality or method of payment.
- **Is Laundering**: A binary indicator showing whether the transaction is illicit (money laundering) or legitimate.

### ►Preprocessing and Exploratory Data Analysis (EDA)

#### • Initial Checks:
- Preliminary checks for missing values and NaN values were performed.
- Datatypes of the raw features were adjusted for consistency.
- New time-based features, such as **DayOfWeek** and **Hour**, were derived from the timestamp to enable more detailed analysis.

#### • Skewness and Outliers:
- The **Amount Paid** and **Amount Received** values were highly skewed, as visualised using **Matplotlib** and **Seaborn**.
- Due to the nature of the dataset, **class imbalance** was significant, with a ratio of approximately **1,000 laundering transactions per million legitimate transactions**.

![image](https://github.com/user-attachments/assets/25aaa0e6-db16-4831-847e-757133008ac2)


#### • Class Imbalance:
- The imbalance in laundering transactions was visualised, underscoring the rarity of illicit transactions and their importance for model training.

#### • Outlier Detection:
To address outliers that could significantly impact model performance, three methods were used:
1. **Inter Quartile Range (IQR)**: Flagged many outliers, though laundering transactions remained consistent across features.
2. **Z-Score**: Identified fewer outliers, with fewer laundering transactions flagged.
3. **Isolation Forest**: Identified many non-laundering transactions as outliers by isolating anomalies near the tree root.

While outliers could contain important information, removing them could lead to information loss. Instead, a **log transformation** was applied to compress the scale of **Amount Paid** and **Amount Received**, reducing the impact of extreme values without removing them, which resulted in a more even distribution.

#### • Visualisations:
- The distributions of transactions over time, payment formats, and currencies were visualised to better understand the dataset’s transactional patterns.

---

## Step 2:  **Feature Engineering**
A variety of **graph-based**, **transaction-based**, and **aggregate features** were created to capture the nuances of money laundering activities. These include:

- **Graph-based features**: Degree Centrality, Local Clustering Coefficient, Ego Network Size, etc.
- **Transaction-based features**: Time Since First Transaction, Small Transactions in 24 hours, Rolling Averages.
- **Advanced features**: Structuring Score, IsPotentialSmurfing, IsPotentialStructuring.

For further details, check the [Feature Engineering Details](#feature-engineering-details) section below.

---

## Feature Engineering

### • Graph-Based Features
Transactional data was transformed into a network where accounts acted as nodes, and transactions between them were represented as edges. By adopting this approach, meaningful graph-based features were engineered to help identify patterns of suspicious behavior. The following graph-based features were created:

- **Degree Centrality & Weighted Degree**: These metrics capture the importance and transaction volume of accounts, identifying potential hubs for money laundering.
- **Local Clustering Coefficient**: This feature highlights closely connected transaction groups, which may indicate coordinated laundering activities.
- **Isolation Check & Ego Network Size**: These features provide insights into the connectivity of accounts within the network, revealing isolated accounts or those involved in intricate transaction patterns.

Other complementary features such as Incoming/Outgoing Transactions and Average Neighbour Degree provide a comprehensive view of the transaction network.

### • Transaction-Based Features
Transaction-based features focus on capturing suspicious activity by analysing transactional patterns, behaviours, and characteristics. The features include:

- **Time Since First Transaction & Near 10K Threshold**: These features identify accounts that engage in rapid transactions and those that structure amounts to avoid scrutiny.
- **24-Hour Transaction Patterns**: Features such as `small_transactions_24h` and `unique_banks_24h` capture behaviours within a single day, identifying frequent small transactions or dealings with multiple banks.
- **Rolling Averages and Relative Amounts**: These features help uncover consistent transactional behaviours and anomalies.
- **Rounding Features, Currency Consistency & Payment Method**: These are designed to detect transactions rounded to specific amounts (a possible tactic used by launderers) and flag uncommon or suspicious transaction methods.

### • Aggregate Features
Aggregate features offer a broader perspective on account behavior. To avoid data leakage, these were engineered post-split (after train/test splitting). Key aggregate features include:

- **Amount Ratio & Incoming to Outgoing Ratio**: Capture the flow of money through an account, highlighting irregular behaviours.
- **Time-Related Features**: `Time Difference` and `Average Time Difference` help detect irregular timing of transactions.
- **Statistical Features**: Features like `Amount Received Deviation` measure how much an account's activity deviates from the norm, helping identify suspicious accounts.

### • Advanced Features
To enhance the detection of more advanced money laundering patterns, several advanced features were engineered:

- **NearThreshold**: Flags transactions close to the reporting threshold (e.g., 10,000 USD), a common laundering tactic.
- **IsPotentialSmurfing**: A binary indicator that flags transactions that may belong to a smurfing strategy, where large amounts are broken into smaller ones.
- **Rapid Transactions**: Identifies accounts engaging in frequent transactions within a short time.
- **Structuring Score**: A composite score combining several features, weighted to detect structuring activities—another common laundering method. Additionally, a binary feature called `IsPotentialStructuring` flags accounts likely to engage in structuring.

### • Categorical Features & Encoding
For features such as `Receiving Currency`, `Payment Currency`, and `Payment Format`, target encoding was applied. This method replaces categorical values with the average target (in this case, laundering likelihood), ensuring the model can capture the relationship between categories and money laundering risk. This approach is particularly useful for categorical variables with high cardinality.

### • Bank Clusters
Due to the large number of unique banks in the dataset (nearly 40,000), banks were grouped into clusters using K-Means clustering. The optimal number of clusters was determined using the elbow method. These clusters were then ordinally encoded based on risk levels, providing meaningful features that carried information about potential laundering risks.

### •Temporal Features
Temporal features derived from the `Timestamp` included `DayOfWeek` and `Hour`. Instead of using raw values, sin and cosine transformations were applied to capture the cyclical nature of time. This transformation ensures that the model properly recognises that days and hours follow a cyclical pattern (e.g., Monday is close to Sunday, and 12 AM is close to 11 PM), which aids in improving model performance.

---

## Step 3: Data Splitting

### ► Temporal Split for Train and Test
Given the time series nature of the dataset, it was essential to ensure that the model was trained on past data and validated on future data to simulate real-world scenarios where the model will only encounter future transactions. The data was sorted by timestamp and split into training and testing sets as follows:
- **Training Data**: From 2022-09-01 to 2022-09-08
- **Testing Data**: From 2022-09-09 to 2022-09-18

This temporal split prevents data leakage and ensures that the model evaluation is robust and reflects its performance on future unseen data.

### ► Normal Data Preparation
Once the data was split, further preprocessing was done to separate the features (X) and target variables (y) into:
- **X_train** and **y_train** for the training data
- **X_test** and **y_test** for the testing data

This step prepared the data for modelling.

### ► Addressing Class Imbalance with Upsampling
One of the major challenges in this study was the high class imbalance, a common issue in fraud and money laundering detection. As the dataset contained significantly fewer illicit transactions compared to legitimate ones, it was important to address this imbalance.

- **Upsampling** was employed to increase the representation of the minority class (illicit transactions). The minority class was upsampled by a factor of 10 after separating the classes from the training set.
- This step was undertaken with caution, as excessive upsampling can lead to overfitting. The goal was to compare model performance on both the original and upsampled datasets to determine if upsampling improved precision and recall in detecting the minority class.

### ► Downsampling the Majority Class
In addition to upsampling the minority class, **downsampling** of the majority class (legitimate transactions) was performed to further investigate the impact of class distribution on model performance. This approach helps create a more balanced dataset by reducing the number of legitimate transactions.
- Multiple downsampled sets were created with different ratios: 1:1, 10:1, 20:1, 30:1, 60:1, 80:1, and 100:1.
- These varying ratios allowed for a thorough evaluation of how different levels of class imbalance affected the models' ability to detect both legitimate and illicit transactions.

### ► Key Considerations
- **Information Loss**: Downsampling, while useful in creating a balanced dataset, can lead to a loss of information, reducing the model’s ability to capture unique patterns in legitimate transactions. This trade-off between class balance and information retention was carefully monitored throughout the analysis.
- **Comparison of Results**: Models trained on both upsampled and downsampled datasets were compared to determine which technique provided the best balance between detecting illicit and legitimate transactions.

### 5. Correlation Analysis
During the data preprocessing phase, **correlation analysis** was conducted to identify relationships between the engineered features. After feature engineering, the total number of features stood at 65. Correlation analysis is crucial for several reasons:

- **Multicollinearity**: Highly correlated features can lead to multicollinearity, which negatively affects the performance of certain machine learning algorithms.
- **Feature Reduction**: By removing highly correlated features (above a certain correlation coefficient threshold), the feature space was fine-tuned, enabling:
  - **Faster training** of machine learning models.
  - **Improved model performance** by reducing redundant information.
  - **Enhanced interpretability** of the model by focusing on the most impactful features.

By excluding features with high correlation, we reduced noise and ensured that only the most relevant features contributed to the model's decision-making process, leading to better overall performance and efficiency.

---

## Step 4: Modelling  
The following machine learning models were employed to detect money laundering patterns, with tailored strategies for handling class imbalance and optimising performance:

1. **Decision Trees**: Custom class weights were applied to handle class imbalance. Evaluated on raw, resampled, and downsampled datasets for comparative analysis.
2. **Random Forest**: A standard implementation was initially tested, but RandomizedSearchCV was employed to find optimal hyperparameters. Performance was fine-tuned to enhance the model’s ability to detect minority class transactions.
3. **XGBoost**: Tuned with a binary logistic objective function and logloss evaluation metric. Cross-validation was conducted using Stratified K-fold to improve the model’s handling of minority classes.
4. **LightGBM (LGBM)**: Known for its efficiency with large datasets, LGBM was tested to compare performance across normal, resampled, and downsampled datasets.
5. **Stochastic Gradient Descent (SGD)**: Applied for its ability to handle vast datasets, with logistic regression and support vector machine (SVM) objectives tested on various dataset configurations.
6. **Support Vector Machines (SVM)**: Multiple kernels (linear, RBF, poly, sigmoid) were explored to find the best fit for the data. Hyperparameter tuning was employed to improve classification accuracy.

Each model was rigorously tested on original, resampled, and downsampled data, with hyperparameter optimisation applied to achieve the best results. The performance metrics were focused on detecting the minority class (illicit transactions) to ensure robust money laundering detection.

More details on the modelling process are provided in the [Modelling Flow Details](#modelling-flow-details) section.

---

## Modelling Flow Details  
This section provides an in-depth look at how each model was tuned and applied, ensuring optimal performance for detecting minority classes (illicit transactions):

- **Decision Trees**: Applied custom class weights (`class_weight={0: 1.001, 1: 1001}`) to better predict minority class instances. Evaluated on raw, resampled, and downsampled datasets to see how it handles different data configurations.
- **Random Forest**: Utilised RandomizedSearchCV for hyperparameter tuning with the macro F1 score as the evaluation metric. Focused on optimising parameters like `n_estimators`, `max_depth`, and `min_samples_split` to enhance minority class detection.
- **XGBoost**: Tuned using RandomizedSearchCV with a focus on parameters such as `subsample`, `learning_rate`, `max_depth`, and `colsample_bytree`. Stratified K-fold cross-validation was used to validate the model’s ability to detect minority class transactions effectively.
- **LightGBM (LGBM)**: Tested for its computational speed and efficiency on large datasets. Applied the same modelling configurations across normal, resampled, and downsampled datasets to measure the impact on minority class prediction.
- **SGD (Stochastic Gradient Descent)**: Used to approximate logistic regression and SVM tasks with log loss and hinge loss. Evaluated on both normal and resampled datasets, focusing on its ability to scale and perform under class imbalance.
- **Support Vector Machines (SVM)**: Hyperparameter tuning was applied across multiple kernels (linear, RBF, poly, sigmoid) to improve performance. Models were tested with different regularisation and gamma parameters to find the optimal boundary for classification.

---

## Step 4:  Evaluation 
Due to the significant class imbalance in the dataset, traditional accuracy metrics are misleading. Hence, specific metrics were used:

1. **Minority Class Precision, Recall, and F1 Score**: Focused on the minority class to evaluate how well the models detected laundering transactions.
2. **AUC-ROC Curve**: Measures the ability of the model to distinguish between laundering and legitimate transactions.
3. **Precision-Recall Curve**: Provides more meaningful insights for imbalanced datasets than the ROC curve.
4. **Confusion Matrix**: Used to identify true positives, true negatives, false positives, and false negatives.

Detailed information on why these metrics were used can be found in the [Evaluation Metrics Details](#evaluation-metrics-details) section.

---
## **Evaluation Metrics Details**

Given the class imbalance in the dataset, traditional accuracy metrics would not provide an accurate representation of model performance, as they primarily reflect the model's ability to predict the majority class (legitimate transactions). To better evaluate the model’s performance in identifying the minority class (illicit transactions), the following metrics were used:

### ► Minority Class Precision, Recall, and F1 Score
- **Precision**: Measures the proportion of correctly identified money laundering transactions (true positives) out of all predicted positives.
- **Recall (Sensitivity)**: Measures how well the model identifies actual money laundering transactions (true positives) from all real instances.
- **F1 Score**: The harmonic mean of precision and recall, offering a balanced metric when false positives and false negatives are both important. This is crucial for evaluating the model’s efficiency in detecting the minority class.

### ► AUC-ROC Curve 
- The **Area Under the Receiver Operating Characteristic Curve (AUC-ROC)** reflects the model’s ability to distinguish between the majority class (legitimate transactions) and the minority class (illicit transactions). A higher AUC indicates better performance, with values ranging from 0.5 (no discrimination) to 1.0 (perfect discrimination).

### ► Precision-Recall Curve
Given the class imbalance, the **Precision-Recall Curve** provides a more informative metric than the AUC-ROC. It shows the trade-off between precision and recall at various thresholds. A model with high precision and recall will have a curve that reaches the top right corner of the plot, indicating better performance in identifying the minority class.

### ► Confusion Matrix
The **Confusion Matrix** provides a detailed breakdown of:
- **True Positives (TP)**: Correctly predicted illicit transactions.
- **True Negatives (TN)**: Correctly predicted legitimate transactions.
- **False Positives (FP)**: Legitimate transactions incorrectly flagged as illicit.
- **False Negatives (FN)**: Illicit transactions incorrectly flagged as legitimate.

This matrix offers comprehensive insights into the model’s performance and helps identify models with minimal false positives and false negatives, which is essential for this study.

---

## **Results and Insights**
Here are the key findings from the study:

- **Random Forest** and **XGBoost** performed the best in detecting money laundering patterns.
- SHAP and LIME were used to interpret feature importance, revealing that features like **unique banks in 24 hours**, **log Amount Received**, and **Structuring Score** were crucial.
- The models effectively detected patterns such as **Gather-Scatter**, **Scatter-Gather**, **Fan-Out**, and **Bipartite**.

Below is the comparison of model performance by percentage accuracy for laundering patterns identified by the best performing hyperparameter tuned models:

| **Laundering Type**  | **RF Identified (%)** | **XGBoost Identified (%)** |  
|----------------------|-----------------------|----------------------------|  
| Gather-Scatter        | 27.5%                 | 39.4%                      |  
| Scatter-Gather        | 18.7%                 | 24.8%                      |  
| Stack                 | 16.9%                 | 20.6%                      |  
| Fan-Out               | 22.8%                 | 34.9%                      |  
| Fan-In                | 14.8%                 | 22.4%                      |  
| Cycle                 | 10.2%                 | 15.7%                      |  
| Random                | 11.5%                 | 24.6%                      |  
| Bipartite             | 7.9%                  | 20.2%                      |

For more detailed results, check the [Results Details](#results-details) section.

---

## Results Details

### ► Algorithm Performance:

Random Forest and XGBoost outperformed the other algorithms, particularly in handling the highly imbalanced data. These ensemble methods were able to capture more minority class instances (fraudulent transactions) with higher recall and precision compared to linear models like Logistic Regression and Stochastic Gradient Descent (SGD).
Support Vector Machines (SVM) with varied kernels showed strong performance in certain aspects, but due to the large dataset size and complexity, its training time was significantly longer, which made it less efficient compared to ensemble methods.
LightGBM (LGBM) performed similarly to XGBoost, but XGBoost edged it out slightly in terms of precision and recall, making it the most effective for this dataset.

### ► Evaluation Metrics on Normal Data

| Model                            | Precision | Recall | F1 Score | TP  | FP    | FN   | ROC-AUC | PR-AUC |
|-----------------------------------|-----------|--------|----------|-----|-------|------|---------|--------|
| Decision Trees (balanced weight)  | 0.26      | 0.21   | 0.23     | 334 | 940   | 1269 | 0.60    | 0.06   |
| Decision Trees with custom weight | 0.27      | 0.20   | 0.23     | 326 | 896   | 1277 | 0.60    | 0.06   |
| Random Forests                    | 0.89      | 0.33   | 0.48     | 533 | 67    | 1070 | 0.97    | 0.52   |
| XGBoost                           | 0.53      | 0.50   | 0.51     | 806 | 727   | 797  | 0.98    | 0.52   |
| LGBM                              | 0.02      | 0.53   | 0.03     | 849 | 52495 | 754  | 0.80    | 0.01   |
| SGD (Log loss)                    | 0.00      | 0.03   | 0.01     | 49  | 11044 | 1154 | 0.91    | 0.01   |
| SGD (Hinge loss)                  | 0.00      | 0.03   | 0.01     | 53  | 11686 | 1150 | N/A     | N/A    |

---
### ► Evaluation Metrics on Resampled Data where the minority class was upsampled

| Model                           | Precision | Recall | F1 Score | TP  | FP   | FN   | ROC-AUC | PR-AUC |
|----------------------------------|-----------|--------|----------|-----|------|------|---------|--------|
| Decision Trees (balanced weight) | 0.19      | 0.21   | 0.20     | 342 | 1456 | 1261 | 0.61    | 0.04   |
| Decision Trees with custom weight| 0.45      | 0.11   | 0.18     | 178 | 221  | 1425 | 0.56    | 0.05   |
| Random Forests                   | 0.84      | 0.33   | 0.48     | 530 | 98   | 1073 | 0.97    | 0.51   |
| XGBoost                          | 0.38      | 0.56   | 0.45     | 903 | 1485 | 700  | 0.98    | 0.47   |
| LGBM                             | 0.10      | 0.61   | 0.18     | 980 | 8534 | 623  | 0.98    | 0.12   |
| SGD (Log loss)                   | 0.01      | 0.05   | 0.01     | 83  | 9452 | 1520 | 0.95    | 0.03   |
| SGD (Hinge loss)                 | 0.00      | 0.00   | 0.00     | 0   | 14   | 1603 | N/A     | N/A    |


### ► Evaluation Metrics on Downsampled Data, where the majority class was downsampled

- **Model: Decision Trees (balanced weight)**

| Downsampling Ratio | Precision | Recall | F1 Score | TP  | FP    | FN   | ROC-AUC | PR-AUC |
|--------------------|-----------|--------|----------|-----|-------|------|---------|--------|
| 1:1                | 0.01      | 0.80   | 0.03     | 1285| 87320 | 318  | 0.85    | 0.01   |
| 10:1               | 0.03      | 0.57   | 0.05     | 918 | 32027 | 685  | 0.77    | 0.02   |
| 20:1               | 0.05      | 0.51   | 0.09     | 818 | 16663 | 785  | 0.75    | 0.02   |
| 30:1               | 0.05      | 0.43   | 0.09     | 689 | 12576 | 914  | 0.71    | 0.02   |
| 60:1               | 0.09      | 0.43   | 0.14     | 696 | 7413  | 907  | 0.71    | 0.04   |
| 80:1               | 0.08      | 0.33   | 0.13     | 534 | 6038  | 1069 | 0.66    | 0.03   |
| 100:1              | 0.10      | 0.33   | 0.15     | 535 | 4996  | 1068 | 0.66    | 0.03   |

- **Model: Random Forests**

| Downsampling Ratio | Precision | Recall | F1 Score | TP  | FP    | FN   | ROC-AUC | PR-AUC |
|--------------------|-----------|--------|----------|-----|-------|------|---------|--------|
| 1:1                | 0.02      | 0.95   | 0.03     | 1521| 91037 | 82   | 0.98    | 0.22   |
| 10:1               | 0.07      | 0.80   | 0.12     | 1279| 18099 | 324  | 0.98    | 0.46   |
| 20:1               | 0.10      | 0.73   | 0.17     | 1172| 10898 | 431  | 0.98    | 0.48   |
| 30:1               | 0.17      | 0.69   | 0.27     | 1114| 5608  | 489  | 0.98    | 0.51   |
| 60:1               | 0.27      | 0.61   | 0.37     | 985 | 2705  | 618  | 0.97    | 0.51   |
| 80:1               | 0.35      | 0.59   | 0.44     | 938 | 1725  | 665  | 0.98    | 0.51   |
| 100:1              | 0.39      | 0.55   | 0.44     | 902 | 1600  | 615  | 0.98    | 0.52   |

- **Model: XGBoost**

| Downsampling Ratio | Precision | Recall | F1 Score | TP  | FP     | FN   | ROC-AUC | PR-AUC |
|--------------------|-----------|--------|----------|-----|--------|------|---------|--------|
| 1:1                | 0.02      | 0.97   | 0.03     | 1549| 101257 | 54   | 0.98    | 0.28   |
| 10:1               | 0.05      | 0.84   | 0.09     | 1353| 26702  | 250  | 0.98    | 0.42   |
| 20:1               | 0.10      | 0.76   | 0.17     | 1226| 11374  | 377  | 0.98    | 0.45   |
| 30:1               | 0.12      | 0.73   | 0.21     | 1167| 8338   | 436  | 0.98    | 0.45   |
| 60:1               | 0.16      | 0.68   | 0.26     | 1084| 5608   | 519  | 0.98    | 0.46   |
| 80:1               | 0.23      | 0.63   | 0.33     | 1002| 3451   | 601  | 0.98    | 0.45   |
| 100:1              | 0.21      | 0.64   | 0.32     | 1032| 3862   | 571  | 0.98    | 0.47   |

- **Model: LGBM**

| Downsampling Ratio | Precision | Recall | F1 Score | TP  | FP     | FN   | ROC-AUC | PR-AUC |
|--------------------|-----------|--------|----------|-----|--------|------|---------|--------|
| 1:1                | 0.00      | 0.93   | 0.00     | 1494| 740782 | 109  | 0.80    | 0.02   |
| 10:1               | 0.03      | 0.63   | 0.05     | 1003| 34763  | 600  | 0.95    | 0.04   |
| 20:1               | 0.04      | 0.30   | 0.07     | 476 | 12067  | 1127 | 0.92    | 0.04   |
| 30:1               | 0.07      | 0.35   | 0.11     | 554 | 7530   | 1049 | 0.94    | 0.07   |
| 60:1               | 0.01      | 0.49   | 0.02     | 787 | 87232  | 816  | 0.87    | 0.01   |
| 80:1               | 0.06      | 0.48   | 0.11     | 768 | 11156  | 835  | 0.95    | 0.09   |
| 100:1              | 0.11      | 0.63   | 0.19     | 1006| 8163   | 597  | 0.98    | 0.34   |

- **Model: SGD (Hinge loss)**

| Downsampling Ratio | Precision | Recall | F1 Score | TP  | FP    | FN   |
|--------------------|-----------|--------|----------|-----|-------|------|
| 1:1                | 0.01      | 0.99   | 0.01     | 1586| 216012| 17   |
| 10:1               | 0.01      | 0.56   | 0.02     | 890 | 69799 | 713  |
| 20:1               | 0.00      | 0.05   | 0.00     | 80  | 48466 | 1523 |
| 30:1               | 0.00      | 0.08   | 0.00     | 122 | 55100 | 1481 |
| 60:1               | 0.10      | 0.01   | 0.02     | 18  | 163   | 1585 |
| 80:1               | 0.01      | 0.03   | 0.02     | 56  | 3896  | 1547 |
| 100:1              | 0.02      | 0.00   | 0.00     | 1   | 53    | 1602 |

- **Model: SVM**

| Kernel     | Precision | Recall | F1 Score | TP  | FP   | FN   |
|------------|-----------|--------|----------|-----|------|------|
| Linear     | 0.01      | 0.53   | 0.02     | 847 | 91950| 756  |
| Polynomial | 0.05      | 0.18   | 0.08     | 282 | 5146 | 1321 |
| RBF        | 0.08      | 0.22   | 0.11     | 359 | 4345 | 1244 |
| Sigmoid    | 0.01      | 0.37   | 0.02     | 598 | 55458| 1005 |


### ► Metrics of Hyperparameter Tuned Models

- **Model: Tuned Random Forest**

| Dataset               | Precision | Recall | F1 Score | TP  | FP  | FN  | ROC-AUC | PR-AUC |
|-----------------------|-----------|--------|----------|-----|-----|-----|---------|--------|
| On 100:1 Resampled Data| 0.33      | 0.57   | 0.42     | 920 | 1868| 683 | 0.97    | 0.49   |
| On normal data         | 0.85      | 0.37   | 0.52     | 594 | 106 | 1009| 0.95    | 0.52   |

- **Model: Tuned XGBoost**

| Dataset       | Precision | Recall | F1 Score | TP  | FP  | FN  | ROC-AUC | PR-AUC |
|---------------|-----------|--------|----------|-----|-----|-----|---------|--------|
| On normal data| 0.66      | 0.51   | 0.57     | 810 | 410 | 793 | 0.99    | 0.60   |


## Feature Importance:
Feature importance analysis revealed that network-based and graph-based features played a crucial role in improving model performance. Features related to transaction connectivity, like the degree of connections between entities and transaction volumes, were found to be the most predictive of money laundering patterns.
Financial patterns detected through K-means clustering also contributed significantly to the predictive power of the models, especially in identifying high-risk clusters of banks (e.g., those with high transaction volumes and numerous connections).

## Handling Imbalanced Data:
The dataset was highly imbalanced, with fraudulent transactions representing only a small fraction of the overall data.
Models that did not account for the class imbalance (such as SGD and Logistic Regression) struggled with detecting minority class instances, often skewing predictions toward the majority class (non-fraudulent transactions).
Ensemble models like Random Forest and XGBoost, along with SMOTE, showed the best results in terms of balancing Precision and Recall, making them suitable for identifying fraudulent activity without generating an excessive number of false positives.

## Interpretability and Explainability:
LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations) were used to interpret individual predictions, making it easier to explain model decisions to non-technical stakeholders. This interpretability was particularly important in the context of fraud detection, where the rationale behind identifying certain transactions as fraudulent needs to be transparent.
The SHAP values highlighted key variables that contributed to predictions, with certain transaction patterns and connectivity metrics being the most significant in detecting money laundering activities.

## Money Laundering Patterns:
The project successfully identified the eight distinct money laundering patterns which were present in the dataset, such as Gather-Scatter and Scatter-Gather, using network-based features. These patterns were associated with specific transaction flows between banks, where large sums of money were often moved through a series of transactions that appeared benign individually but formed a suspicious pattern when viewed collectively.
K-means clustering further enhanced the model’s ability to identify risky behavior, with certain clusters of banks displaying characteristics that aligned with known money laundering techniques.

## Overall Impact:
The project demonstrated that ensemble methods like Random Forest and XGBoost, when combined with feature engineering techniques and appropriate handling of imbalanced data, were the most effective in detecting money laundering patterns.
Feature importance, combined with model interpretability tools like LIME and SHAP, allowed for greater transparency in predictions, making the models not only effective but also explainable to financial institutions, which is a critical requirement in high-stakes applications like anti-money laundering.

---

## **Conclusion**
This study demonstrates that feature engineering and addressing class imbalance are critical in detecting money laundering patterns. The models developed in this study, particularly Random Forest and XGBoost, proved effective in identifying illicit activities, even in an imbalanced dataset.

---


