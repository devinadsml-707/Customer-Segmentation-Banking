# 💳 Credit Card Customer Segmentation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)
![Model](https://img.shields.io/badge/Model-KMeans%20Clustering-blueviolet?style=for-the-badge)
![Clusters](https://img.shields.io/badge/Clusters-5-success?style=for-the-badge)

**An unsupervised machine learning pipeline that segments Berlian Bank's credit card customers into 5 behavioral clusters using PCA + K-Means, enabling the marketing team to deliver targeted promotions and reward strategies for each customer segment.**

[Problem Background](#-problem-background) · [Methodology](#-methodology) · [Cluster Results](#-cluster-results) · [Setup](#-local-setup) · [Inference](#-model-inference) · [Stack](#-tech-stack)

</div>

---

## 📌 Problem Background

The marketing team at **Berlian Bank** needs to segment their credit card users based on six months of transaction data. Without a structured segmentation framework, the team cannot effectively:

- Identify high-value customers for premium offerings and card upgrades
- Design promotions tailored to different spending behaviors
- Allocate marketing resources efficiently across customer groups

This project addresses these challenges by building an **end-to-end clustering pipeline** (PCA + K-Means) that groups customers into 5 distinct behavioral segments, each with actionable marketing recommendations.

---

## 🎯 Objectives

| # | Objective |
|---|-----------|
| 1 | Cluster existing credit card users based on 6 months of transaction behavior |
| 2 | Provide actionable marketing recommendations for each customer segment |
| 3 | Analyze whether customer tenure influences purchase habits, balance, and payment behavior |
| 4 | Explore whether a higher credit limit affects purchase frequency |
| 5 | Predict which cluster a new customer belongs to via model inference |

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| **Source** | Google BigQuery |
| **Filter** | `WHERE MOD(CUST_ID, 2) = 0` (even-numbered customer IDs) |
| **Rows** | 4,475 |
| **Columns** | 18 |
| **Missing values** | `CREDIT_LIMIT` (1), `MINIMUM_PAYMENTS` (158) — imputed in feature engineering |
| **Feature types** | All numerical (INT64 / FLOAT64) |

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `CUST_ID` | INT64 | Credit card holder identification |
| `BALANCE` | FLOAT64 | Balance amount left for purchases |
| `BALANCE_FREQUENCY` | FLOAT64 | How frequently balance is updated (0–1) |
| `PURCHASES` | FLOAT64 | Total amount of purchases made |
| `ONEOFF_PURCHASES` | FLOAT64 | Maximum single purchase amount |
| `INSTALLMENTS_PURCHASES` | FLOAT64 | Amount purchased in installments |
| `CASH_ADVANCE` | FLOAT64 | Cash advance given by the user |
| `PURCHASES_FREQUENCY` | FLOAT64 | How frequently purchases are made (0–1) |
| `ONEOFF_PURCHASES_FREQUENCY` | FLOAT64 | Frequency of one-off purchases (0–1) |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | FLOAT64 | Frequency of installment purchases (0–1) |
| `CASH_ADVANCE_FREQUENCY` | FLOAT64 | Frequency of cash advances (0–1) |
| `CASH_ADVANCE_TRX` | INT64 | Number of cash advance transactions |
| `PURCHASES_TRX` | INT64 | Number of purchase transactions |
| `CREDIT_LIMIT` | FLOAT64 | Credit card limit |
| `PAYMENTS` | FLOAT64 | Total payment amount made |
| `MINIMUM_PAYMENTS` | FLOAT64 | Minimum payment amount made |
| `PRC_FULL_PAYMENT` | FLOAT64 | Percentage of full payment paid |
| `TENURE` | INT64 | Tenure of credit card service (months) |

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Descriptive statistics: mean, median, std, skewness, kurtosis
- Correlation heatmap analysis between tenure, purchases, balance, and payments
- Distribution analysis across all numerical features

### 2. Feature Engineering
- Column renaming for readability
- Missing value imputation (`CREDIT_LIMIT` → median, `MINIMUM_PAYMENTS` → median)
- Outlier handling via **Winsorizer**
- Feature selection — 15 numerical features used for modeling

### 3. Dimensionality Reduction — PCA
- Applied **StandardScaler** before PCA
- Variance ratio threshold: **95%** → **11 optimal components**
- Eigenvalue threshold used to confirm component selection

### 4. Clustering — K-Means
- Optimal k determined using three methods:

| Method | Finding |
|--------|---------|
| Elbow Method | Candidate k = 5, 6, 7 |
| Silhouette Score | k=5 → score of **0.2447** (highest among candidates) |
| Silhouette Plot | k=5 selected as optimal |

- Final model: `KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=20)`

### 5. Pipeline
Full inference pipeline: `StandardScaler → PCA(n_components=11) → KMeans(k=5)`

---

## 🏷️ Cluster Results

> Based on key features: Balance Frequency, Purchases Frequency, Purchases Installments Frequency, Purchases Transaction, Balance

| Cluster | Size | Profile | Marketing Priority |
|---------|------|---------|-------------------|
| **A** | ~18% | Low-to-moderate activity. Low balance frequency and purchase frequency. Moderate balance. | 🟡 Entry-level promotions to encourage active usage |
| **B** | ~32% | High balance frequency but lowest purchase activity. Primarily maintains a running balance. | 🔴 Lowest priority — focus on encouraging active spending |
| **C** | ~18% | Moderate activity. High balance frequency but low-to-moderate purchase frequency. | 🟡 Mid-tier promotions to increase spending engagement |
| **D** | ~12% | Highest value. Maximum purchase frequency, installment usage, transactions, and balance. | 🟢 Top priority — premium offerings, card upgrades, exclusive perks |
| **E** | ~20% | High activity, similar to D but lower balance. High purchase frequency and transactions. | 🟢 High priority — premium promotions and rewards retention |

**Key insight:** Clusters D and E are the highest-value segments (~32% of customers) and should be the primary focus for premium marketing investment.

---

## 🖥️ Model Inference

The inference notebook (`02_inference.ipynb`) predicts which cluster a new customer belongs to.

**Example inference input:**

```python
data_inf = {
    'Customer ID': 9999,
    'Balance': 2000,
    'Purchases': 1000,
    'Credit Limit': 5000,
    'Payments': 800,
    'Tenure': 12,
    # ... (all 15 feature columns)
}
```

**Example output:**
```
====================================================================================================
CLUSTER PREDICTION
====================================================================================================
| Cluster Prediction: E |
```

Cluster E → **High-value customer**. Suitable for premium offerings, card upgrades, and exclusive perks.

---

## 🚀 Local Setup

### Prerequisites
- Python 3.9+
- Jupyter Notebook or JupyterLab

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/your-username/credit-card-segmentation.git
cd credit-card-segmentation
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the analysis notebook**
```bash
jupyter notebook notebooks/01_data_analysis.ipynb
```

**4. Run inference**
```bash
jupyter notebook notebooks/02_inference.ipynb
```

---

## 📁 Repository Structure

```
credit-card-segmentation/
│
├── README.md                        ← You are here
├── requirements.txt                 ← Python dependencies
│
├── data/
│   └── raw_data.csv                 ← Raw credit card transaction data (4,475 rows)
│
├── models/
│   ├── pipelines.pkl                ← Trained pipeline (StandardScaler → PCA → KMeans)
│   └── num_columns.txt              ← Feature column names used for inference
│
├── notebooks/
│   ├── 01_data_analysis.ipynb       ← EDA, feature engineering, PCA, clustering, evaluation
│   └── 02_inference.ipynb           ← Cluster prediction for new customer data
│
└── images/
    └── dataset_description.png      ← Dataset column reference
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.9 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Preprocessing** | Scikit-learn (StandardScaler), Feature-engine (Winsorizer) |
| **Dimensionality Reduction** | Scikit-learn PCA |
| **Clustering** | Scikit-learn KMeans |
| **Evaluation** | Silhouette Score, Elbow Method, ANOVA F-Statistic |
| **Model Serialization** | Pickle, JSON, Dill |
| **Notebook** | Jupyter Notebook |
| **Data Source** | Google BigQuery |

---

## 📈 Key Findings

1. **Best model**: K-Means with k=5, Silhouette Score of **0.2447** after PCA reduction to 11 components (95% variance retained)
2. **Top features** for cluster differentiation (by ANOVA F-statistic): Balance Frequency, Purchases Frequency, Purchases Installments Frequency, Purchases Transaction, Balance
3. **Tenure** does not significantly differentiate clusters — customers across all tenure lengths are evenly distributed across segments
4. **Cluster B** is the largest segment (31.90%) but the lowest spending group — requires a different engagement strategy
5. **Clusters D & E** combined represent the highest-value customers and are primary targets for premium marketing

---

## 👤 Author

**Devina Agustina**  
Batch: FTDS-052-RMT  
Phase 1 — Milestone 6

---

## 📚 References

- [Scikit-learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Yellowbrick Silhouette Visualizer](https://www.scikit-yb.org/en/latest/api/cluster/silhouette.html)
