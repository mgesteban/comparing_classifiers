# Comparing Classifiers: Bank Marketing Campaign Analysis

A machine learning project comparing the performance of K-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines to predict term deposit subscriptions from bank marketing campaigns.

## Project Overview

This project analyzes data from Portuguese banking institution marketing campaigns to predict whether a client will subscribe to a term deposit. The analysis compares four classification algorithms to determine which provides the best performance for this business problem.

**Business Objective**: Predict whether a client will subscribe to a term deposit, enabling the bank to improve the efficiency of future marketing campaigns and optimize client targeting.

## Dataset

**Source**: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

**Description**: Collection of results from 17 marketing campaigns conducted by a Portuguese banking institution via telephone.

**Records**: 41,188 observations  
**Features**: 20 input variables + 1 target variable  
**Target**: Binary classification (yes/no) - Has the client subscribed to a term deposit?

### Feature Categories

#### Bank Client Data
- Age, job type, marital status, education level
- Credit default status, housing loan, personal loan

#### Campaign Contact Information
- Contact communication type, contact month, day of week
- Contact duration (note: excluded from realistic predictive models)

#### Campaign History
- Number of contacts in current campaign
- Days since last contact from previous campaign
- Previous campaign outcome

#### Economic Context Indicators
- Employment variation rate
- Consumer price index
- Consumer confidence index
- Euribor 3-month rate
- Number of employees

## Project Structure

```
comparing_classifiers/
├── README.md
├── comparing_classifiers.ipynb           # Main analysis notebook
├── CRISP-DM-BANK.pdf         # Research paper reference
├── data/
│   ├── bank-additional-full.csv
│   ├── bank-additional.csv
│   └── bank-additional-names.txt
└── comparing_classifiers.code-workspace
```

## Installation & Setup

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Required Libraries
```bash
pip install pandas numpy scikit-learn
```

### Running the Analysis
```bash
# Clone the repository (if applicable)
# Navigate to project directory
cd comparing_classifiers

# Launch Jupyter Notebook
jupyter notebook prompt_III.ipynb
```

## Methodology

### Data Preprocessing
1. **Handling Missing Values**: "Unknown" values treated as separate category
2. **Feature Engineering**: Focus on bank client information features
   - Numeric: Age (StandardScaler)
   - Categorical: Job, marital status, education, default, housing, loan (One-Hot Encoding)
3. **Train/Test Split**: 70/30 split with stratification to maintain class balance
4. **Class Imbalance**: ~89% negative class, ~11% positive class

### Models Compared

| Model | Configuration | Key Characteristics |
|-------|--------------|-------------------|
| **Logistic Regression** | `max_iter=1000, class_weight='balanced'` | Baseline model, interpretable coefficients |
| **K-Nearest Neighbors** | Default settings | Instance-based learning, no training phase |
| **Decision Tree** | `random_state=42` | Non-linear boundaries, prone to overfitting |
| **Support Vector Machine** | Default settings | Finds optimal decision boundary |

## Key Findings

### Model Performance Summary

| Model | Train Time (s) | Train Accuracy | Test Accuracy |
|-------|---------------|----------------|---------------|
| **Logistic Regression** | 0.16 | 88.87% | 88.74% |
| **SVM** | 45.47 | 91.88% | 88.68% |
| **KNN** | 0.09 | 89.30% | 87.77% |
| **Decision Tree** | 0.23 | 91.88% | 86.42% |

### Insights

1. **Baseline Performance**: Naive prediction (always "no") achieves ~89% accuracy due to class imbalance
2. **Best Performer**: SVM and Logistic Regression show similar test accuracy (~88.7%) with good generalization
3. **Training Efficiency**: Logistic Regression offers excellent balance of performance and speed
4. **Overfitting Concern**: Decision Tree shows highest training accuracy but lowest test accuracy
5. **Trade-off**: Using `class_weight='balanced'` in Logistic Regression reduces overall accuracy (59.4%) but significantly improves minority class detection

### Business Impact

- All models perform near baseline, indicating the complexity of predicting client behavior
- SVM provides most consistent performance across training and test sets
- Further improvement requires:
  - Hyperparameter tuning
  - Additional feature engineering
  - Balancing metrics (precision/recall vs accuracy)
  - Considering economic context features

## Next Steps

1. **Hyperparameter Optimization**: Grid search for optimal model parameters
2. **Feature Expansion**: Include campaign and economic context features
3. **Metric Adjustment**: Focus on recall/precision for positive class
4. **Ensemble Methods**: Explore Random Forests, Gradient Boosting
5. **Cost-Sensitive Learning**: Account for business costs of false positives/negatives

## References

- [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier.
- UCI Machine Learning Repository: Bank Marketing Dataset
- Included research paper: `CRISP-DM-BANK.pdf`

## License

Dataset license and usage rights governed by UCI Machine Learning Repository terms.

## Acknowledgments

- Dataset provided by UCI Machine Learning Repository
- Original research by S. Moro, P. Cortez, and P. Rita
- Portuguese banking institution for data collection

---

**Note**: This project is for educational purposes as part of a practical application assignment comparing classification algorithms.
