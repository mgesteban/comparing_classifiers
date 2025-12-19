# Comparing Classifiers for Bank Marketing Campaigns  
**An Application of the CRISP-DM Methodology**

---

## Project Overview
This project applies data mining and machine learning techniques to the Bank Marketing dataset from a Portuguese bank, following the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology.  

The goal is to predict whether a client will subscribe to a long-term bank deposit (“yes” or “no”) based on demographic, financial, and campaign-related data.  

This project applies data mining and machine learning techniques to the Bank Marketing dataset from a Portuguese bank, following the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology.

The goal is to predict whether a client will subscribe to a long-term bank deposit (“yes” or “no”) based on demographic, financial, and campaign-related data.

In this notebook, I performed an initial comparison of several classification models — Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Support Vector Machine (SVM) — to establish baseline results. After this, I focused primarily on Logistic Regression, performing hyperparameter tuning and feature engineering to improve its predictive performance and interpretability.

---

## Dataset Information
**Source:** [Moro, Cortez, and Rita (2014)](http://dx.doi.org/10.1016/j.dss.2014.03.001)  
**Paper:** *A Data-Driven Approach to Predict the Success of Bank Telemarketing*, *Decision Support Systems*  

This dataset was collected from a Portuguese bank’s direct marketing campaigns conducted between **May 2008 and November 2010**.  

- **Records:** 41,188 (`bank-additional-full.csv`)  
- **Attributes:** 20 input features + 1 target variable (`y`)  
- **Target:** Whether the client subscribed to a term deposit (`yes` / `no`)  

### Key Attributes
| Category | Example Attributes | Description |
|-----------|--------------------|--------------|
| **Client Information** | `age`, `job`, `marital`, `education`, `loan` | Demographic and financial details |
| **Last Contact** | `month`, `day_of_week`, `duration`, `contact` | Information about the last marketing contact |
| **Campaign Performance** | `campaign`, `pdays`, `previous`, `poutcome` | Interaction history and campaign success |
| **Economic Indicators** | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` | External economic context |

---

## Methodology: CRISP-DM
This project follows the **CRISP-DM** framework, which consists of six iterative phases:

1. **Business Understanding** – Define objectives: improve efficiency of telemarketing campaigns.  
2. **Data Understanding** – Explore data distribution, missing values, and key relationships.  
3. **Data Preparation** – Handle missing values, encode categorical variables, and scale features.  
4. **Modeling** – Train and tune multiple classifiers (Logistic Regression, KNN, Decision Tree, SVM).  
5. **Evaluation** – Compare models using metrics such as Accuracy, Balanced Accuracy, F1-score, ROC-AUC, and PR-AUC.  
6. **Deployment** – Recommend the best-performing model for future campaigns.

---

## Experiments and Models
| Model | Train Accuracy | Test Accuracy | Key Observations |
|--------|----------------|----------------|------------------|
| **Logistic Regression** | 0.8873 | 0.8874 | Fast, interpretable, good baseline |
| **SVM** | 0.8882 | 0.8868 | High precision and robustness (favored in research) |
| **KNN** | 0.8915 | 0.8777 | Sensitive to scaling and feature sparsity |
| **Decision Tree** | 0.9188 | 0.8642 | Tends to overfit; requires pruning |

The **Support Vector Machine (SVM)** model provided the **best predictive performance** and was favored in prior research.  
However, I decided to focus on **Logistic Regression** because it is simpler, faster, and more interpretable — making it easier to understand how changes in parameters directly affect performance.  

While SVM can achieve strong accuracy, it is **computationally intensive** and sensitive to hyperparameters like kernel type, gamma, and C, which require far more time and processing power to tune properly.  
In contrast, Logistic Regression’s main hyperparameter — the **regularization strength (C)** — can be optimized efficiently and offers clear insights into the trade-off between bias and variance.  

This makes Logistic Regression a **practical and educational choice** for improving performance while maintaining interpretability and computational efficiency.

---

### New Baseline After Model Improvements

After all the improvements, the new baseline is **no longer the naïve 88.7% majority-class accuracy**.  
It is now defined by the performance of the **tuned Logistic Regression model**, which becomes the new benchmark for any future models (such as SVM or Random Forest) to beat.

**New Baseline Performance:**

- **Accuracy:** ~0.88  
- **ROC-AUC:** ~0.65  
- **F1-score:** ~0.26  
- **Precision–Recall AUC:** ~0.20  

This updated baseline reflects a **meaningful improvement** — the model not only maintains high accuracy but also begins to **identify true positive cases** (“Yes” responses), offering real predictive value for the bank’s marketing campaigns.


````
Tools and Libraries

Python 3.x
Jupyter Notebook
scikit-learn
pandas, NumPy, matplotlib, seaborn
openpyxl (for exporting results)

---

## Repository Structure

```
comparing_classifiers/
│
├── data/
│   ├── bank-additional.csv
│   ├── bank-additional-full.csv
│   └── bank-additional-names.txt
│
├── notebooks/
│   └── comparing_classifiers.ipynb
│
├── docs/
│   └── CRISP-DM-BANK.pdf
│
├── README.md
└── requirements.txt
```

---

## How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/comparing_classifiers.git
   cd comparing_classifiers
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Open the notebook**

   ```bash
   jupyter notebook notebooks/comparing_classifiers.ipynb
   ```

---

## Citation

If you use this dataset or reference this project, please cite:

**S. Moro, P. Cortez, and P. Rita (2014).**
*A Data-Driven Approach to Predict the Success of Bank Telemarketing.*
*Decision Support Systems, 62*, 22–31.
DOI: [10.1016/j.dss.2014.03.001](http://dx.doi.org/10.1016/j.dss.2014.03.001)

---

## Author

**Grace Esteban**
Developer of AI Applications for Executive Assistants
📍 San Francisco, California
🔗 [LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/)

