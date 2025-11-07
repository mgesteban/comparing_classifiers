# 🧠 Comparing Classifiers for Bank Marketing Campaigns  
**An Application of the CRISP-DM Methodology**

## 📘 Project Overview
This project applies **data mining and machine learning techniques** to the **Bank Marketing dataset** from a Portuguese bank, following the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology.  

The goal is to **predict whether a client will subscribe to a long-term bank deposit** (“yes” or “no”) based on demographic, financial, and campaign-related data.  

The notebook explores and compares multiple classification models — including **Logistic Regression**, **K-Nearest Neighbors (KNN)**, **Decision Tree**, and **Support Vector Machine (SVM)** — to identify the most effective model for improving marketing efficiency.

---

## 🧩 Dataset Information
**Source:** [Moro, Cortez, and Rita (2014)](http://dx.doi.org/10.1016/j.dss.2014.03.001)  
**Paper:** *A Data-Driven Approach to Predict the Success of Bank Telemarketing*, *Decision Support Systems*.  

This dataset was collected from a Portuguese bank’s direct marketing campaigns conducted between **May 2008 and November 2010**.  

- **Records:** 41,188 (for `bank-additional-full.csv`)  
- **Attributes:** 20 input features + 1 target variable (`y`)  
- **Target:** Whether the client subscribed to a term deposit (`yes`/`no`)  

### Key Attributes
| Category | Example Attributes | Description |
|-----------|--------------------|--------------|
| **Client Information** | `age`, `job`, `marital`, `education`, `loan` | Demographic and financial details |
| **Last Contact** | `month`, `day_of_week`, `duration`, `contact` | Information about the last marketing contact |
| **Campaign Performance** | `campaign`, `pdays`, `previous`, `poutcome` | Interaction history and campaign success |
| **Economic Indicators** | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` | External economic context |

⚠️ **Important:** The `duration` variable (call length in seconds) strongly correlates with success but should **not be used in real predictive settings** since it’s only known after a call ends.

---

## 🧠 Methodology: CRISP-DM
This project follows the **CRISP-DM** framework, which consists of six iterative phases:

1. **Business Understanding** – Define objectives: improve efficiency of telemarketing campaigns.  
2. **Data Understanding** – Explore data distribution, missing values, and key relationships.  
3. **Data Preparation** – Handle missing values, encode categorical variables, and scale features.  
4. **Modeling** – Train and tune multiple classifiers (Logistic Regression, KNN, Decision Tree, SVM).  
5. **Evaluation** – Compare models using metrics such as Accuracy, Balanced Accuracy, F1-score, ROC-AUC, and PR-AUC.  
6. **Deployment** – Recommend the best-performing model for future campaigns.

---

## ⚙️ Experiments and Models
| Model | Train Accuracy | Test Accuracy | Key Observations |
|--------|----------------|----------------|------------------|
| **Logistic Regression** | 0.8873 | 0.8874 | Fast, interpretable, good baseline |
| **SVM** | 0.8882 | 0.8868 | High precision and robustness (favored in research) |
| **KNN** | 0.8915 | 0.8777 | Sensitive to scaling and feature sparsity |
| **Decision Tree** | 0.9188 | 0.8642 | Tends to overfit; requires pruning |

The **Support Vector Machine (SVM)** model provided the **best predictive performance** and was favored in prior research.

---

## 📊 Performance Summary
Example output metrics from the notebook:

```
Test Accuracy: 0.887
Test Balanced Acc: 0.5
Test F1: 0.0
Test ROC-AUC: 0.653
Test PR-AUC: 0.203
Confusion Matrix @0.5:
 [[10965     0]
 [ 1392     0]]

Best threshold by F1: 0.125
F1 @best thr: 0.26
Balanced Acc @best thr: 0.60
Confusion Matrix @best thr:
 [[7348 3617]
 [ 643  749]]
```

These results illustrate **severe class imbalance** — the “yes” responses represent only a small fraction of the dataset — highlighting the need for **threshold tuning** and **balanced metrics** (e.g., ROC-AUC, F1, Precision-Recall).

---

## 🧮 Tools and Libraries
- Python 3.x  
- Jupyter Notebook  
- scikit-learn  
- pandas, NumPy, matplotlib, seaborn  
- openpyxl (for exporting results)

---

## 🗂️ Repository Structure
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

## 🧰 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/comparing_classifiers.git
   cd comparing_classifiers
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook notebooks/comparing_classifiers.ipynb
   ```

---

## 📚 Citation
If you use this dataset or reference this project, please cite:

> **S. Moro, P. Cortez, and P. Rita (2014).**  
> *A Data-Driven Approach to Predict the Success of Bank Telemarketing.*  
> Decision Support Systems, 62, 22–31.  
> DOI: [10.1016/j.dss.2014.03.001](http://dx.doi.org/10.1016/j.dss.2014.03.001)

---

## ✍️ Author
**Grace Esteban**  
Data Science Student | Executive Assistant | Developer of AI Applications  
📍 San Francisco, California  
🔗 [LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/)
