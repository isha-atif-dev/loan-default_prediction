# 💳 Loan Default Prediction

A machine learning project that predicts whether a loan applicant will default (fail to repay) — helping banks make smarter, data-driven lending decisions.

---

## 📌 What This Project Does

Banks risk losing millions when borrowers fail to repay loans. This project builds and compares **three machine learning models** to predict loan default risk, selects the best performing model, and evaluates it using industry-standard metrics.

---

## 📁 Project Structure

```
loan-default-prediction/
│
├── data/
│   ├── raw/                        # Original dataset — never modified
│   └── processed/                  # Cleaned and split data ready for training
│
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory Data Analysis
│
├── src/
│   ├── preprocess.py               # Missing values, outliers, encoding, train-test split
│   ├── train.py                    # Training & comparing 3 models, saving the best
│   └── evaluate.py                 # Deep evaluation with precision, recall, F1
│
├── models/
│   └── loan_model.pkl              # Saved best model (Random Forest)
│
├── requirements.txt                
└── README.md                       
```

---

## 🔍 Dataset

**Source:** [Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

- 32,581 loan applicants
- 12 features including income, employment length, loan intent, loan grade, interest rate
- Target variable: `loan_status` (1 = defaulted, 0 = repaid)

---

## 🧠 Key Findings from EDA

- **79% repaid, 21% defaulted** — class imbalance present
- **Two columns had missing values:** `person_emp_length` (895 missing) and `loan_int_rate` (3,116 missing)
- **Outliers detected** in `person_emp_length` — values up to 123 years (impossible, capped at 60)
- Multiple categorical columns required One Hot Encoding

---

## ⚙️ How It Works

**Step 1 — Preprocessing (`preprocess.py`)**
- Filled missing values: `person_emp_length` with median (4.0), `loan_int_rate` with mean (11.01%)
- Capped outliers in `person_emp_length` at 60 years using `clip(upper=60)`
- Applied One Hot Encoding with `drop_first=True` to avoid the Dummy Variable Trap
- Split data: 80% training (26,064 rows), 20% testing (6,517 rows)

**Step 2 — Training (`train.py`)**
- Trained and compared three models with `class_weight='balanced'`:
  - Random Forest
  - Logistic Regression
  - Decision Tree
- Best model selected based on accuracy and saved to `models/loan_model.pkl`

**Step 3 — Evaluation (`evaluate.py`)**
- Deep evaluation of the best model on 6,517 unseen applicants

---

## 📊 Model Comparison

| Model | Accuracy |
|-------|----------|
| **Random Forest** | **92.86%** 🏆 |
| Decision Tree | 89.02% |
| Logistic Regression | 79.08% |

---

## 📈 Final Model Results (Random Forest)

| Metric | Score |
|--------|-------|
| Accuracy | 92.86% |
| Precision | 95.28% |
| Recall | 71.34% |
| F1 Score | 81.59% |

> **Precision 95%** means when the model flags someone as likely to default, it is correct 95% of the time — very few false alarms.
> **Recall 71%** means the model catches 71% of actual defaulters — a significant improvement over baseline.

---

## 🆚 Improvement Over Previous Project

This project builds directly on the [Churn Prediction](https://github.com/isha-atif-dev/churn-prediction) project with notable improvements:

| Metric | Churn Project | Loan Project |
|--------|-------------|--------------|
| Accuracy | 86.95% | 92.86% |
| Recall | 47.07% | 71.34% |
| Precision | 77.73% | 95.28% |
| F1 Score | 58.63% | 81.59% |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data manipulation and imputation |
| Scikit-learn | ML models and metrics |
| Matplotlib / Seaborn | Data visualisation and outlier detection |
| Joblib | Model saving and loading |
| Jupyter Notebook | EDA and exploration |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/isha-atif-dev/loan-default_prediction.git
cd loan-default_prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run preprocessing**
```bash
python src/preprocess.py
```

**4. Train and compare models**
```bash
python src/train.py
```

**5. Evaluate the best model**
```bash
python src/evaluate.py
```

---

## 💡 What I Learned

- Handling **missing values** using mean and median imputation
- Detecting and handling **outliers** using box plots and `clip()`
- Training and **comparing multiple ML models** to select the best
- **Logistic Regression** — how it predicts probability using a sigmoid function
- **Dummy Variable Trap** — why `drop_first=True` matters in One Hot Encoding
- Why **Recall matters more than accuracy** for imbalanced classification problems
- How **class_weight='balanced'** helps models learn from minority classes

---

## 👩‍💻 Author

**Isha Atif**  
MRes Applied Artificial Intelligence Student  
[GitHub](https://github.com/isha-atif-dev) • [LinkedIn](https://linkedin.com/in/isha-atif)