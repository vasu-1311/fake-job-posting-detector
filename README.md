# Fake Job Posting Detector

A machine learning classifier that flags fraudulent job postings, built after
personally running into fake "0-2 years experience" fresher listings during
my own job search.

## Problem

The dataset (Kaggle, "Real / Fake Job Postings") has 17,880 postings, only
4.84% of which are fraudulent — a highly imbalanced classification problem
where accuracy alone would be a misleading metric.

## Key findings from EDA

- Fake postings are missing a company profile 67.8% of the time, vs 16.0%
  for real postings.
- Only 33% of fake postings have a company logo, vs 82% of real ones.
- The phrase "work from home" appears in 9.93% of fake postings but only
  0.46% of real ones — a ~21x difference.
- "No experience" appears ~13x more often in fake postings.

These became the core engineered features, rather than relying on raw text
alone.

## Project status

- [x] EDA — missingness patterns, keyword red-flags
- [x] Feature engineering (`data_prep.py`)
- [ ] Train/test split (company-aware, to avoid leakage)
- [ ] Model training and comparison (Logistic Regression, Random Forest, XGBoost)
- [ ] Evaluation using Precision-Recall AUC (not accuracy)
- [ ] SHAP-based interpretability

## Setup

```bash
pip install -r requirements.txt
python data_prep.py    # produces processed_data.csv
```

## Data source

[Real / Fake Job Posting Prediction — Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
