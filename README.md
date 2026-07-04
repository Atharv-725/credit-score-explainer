# 💳 Credit Risk Predictor

An interactive Streamlit app that predicts loan default risk from financial
inputs using a Random Forest classifier, with real-time probability breakdowns.

**Note:** this is a UI/ML-serving demo, not a formal explainability tool —
see "Current limitations" below.

## How it works

```
   User Input (Streamlit sidebar)
   8 scaled financial features
              │
              ▼
   ┌───────────────────────┐
   │ Random Forest Model    │
   │ (100 estimators,       │
   │  scikit-learn)         │
   └──────────┬─────────────┘
              │
              ▼
   ┌───────────────────────┐
   │ predict() +            │
   │ predict_proba()        │
   └──────────┬─────────────┘
              │
              ▼
   ┌───────────────────────────────────┐
   │ UI Output                          │
   │ • Risk label (Low / High)          │
   │ • Confidence %                     │
   │ • Probability breakdown            │
   │ • Input summary table              │
   └───────────────────────────────────┘
```

## Input features

| # | Feature | Scale | Description |
|---|---|---|---|
| 1 | Income | -3.0 to 3.0 | Standardized annual income |
| 2 | Age | -3.0 to 3.0 | Standardized applicant age |
| 3 | Loan Amount | -3.0 to 3.0 | Standardized requested loan size |
| 4 | Credit Score | -3.0 to 3.0 | Standardized credit score |
| 5 | Employment Years | -3.0 to 3.0 | Standardized years employed |
| 6 | Debt Ratio | -3.0 to 3.0 | Standardized debt-to-income ratio |
| 7 | Number of Credit Lines | -3.0 to 3.0 | Standardized count of open credit lines |
| 8 | Payment History | -3.0 to 3.0 | Standardized payment history score |

*All features are on a standardized (mean-0) scale to match the synthetic
training distribution — see "Current limitations" below.*

## What it does

- Interactive sidebar sliders for 8 financial features (income, age, loan
  amount, credit score, employment years, debt ratio, number of credit
  lines, payment history)
- Real-time default-risk prediction with confidence score
- Risk probability breakdown (No Default vs. Default Risk)
- Input summary table for transparency on what was fed into the model

## Model

- **Random Forest Classifier** (scikit-learn, 100 estimators)
- Trained on a **synthetic dataset** generated with `sklearn.datasets.make_classification`
  (8 features, 6 informative, 2 redundant) — feature names (income, credit
  score, etc.) are illustrative labels on synthetic data, not real financial
  records.
- Inputs are entered on a standardized (scaled) axis, matching the synthetic
  training distribution.

## Current limitations

- **No formal explainability yet** — despite the repo name, there's no
  SHAP/LIME feature-attribution layer at the moment. The app currently
  shows *what* the model predicts, not *why*. Adding SHAP values per
  prediction is the natural next step to make this a true "explainer."
- **Synthetic data only** — no real-world credit dataset used yet, so the
  feature relationships are artificial. Swapping in a public dataset (e.g.
  UCI German Credit or Kaggle's "Give Me Some Credit") would make the
  model's behavior meaningful rather than illustrative.

## Tech Stack
Python · Streamlit · Scikit-learn · Pandas · NumPy

## Installation
```bash
pip install streamlit scikit-learn pandas numpy
streamlit run app.py
```

## Live Demo
https://a725-credit-score-explainer.streamlit.app/
