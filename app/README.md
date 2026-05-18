#  LUNARA Application

## Feature Lifecycle Intelligence Dashboard

This directory contains the Streamlit application for the LUNARA capstone project.

The dashboard provides:
- Portfolio-level feature analytics
- Lifecycle classification
- Machine learning forecasting
- Feature deep dive analysis
- RAG-powered decision intelligence

---

#  Features

## Portfolio Overview
- Lifecycle status monitoring
- Revenue analysis
- Weekly active user tracking
- Portfolio composition visualization

## Adoption Forecasting
- Gradient Boosting forecasting
- Actual vs predicted WAU
- Trend analysis
- Forecast evaluation metrics

## Feature Deep Dive
- Feature health radar
- Engagement analysis
- Operational burden tracking
- Revenue and usage trends

## Decision Reports
- Evidence-backed recommendations
- Risk analysis
- Suggested next steps
- RAG-generated lifecycle intelligence

---

#  Technologies Used

- Streamlit
- Plotly
- Pandas
- NumPy
- Scikit-learn

---

#  Application Structure

```text
app/
│
├── app.py
├── requirements.txt
└── README.md
```

---

#  Running the Application

## Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2 — Run Streamlit

```bash
streamlit run app.py
```

---

#  Dashboard Modules

| Module | Purpose |
|---|---|
| Portfolio Overview | Portfolio-wide feature intelligence |
| Adoption Forecasting | ML-based WAU forecasting |
| Feature Deep Dive | Granular feature analytics |
| Decision Reports | Lifecycle recommendations |

---

#  Machine Learning

The application uses:
- Gradient Boosting Regressor
- Lag features
- Rolling averages
- Time-series feature engineering

to forecast future weekly active users.

---

#  Lifecycle States

| Lifecycle | Meaning |
|---|---|
| Invest | High-value growth feature |
| Maintain | Stable and healthy feature |
| Refactor | Technical debt / issue-heavy feature |
| Sunset | Low-value declining feature |

---

#  Author

Venkata Sai Chandravadan Sobila  
UMBC Data Science Capstone  
Spring 2026
