#  LUNARA Data Directory

## Feature Lifecycle Intelligence Dashboard

This folder contains all raw and processed datasets used in the LUNARA project.

---

#  Directory Structure

```text
data/
│
├── Data.md
└── README.md
```

---

#  Raw Data

The raw datasets include:
- Google Analytics sample engagement data
- GitHub Issues operational data
- NASA KC1 defect dataset

These datasets were used to generate:
- Engagement metrics
- Operational burden metrics
- Defect risk indicators

---
The Processed data set will be created when you have run the notebook file
#  Processed Data

Processed datasets include:
- weekly_feature_data.csv
- feature_value_scores.csv
- forecast_predictions.csv
- decision_reports.json

These files are generated through the preprocessing and modeling pipeline.

---

#  Processed Dataset Descriptions

| File | Description |
|---|---|
| weekly_feature_data.csv | Weekly feature-level analytics |
| feature_value_scores.csv | Composite lifecycle scores |
| forecast_predictions.csv | ML forecasting outputs |
| decision_reports.json | RAG-generated recommendations |

---

#  Engineered Metrics

The processed datasets include:
- Weekly Active Users
- Engagement Intensity
- Issue Burden
- Defect Risk
- Lifecycle Score
- Revenue Metrics
- Bounce Rate

---

#  Data Pipeline

The pipeline includes:
1. Data ingestion
2. Data cleaning
3. Weekly aggregation
4. Feature engineering
5. Lifecycle scoring
6. ML forecasting
7. Decision intelligence generation

---

#  Notes

- Processed datasets are generated automatically from the notebook pipeline.
- Forecasting outputs are regenerated whenever the notebook is rerun.

---

#  Formats Used

| Format | Purpose |
|---|---|
| CSV | Structured tabular datasets |
| JSON | Decision intelligence reports |

---

#  Author

Venkata Sai Chandravadan Sobila  
UMBC Data Science Capstone  
Spring 2026
