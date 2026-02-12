# Feature Lifecycle Intelligence System 

## Project Information

- **Prepared for:** UMBC Data Science Master’s Degree Capstone — Dr. Chaojie (Jay) Wang  
- **Author:** Venkata Sai Chandravadan Sobila  
- **GitHub Repository:** [View Repository](https://github.com/Chandravadan30/UMBC-DATA606-Capstone)  
- **LinkedIn Profile:** [Connect on LinkedIn](https://www.linkedin.com/in/chandravadan-sobila-960447214/)  
- **Presentation Slides:** (to be added)  
- **YouTube Demo Video:** (to be added)


---

## 1. Background

Modern digital applications contain multiple product features such as Search, Recommendations, Wishlist, Reviews, Notifications, and Checkout. Over time, some features grow in popularity and business value, while others experience declining usage, increasing operational issues, or high maintenance cost.

Product managers and engineering teams must continuously decide whether to:

- Invest in a feature
- Maintain it as-is
- Refactor it due to technical debt
- Sunset (retire) it completely

These decisions are often based on fragmented dashboards or subjective opinions. There is a need for a structured, data-driven, and explainable system that supports feature lifecycle decisions.

This project builds an end-to-end intelligence system for a fictional e-commerce application called **Lunara**, using publicly available real datasets as proxies for user behavior, operational burden, and software quality.

---

## 2. Research Questions

1. Can historical user behavior data forecast future feature adoption?
2. Can usage, engagement, operational burden, and defect risk be combined into a transparent feature value score?
3. Can machine learning models classify features into lifecycle categories such as Invest, Maintain, Refactor, or Sunset?
4. Can Retrieval-Augmented Generation (RAG) generate evidence grounded explanations for lifecycle decisions?

---

## 3. Data

This project uses real public datasets (medium scale) totaling approximately 300–350 MB of raw data.

### 3.1 User Behavior Dataset

**Source:** Google Analytics BigQuery Public Dataset  
Dataset: `bigquery-public-data.google_analytics_sample.ga_sessions_*`

- Approx. 350,000 rows (subset)
- Approx. 200 MB
- Each row represents one user session
- Key fields:
  - user_id
  - session_start
  - pageviews
  - transactions
  - device
  - traffic source
  - country

This dataset is used to model feature adoption, engagement, and time-series trends.

---

### 3.2 Operational Issues Dataset

**Source:** GitHub Public Dataset (BigQuery)

- Approx. 200,000 issues (subset)
- Approx. 100–120 MB
- Each row represents one issue (bug report or feature request)
- Key fields:
  - repo_name
  - created_at
  - state
  - title
  - body

This dataset is used as a proxy for operational burden and user-reported problems.

---

### 3.3 Software Quality Dataset

**Source:** NASA Software Defect Dataset

- Approx. 10,000 rows
- Approx. 5 MB
- Each row represents one software module
- Includes complexity metrics and defect indicators

This dataset is used as a proxy for technical debt and maintenance risk.

---

### 3.4 Derived Dataset (Created via ETL)

A weekly aggregated dataset will be created:

Granularity:
- One row = one feature × one week

Estimated size:
- 800–1200 rows

Key columns:
- weekly_active_users (target for forecasting)
- engagement metrics
- issue count
- defect risk score
- feature age
- feature type

Target variables:
- Regression target: weekly_active_users
- Classification target: lifecycle label (Invest / Maintain / Refactor / Sunset)

---

## 4. Exploratory Data Analysis (EDA)

EDA will be conducted using Jupyter Notebook and Plotly Express.

Planned analyses:
- Summary statistics of usage and engagement
- Time-series visualization of feature trends
- Correlation between issue count and adoption decline
- Defect risk distribution analysis
- Missing value and duplicate checks
- Validation of feature mapping rules

---

## 5. Model Training

### 5.1 Adoption Forecasting

Objective:
Predict weekly active users for each feature for the next 4–12 weeks.

Models:
- Baseline: rolling average
- Primary model: LightGBM Regressor with lag features

Evaluation:
- RMSE
- MAPE
- Time-based backtesting

---

### 5.2 Feature Value Score

A composite 0–100 value score will be computed using:

- Adoption trend (positive weight)
- Engagement intensity (positive weight)
- Operational issue burden (negative weight)
- Defect risk proxy (negative weight)

The scoring logic will be transparent and documented.

---

### 5.3 Lifecycle Classification

Target Classes:
- Invest
- Maintain
- Refactor
- Sunset

Approach:
- Generate heuristic labels based on score thresholds
- Train LightGBM Classifier
- Output predicted class and confidence score

Evaluation:
- Accuracy
- Macro F1-score
- Confusion matrix

---

## 6. Application of the Trained Models

A Streamlit web application will be developed with:

1. Executive Overview:
   - Feature health summary
   - Risk ranking
   - Growth opportunities

2. Feature Drilldown:
   - Adoption trend and forecast
   - Value score breakdown
   - Lifecycle recommendation with confidence
   - Evidence cards (RAG)

3. Portfolio Simulator:
   - What-if analysis for reducing issues or defect risk

4. Decision Log:
   - Save lifecycle decisions with timestamps

---

## 7. AI Explainability Layer (RAG)

Retrieval-Augmented Generation will provide evidence-based explanations.

Process:
- Collect real public engineering documents
- Chunk and embed documents
- Store embeddings in a vector database
- Retrieve relevant evidence for each feature context
- Generate structured decision reports

The system will not function as a chatbot but as a decision report generator.

---

## 8. Conclusion

This project delivers an end-to-end feature lifecycle intelligence system that integrates multiple real datasets, machine learning models, and explainable AI techniques.

Potential Applications:
- Product portfolio management
- Data-driven feature prioritization
- Technical debt monitoring

Limitations:
- Public datasets are proxies for real internal data
- Feature mapping assumptions may influence results

Future Work:
- Integration with real production telemetry
- Advanced causal modeling
- Automated experimentation recommendations

---
