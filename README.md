# UMBC-DATA606-Capstone

# Lunara - Feature Lifecycle Intelligence System

## Project Information

* **Prepared for:** UMBC Data Science Master’s Degree Capstone — Dr. Chaojie (Jay) Wang
* **Author:** Venkata Sai Chandravadan Sobila
* **GitHub Repository:** [View Github](https://github.com/Chandravadan30/UMBC-DATA606-Capstone)
* **LinkedIn Profile:** [Linkedin](https://www.linkedin.com/in/chandravadan-s-960447214/)
* **Presentation Slides:** [Slides](https://docs.google.com/presentation/d/1HhX2ESY866Q_5Yu7D9NPph2QsxaGM4TNkazZ3F9_PHQ/edit?usp=sharing)
* **YouTube Demo Video:** (to be added)

---

#  Project Overview

Modern digital applications contain multiple product features such as:

* Search
* Recommendations
* Wishlist
* Reviews
* Notifications
* Checkout

Over time, some features grow in popularity and business value, while others experience declining usage, increasing operational issues, or higher maintenance burden.

Product managers and engineering teams must continuously decide whether to:

*  Invest in a feature
*  Maintain it
*  Refactor it
*  Sunset it completely

These decisions are often based on fragmented dashboards or subjective judgment.

This project builds an end-to-end Feature Lifecycle Intelligence System for a fictional e-commerce platform called **Lunara**, using publicly available datasets as proxies for product behavior, operational burden, and software quality.

---

#  Research Questions

1. Can historical user behavior forecast future feature adoption?
2. Can engagement, operational burden, and defect risk be combined into a transparent feature value score?
3. Can lifecycle categories such as Invest, Maintain, Refactor, and Sunset be generated using analytical scoring?
4. Can Retrieval-Augmented Generation (RAG) generate evidence-grounded lifecycle explanations?

---

#  Datasets Used

This project uses medium-scale public datasets totaling approximately 300–350 MB of raw data.

---

## 1️⃣ User Behavior Dataset

### Source

Google Analytics BigQuery Public Dataset

```sql
bigquery-public-data.google_analytics_sample.ga_sessions_*
```

### Dataset Details

* Approx. 350,000 rows
* Approx. 200 MB
* Each row represents one user session

### Key Fields

* user_id
* session_start
* pageviews
* transactions
* device
* traffic_source
* country

### Usage

This dataset is used to analyze:

* Feature adoption
* User engagement
* Revenue contribution
* Time-series usage trends

---

## 2️⃣ Operational Issues Dataset

### Source

GitHub Public Dataset (BigQuery)

### Dataset Details

* Approx. 200,000 issues
* Approx. 100–120 MB
* Each row represents an issue or feature request

### Key Fields

* repo_name
* created_at
* state
* title
* body

### Usage

This dataset is used as a proxy for:

* Operational burden
* User-reported problems
* Feature instability

---

## 3️⃣ Software Quality Dataset

### Source

NASA Software Defect Dataset

### Dataset Details

* Approx. 10,000 rows
* Approx. 5 MB

### Usage

This dataset is used as a proxy for:

* Technical debt
* Defect risk
* Maintenance complexity

---

#  Derived Weekly Dataset

A weekly aggregated dataset is created during ETL processing.

### Granularity

```text
1 row = 1 feature × 1 week
```

### Key Columns

* weekly_active_users
* engagement metrics
* issue count
* defect risk score
* feature age
* feature type

### Target Variables

* Forecasting target → weekly_active_users
* Lifecycle target → Invest / Maintain / Refactor / Sunset

---

#  Exploratory Data Analysis (EDA)

EDA was performed using:

* Jupyter Notebook
* Pandas
* NumPy
* Plotly Express

---

#  EDA Objectives

The primary goals of analysis were:

* Understand feature usage behavior
* Identify growth and decline trends
* Examine issue burden impact
* Analyze defect risk distribution
* Validate data quality and mappings

---

#  EDA Workflow

## Data Inspection

* Load datasets
* Validate schema
* Inspect sample records

## Data Overview

* Review data types
* Generate summary statistics
* Analyze numerical and categorical features

## Missing Value Analysis

* Detect missing values
* Measure missing percentages
* Assess impact on analysis

## Duplicate Detection

* Identify duplicate records
* Remove redundant rows
* Improve dataset reliability

## Feature Usage Analysis

* Compare feature engagement
* Detect high and low adoption features
* Analyze usage distribution

## Time-Series Analysis

* Track feature usage trends over time
* Detect spikes and declining behavior
* Visualize adoption movement

## Correlation Analysis

* Examine relationships between:

  * Issue count vs usage
  * Engagement vs stability

## Defect Risk Analysis

* Study defect distribution
* Detect high-risk features
* Identify operational outliers

## Feature Mapping Validation

* Validate dataset consistency
* Verify feature alignment across datasets

---

#  Key Insights

The EDA process revealed:

* Features with declining adoption trends
* Features with high issue burden and low engagement
* Stable features with healthy usage
* Operational patterns impacting adoption

---

#  Feature Value Scoring

A transparent 0–100 feature value score is generated using weighted metrics.

## Scoring Components

| Component                | Weight |
| ------------------------ | ------ |
| Adoption Trend           | +30%   |
| Engagement Intensity     | +25%   |
| Operational Issue Burden | -25%   |
| Defect Risk              | -20%   |

The scoring framework provides a measurable indicator of overall feature health.

---

#  Lifecycle Intelligence Categories

Features are categorized into lifecycle stages using score thresholds and trend analysis.

| Category    | Meaning                           |
| ----------- | --------------------------------- |
|  Invest   | High growth and engagement        |
|  Maintain | Stable and healthy                |
|  Refactor | Technical or operational concerns |
|  Sunset   | Declining adoption and low value  |

---

#  Adoption Forecasting

The system forecasts future weekly active users using historical behavioral patterns.

## Forecasting Components

* Historical trend analysis
* Rolling averages
* Time-series comparison
* Actual vs predicted visualization

## Evaluation Metrics

* RMSE
* MAPE
* Trend consistency analysis

---

#  Explainable AI Layer (RAG)

The platform includes a Retrieval-Augmented Generation (RAG) layer for evidence-grounded lifecycle recommendations.

## RAG Workflow

```text
Engineering Documents
        ↓
Document Chunking
        ↓
Embedding Generation
        ↓
Vector Database
        ↓
Evidence Retrieval
        ↓
Decision Report Generation
```

## Generated Outputs

* Evidence cards
* Lifecycle reasoning
* Risk analysis
* Recommended next steps

---

#  Streamlit Dashboard

The project includes an interactive executive dashboard built using Streamlit.

---

#  Dashboard Modules

## Portfolio Overview

* Feature portfolio health
* Lifecycle distribution
* Revenue analysis
* Engagement heatmaps
* Operational trends

## Adoption Forecasting

* Actual vs predicted WAU
* Forecast trend analysis
* Performance metrics

## Feature Deep Dive

* Health radar charts
* Trend analysis
* Feature-level metrics
* Risk visibility

## Decision Reports

* RAG-generated lifecycle reports
* Evidence-backed recommendations
* Risk assessments
* Suggested next actions

---

#  Dashboard Screenshots

## Portfolio Dashboard

<img width="1512" height="831" alt="Screenshot 2026-05-16 at 21 20 58" src="https://github.com/user-attachments/assets/fc99b814-aa49-48bb-88a5-3b953976a4eb" />


## Forecast Dashboard

<img width="1491" height="821" alt="Screenshot 2026-05-16 at 21 22 59" src="https://github.com/user-attachments/assets/b9d12935-db3b-45ab-bb8b-61e801dc1393" />


## Feature Deep Dive

<img width="1512" height="825" alt="Screenshot 2026-05-16 at 21 24 03" src="https://github.com/user-attachments/assets/b9c86dd5-6424-48e4-934b-2366e7d98dcd" />


## RAG Decision Reports

<img width="1512" height="822" alt="Screenshot 2026-05-16 at 21 24 24" src="https://github.com/user-attachments/assets/25dd8ee2-5ce6-4303-aa64-4a4e2e5eeece" />


---

#  Technology Stack

## Languages

* Python

## Data Analysis

* Pandas
* NumPy

## Visualization

* Plotly
* Plotly Express

## Dashboard

* Streamlit

## AI / Explainability

* Retrieval-Augmented Generation (RAG)

---

#  Project Structure

```text
Lunara/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   └── Lunara_Analysis.ipynb
│
├── app.py
├── requirements.txt
├── README.md
│
└── outputs/
    ├── reports/
    ├── forecasts/
    └── charts/
```

---

#  How to Run

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Chandravadan30/UMBC-DATA606-Capstone.git
cd UMBC-DATA606-Capstone
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run Analysis Notebook

```bash
jupyter notebook
```

Open:

```text
Lunara_Analysis.ipynb
```

Run all notebook cells.

---

## 4️⃣ Launch Dashboard

```bash
streamlit run app.py
```

---

#  Example Insights

* Checkout showed stable growth and strong engagement →  Invest
* Notifications showed increased operational burden →  Refactor
* Wishlist showed declining adoption →  Sunset
* Recommendations maintained healthy engagement →  Maintain

---

#  Future Enhancements

* Real-time telemetry integration
* Cloud deployment
* Automated experimentation recommendations
* Advanced causal analysis
* Real production monitoring integration

---

#  Academic Information

* **University:** University of Maryland Baltimore County (UMBC)
* **Program:** Master’s in Data Science
* **Capstone Project**
* **Instructor:** Dr. Chaojie (Jay) Wang

---

#  Author

## Venkata Sai Chandravadan Sobila

* GitHub: https://github.com/Chandravadan30/UMBC-DATA606-Capstone


---

#  Final Outcome

This project demonstrates:

End-to-end data analysis
Time-series forecasting
Feature lifecycle intelligence
Explainable AI integration
Interactive dashboard development
Product analytics workflows

The platform combines analytics, operational intelligence, and explainable AI into a production-style feature lifecycle intelligence system.

---

#  References

* Google Analytics Public Dataset
* GitHub Public Dataset
* NASA Software Defect Dataset
* Streamlit Documentation
* Plotly Documentation

---
