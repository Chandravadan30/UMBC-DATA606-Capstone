# 1. Title and Author

## Project Title

# LUNARA — Feature Lifecycle Intelligence Dashboard

Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang

---

## Author Information

- **Author Name:** Venkata Sai Chandravadan Sobila  
- **Program:** Master’s in Data Science  
- **University:** University of Maryland, Baltimore County (UMBC)  
- **Semester:** Spring 2026  

---

## Project Links

### GitHub Repository
[GitHub](https://github.com/Chandravadan30/UMBC-DATA606-Capstone)

### LinkedIn Profile
[LinkedIn](https://www.linkedin.com/in/chandravadan-s-960447214/)

### PowerPoint Presentation File
[Powerpoint](https://docs.google.com/presentation/d/1HhX2ESY866Q_5Yu7D9NPph2QsxaGM4TNkazZ3F9_PHQ/edit?usp=sharing)

### YouTube Presentation Video
[Youtube]()

---

# 2. Background

## What is the project about?

LUNARA is a Feature Lifecycle Intelligence Dashboard designed to help product and engineering teams make data-driven decisions about product features.

Modern software products contain many features, but teams often struggle to determine:
- Which features should receive more investment
- Which features should be maintained
- Which features require refactoring
- Which features should be sunset or retired

This project combines:
- Product analytics
- Operational issue tracking
- Defect risk analysis
- Machine learning forecasting
- Retrieval-Augmented Generation (RAG)

to build a unified feature intelligence platform.

The system evaluates feature performance using adoption, engagement, issue burden, and defect risk signals to generate actionable lifecycle recommendations.

---

## Why does it matter?

Product organizations frequently make feature decisions using intuition instead of evidence.

This creates several problems:
- Engineering resources are wasted on low-value features
- Technical debt grows unnoticed
- Teams lack visibility into feature health
- Product prioritization becomes inconsistent

LUNARA addresses these issues by creating a centralized intelligence framework that transforms raw operational data into clear product strategy recommendations.

The platform helps:
- Product managers
- Engineering teams
- Business stakeholders
- Analysts

understand feature performance and make informed decisions backed by data.

---

## Research Questions

1. Which product features provide the highest business value?
2. Which features are showing declining adoption or engagement?
3. How do operational issues and defect risk affect feature lifecycle decisions?
4. Can machine learning forecast future feature adoption trends?
5. Can multiple feature signals be combined into a unified lifecycle intelligence framework?

---

# 3. Data

## Data Sources

This project integrates three primary data sources to evaluate feature health and lifecycle performance.

---

## Google Analytics Sample Data

Used to capture:
- Weekly active users
- Pageviews
- Session duration
- Bounce rate
- User engagement metrics

---

## GitHub Issues Data

Used to measure:
- Operational burden
- Bug counts
- Maintenance overhead
- Feature-level issue tracking

---

## NASA KC1 Defect Dataset

Used to estimate:
- Defect risk
- Software reliability
- Engineering quality indicators

---

## Dataset Characteristics

The project combines multiple datasets into a unified feature-level analytical framework.

---

## Final Dataset Structure

Each row in the final dataset represents:
- A specific feature
- During a specific week
- Along with associated engagement and operational metrics

---

## Engineered Metrics

The final dataset includes engineered metrics such as:
- weekly_active_users
- avg_pageviews
- avg_session_duration
- bounce_rate
- issue_count
- total_revenue
- adoption_trend
- engagement_intensity
- issue_burden
- defect_risk
- lifecycle_label
- value_score

---

## Lifecycle Categories

Features are classified into:
- Invest
- Maintain
- Refactor
- Sunset

based on their overall composite value score.

---

# 4. Exploratory Data Analysis (EDA)

## Data Exploration

Exploratory Data Analysis was performed using Jupyter Notebook and Plotly visualizations to understand:
- Feature adoption patterns
- Engagement trends
- Revenue contribution
- Operational burden
- Defect risk behavior

The analysis focused on identifying relationships between product value and engineering health.

---

## Summary Insights

EDA revealed several important trends:
- Features with higher engagement often generated higher revenue
- Features with high issue burden showed declining lifecycle scores
- Some features demonstrated strong adoption but poor operational health
- Bounce rate negatively correlated with feature value

These findings validated the lifecycle scoring framework.

---

## Visualizations Created

Several visualizations were created, including:
- Time-series trend charts
- Heatmaps
- Correlation matrices
- Revenue analysis charts
- Radar charts
- Box plots
- Lifecycle distribution charts

These visualizations helped identify hidden patterns and compare feature performance across the portfolio.

---

## Data Cleaning

### Missing Values
- Missing values were identified and handled appropriately
- Numerical fields were standardized before scoring

### Duplicate Handling
- Duplicate entries were checked and removed where necessary

---

## Data Transformation

The preprocessing pipeline included:
- Weekly aggregation
- Feature normalization
- Metric standardization
- Composite score generation
- Lifecycle classification

---

## Key Findings from EDA

The analysis showed:
- High-performing features consistently demonstrated strong engagement and lower issue burden
- Features with elevated defect risk often required refactoring
- Revenue generation was strongly tied to sustained user engagement
- Lifecycle classification aligned well with observed operational behavior

---

# 5. Model Training

## Machine Learning Approach

This project uses a Gradient Boosting Regressor model to forecast future weekly active users for each feature.

The forecasting system helps identify:
- Growth trends
- Adoption decline
- Seasonal patterns
- Feature engagement trajectories

---

## Why Gradient Boosting?

Gradient Boosting was selected because:
- It performs well on structured tabular data
- It captures nonlinear relationships
- It handles mixed feature types effectively
- It provides strong predictive performance with relatively low complexity

---

## Features Used for Forecasting

The forecasting model uses:
- Historical weekly active users
- Rolling averages
- Lag features
- Engagement metrics
- Operational signals

to predict future adoption behavior.

---

## Lifecycle Scoring System

A composite lifecycle value score is calculated using:
- Adoption Trend
- Engagement Intensity
- Issue Burden
- Defect Risk

Positive signals increase the score, while operational risk signals reduce it.

---

## Lifecycle Classification Logic

Features are categorized into:
- Invest
- Maintain
- Refactor
- Sunset

based on predefined threshold ranges applied to the final value score.

---

## Python Libraries Used

- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit

---

## Development Environment

The project was developed using:
- Jupyter Notebook
- Local Python Environment
- Streamlit Dashboard Framework

---

## Model Evaluation

The forecasting system was evaluated using:
- RMSE
- MAPE
- Trend comparison
- Actual vs predicted weekly active users

Lifecycle recommendations were validated through:
- Consistency checks
- Feature comparisons
- Operational signal analysis

---

# 6. Application of the Trained Models

## Streamlit Dashboard

An interactive Streamlit dashboard was developed to serve as the primary interface for lifecycle intelligence analysis.

The dashboard integrates:
- Analytics
- Forecasting
- Lifecycle scoring
- Decision intelligence
- Interactive visualizations

into a unified platform.

---

## Dashboard Modules

### Portfolio Overview
Provides:
- Portfolio-level KPIs
- Lifecycle distribution
- Revenue analysis
- Feature status monitoring

---

### Adoption Forecasting
Displays:
- Actual vs predicted weekly active users
- Feature-level growth trends
- Forecast confidence insights

---

### Feature Deep Dive
Allows users to:
- Analyze individual feature health
- Explore engagement trends
- Compare against portfolio averages

---

### Decision Reports
Generates:
- Evidence-backed recommendations
- Risk summaries
- Suggested next steps
- RAG-powered decision intelligence

---

## Technologies Used

- Streamlit
- Plotly
- Scikit-learn
- Pandas
- NumPy

---

## Purpose of the Dashboard

The dashboard transforms complex feature analytics into:
- Actionable business insights
- Engineering prioritization guidance
- Product strategy recommendations
- Interactive lifecycle intelligence

---

# 7. Conclusion

## Summary of Work

This project developed a complete Feature Lifecycle Intelligence platform called LUNARA.

The project:
- Integrated multiple data sources
- Engineered feature health metrics
- Built a lifecycle scoring framework
- Trained forecasting models
- Developed an interactive Streamlit dashboard
- Generated evidence-based recommendations

The final system enables organizations to make data-driven product decisions instead of relying on intuition.

---

## Practical Applications

The platform can support:
- Product management
- Engineering prioritization
- Feature investment decisions
- Technical debt analysis
- Lifecycle monitoring
- Product strategy planning

---

## Limitations

Several limitations exist:
- Some datasets were simulated/sample datasets
- Forecasting performance depends on historical data quality
- Lifecycle thresholds are manually defined
- Additional behavioral signals could improve accuracy

---

## Lessons Learned

This project provided practical experience in:
- Data preprocessing
- Feature engineering
- Machine learning forecasting
- Dashboard development
- Data visualization
- Product analytics
- Retrieval-Augmented Generation systems

---

## Future Improvements

Future enhancements may include:
- Real-time streaming analytics
- Cloud deployment
- User authentication
- Advanced time-series forecasting models
- Explainable AI components
- Enterprise-scale feature monitoring
- Integration with production analytics platforms

---

# 8. References

1. Streamlit Documentation  
https://docs.streamlit.io/

2. Plotly Documentation  
https://plotly.com/python/

3. Scikit-learn Documentation  
https://scikit-learn.org/

4. Pandas Documentation  
https://pandas.pydata.org/

5. NumPy Documentation  
https://numpy.org/

6. Jupyter Notebook Documentation  
https://jupyter.org/

7. NASA KC1 Dataset  
https://www.openml.org/

8. Google Analytics Documentation  
https://support.google.com/analytics/

9. GitHub Issues Documentation  
https://docs.github.com/en/issues
