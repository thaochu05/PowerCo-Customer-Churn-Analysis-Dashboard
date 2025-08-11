# ⚡ BCG Power Company Churn Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing customer churn in the power company industry, built with a professional power company color theme.
link at https://powerco-customer-churn-model-analysis.streamlit.app/

## 🚀 Features

- **Executive Summary**: Key metrics and business insights
- **Model Performance**: Detailed model evaluation with confusion matrix
- **Feature Analysis**: Interactive feature importance visualization
- **Customer Insights**: Churn distribution and behavioral analysis
- **Prediction Tool**: Interactive churn prediction interface
- **Data Explorer**: Interactive data exploration tools

## 📋 Prerequisites

- Python 3.8 or higher
- All data files from your BCG analysis:
  - `data_for_predictions.csv`
  - `clean_data_after_eda.csv`
  - `client_data.csv`

## 🛠️ Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your data files are in the same directory as `app.py`**

## 🚀 Running the Dashboard

1. **Navigate to the project directory**

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and go to the URL shown in the terminal (usually `http://localhost:8501`)

## 📊 Dashboard Sections

### 🏠 Executive Summary
- Total customer count and churn rate
- Model accuracy and precision metrics
- Key findings about churn drivers
- Business recommendations

### 📈 Model Performance
- Accuracy, precision, recall, and F1-score
- Interactive confusion matrix
- Prediction probability distribution
- Detailed performance analysis

### 🔍 Feature Analysis
- Top 15 feature importance ranking
- Feature categories analysis (financial, consumption, temporal, contract)
- Interactive feature importance visualization

### 👥 Customer Insights
- Churn vs no-churn distribution
- Key customer behavior insights
- Customer retention strategies

### 🔮 Prediction Tool
- Interactive form for customer data input
- Churn probability prediction (requires full feature engineering implementation)
- Risk level assessment

### 📋 Data Explorer
- Interactive exploration of all datasets
- Statistical summaries
- Missing value analysis


## 📈 Key Insights from the Analysis

Based on the BCG analysis, the dashboard highlights:

1. **Primary Churn Drivers**:
   - Net margin (most influential)
   - 12-month consumption patterns
   - Customer tenure
   - Contract update timing

2. **Model Performance**:
   - High accuracy but low recall
   - Good at identifying non-churners
   - Struggles with identifying actual churners

3. **Business Recommendations**:
   - Focus on high-value customers
   - Monitor consumption patterns
   - Proactive contract management
   - Tenure-based loyalty programs

## 🚀 Deployment

### Local Deployment
The dashboard runs locally using Streamlit's development server.

### Cloud Deployment
You can deploy to Streamlit Cloud:
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Deploy automatically

## 📝 Notes

- The prediction tool requires implementation of the complete feature engineering pipeline
- All visualizations are interactive using Plotly
- Data is cached for optimal performance
- The dashboard is responsive and works on different screen sizes

