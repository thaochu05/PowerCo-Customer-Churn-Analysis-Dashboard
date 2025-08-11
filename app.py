import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="BCG - PowerCo Churn Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for power company theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #3b82f6;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f59e0b;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #10b981;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    }
</style>
""", unsafe_allow_html=True)

# Color palette for power company theme
COLORS = {
    'primary': '#1e3a8a',      # Dark blue
    'secondary': '#3b82f6',    # Blue
    'accent': '#f59e0b',       # Orange
    'success': '#10b981',      # Green
    'warning': '#f59e0b',      # Orange
    'danger': '#ef4444',       # Red
    'light': '#f8fafc',        # Light gray
    'dark': '#1e293b'          # Dark gray
}

@st.cache_data
def load_data():
    """Load and cache the data"""
    try:
        df = pd.read_csv('./data_for_predictions.csv')
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data_for_predictions.csv' is in the same directory.")
        return None

@st.cache_data
def load_clean_data():
    """Load cleaned data for exploration"""
    try:
        return pd.read_csv('./clean_data_after_eda.csv')
    except FileNotFoundError:
        return None

@st.cache_data
def load_client_data():
    """Load client data"""
    try:
        return pd.read_csv('./client_data.csv')
    except FileNotFoundError:
        return None

def create_confusion_matrix_plot(y_true, y_pred):
    """Create an interactive confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No Churn', 'Predicted Churn'],
        y=['Actual No Churn', 'Actual Churn'],
        colorscale=[[0, COLORS['light']], [1, COLORS['primary']]],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=500,
        height=400
    )
    
    return fig

def create_feature_importance_plot(feature_importances):
    """Create feature importance plot"""
    # Get top 15 features
    top_features = feature_importances.head(15)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='features',
        orientation='h',
        title="Top 15 Feature Importances",
        color='importance',
        color_continuous_scale=[COLORS['light'], COLORS['primary']]
    )
    
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=600,
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>BCG Power Company - Customer Churn Analysis</h1>
        <p>Executive Report | Data-Driven Insights for Strategic Decision Making</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.markdown("## 📊 Report Sections")
    page = st.sidebar.selectbox(
        "Select section:",
        ["📋 Executive Summary", "📈 Model Performance", "🔍 Feature Analysis", "👥 Customer Insights", "🔮 Prediction Tool", "📋 Data Explorer"]
    )
    
    # Separate target variable from features
    y = df['churn']
    X = df.drop(columns=['id', 'churn'])
    
    # Train model (cached)
    @st.cache_resource
    def train_model():
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = RandomForestClassifier(n_estimators=1000, random_state=42)
        model.fit(X_train, y_train)
        return model, X_test, y_test
    
    model, X_test, y_test = train_model()
    
    # Page routing
    if page == "📋 Executive Summary":
        show_executive_summary(df, model, X_test, y_test)
    elif page == "📈 Model Performance":
        show_model_performance(model, X_test, y_test)
    elif page == "🔍 Feature Analysis":
        show_feature_analysis(model, X)
    elif page == "👥 Customer Insights":
        show_customer_insights(df)
    elif page == "🔮 Prediction Tool":
        show_prediction_tool(model, X.columns)
    elif page == "📋 Data Explorer":
        show_data_explorer()

def show_executive_summary(df, model, X_test, y_test):
    """Executive Summary Page"""
    st.markdown("## 📋 EXECUTIVE SUMMARY")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Customers</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        churn_rate = (df['churn'].sum() / len(df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ Churn Rate</h3>
            <h2>{churn_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Model Accuracy</h3>
            <h2>{accuracy:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        precision = precision_score(y_test, predictions) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Precision</h3>
            <h2>{precision:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Executive Summary - Key Finding
    st.markdown("### 📋 EXECUTIVE SUMMARY")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 2rem; border-radius: 10px; color: white; margin: 2rem 0; border-left: 6px solid #f59e0b;">
        <h2 style="color: white; margin-bottom: 1rem; font-size: 1.8rem;">KEY FINDING</h2>
        <h3 style="color: white; margin-bottom: 1rem; font-size: 1.4rem;">Primary Research Question: "Is customer churn driven by price sensitivity?"</h3>
        <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="color: white; margin-bottom: 0.5rem; font-size: 1.2rem;">CONCLUSION</h4>
            <p style="font-size: 1.1rem; margin-bottom: 0.5rem;"><strong>Price sensitivity is NOT the primary driver of customer churn.</strong></p>
            <p style="font-size: 1rem; opacity: 0.9;">While price sensitivity contributes to churn behavior, our analysis identifies more significant factors that require strategic attention.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key insights
    st.markdown("### 📊 ANALYSIS FINDINGS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
            <h4>Primary Churn Drivers</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li style="margin-bottom: 0.5rem;"><strong>1. Net Margin</strong> - Most influential factor in churn prediction</li>
                <li style="margin-bottom: 0.5rem;"><strong>2. Consumption Patterns</strong> - 12-month usage trends are key indicators</li>
                <li style="margin-bottom: 0.5rem;"><strong>3. Customer Tenure</strong> - Relationship duration significantly impacts retention</li>
                <li style="margin-bottom: 0.5rem;"><strong>4. Contract Engagement</strong> - Time since last contract update is important</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>Model Performance Assessment</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li style="margin-bottom: 0.5rem;"><strong>Accuracy:</strong> 90.4% - Strong performance with improvement potential</li>
                <li style="margin-bottom: 0.5rem;"><strong>Recall:</strong> Limited - Model struggles to identify actual churners</li>
                <li style="margin-bottom: 0.5rem;"><strong>Data Balance:</strong> Imbalanced - Churn rate represents ~10% of customer base</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Business recommendations
    st.markdown("### 💼 STRATEGIC RECOMMENDATIONS")
    
    st.markdown("""
    <div class="success-box">
        <h4>Priority Actions</h4>
        <ol style="padding-left: 1.5rem;">
            <li style="margin-bottom: 0.8rem;"><strong>High-Value Customer Retention</strong> - Prioritize retention efforts for customers with high net margins</li>
            <li style="margin-bottom: 0.8rem;"><strong>Consumption Pattern Monitoring</strong> - Implement systems to track 12-month usage trends for early risk identification</li>
            <li style="margin-bottom: 0.8rem;"><strong>Proactive Contract Management</strong> - Develop engagement strategies before contract renewal periods</li>
            <li style="margin-bottom: 0.8rem;"><strong>Tenure-Based Loyalty Programs</strong> - Design retention initiatives for long-term customer relationships</li>
            <li style="margin-bottom: 0.8rem;"><strong>Margin Optimization Strategy</strong> - Focus on improving net margins through operational efficiency rather than price reductions</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <h4>Strategic Considerations</h4>
        <p style="margin-bottom: 1rem;"><strong>Price sensitivity, while present, is not the primary driver of churn.</strong> Strategic focus should prioritize:</p>
        <ul style="list-style-type: none; padding-left: 0;">
            <li style="margin-bottom: 0.5rem;">• Enhanced customer relationship management</li>
            <li style="margin-bottom: 0.5rem;">• Advanced usage pattern analytics</li>
            <li style="margin-bottom: 0.5rem;">• Margin optimization initiatives</li>
            <li style="margin-bottom: 0.5rem;">• Proactive customer engagement programs</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def show_model_performance(model, X_test, y_test):
    """Model Performance Page"""
    st.markdown("## 📈 Model Performance")
    
    # Get predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = accuracy_score(y_test, predictions) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")
    
    with col2:
        precision = precision_score(y_test, predictions) * 100
        st.metric("Precision", f"{precision:.1f}%")
    
    with col3:
        recall = recall_score(y_test, predictions) * 100
        st.metric("Recall", f"{recall:.1f}%")
    
    with col4:
        f1 = 2 * (precision * recall) / (precision + recall)
        st.metric("F1-Score", f"{f1:.1f}%")
    
    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm_fig = create_confusion_matrix_plot(y_test, predictions)
    st.plotly_chart(cm_fig, use_container_width=True)
    
    # Detailed analysis
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Detailed Metrics")
        st.markdown(f"""
        - **True Positives (TP):** {tp} - Correctly identified churners
        - **False Positives (FP):** {fp} - Incorrectly flagged as churners
        - **True Negatives (TN):** {tn} - Correctly identified non-churners
        - **False Negatives (FN):** {fn} - Missed churners
        """)
    
    with col2:
        st.markdown("### 🎯 Performance Analysis")
        st.markdown(f"""
        - **Sensitivity (Recall):** {recall:.1f}% - Ability to find actual churners
        - **Specificity:** {(tn/(tn+fp)*100):.1f}% - Ability to identify non-churners
        - **Positive Predictive Value:** {precision:.1f}% - Accuracy of positive predictions
        - **Negative Predictive Value:** {(tn/(tn+fn)*100):.1f}% - Accuracy of negative predictions
        """)
    
    # Probability distribution
    st.markdown("### 📈 Prediction Probability Distribution")
    
    fig = px.histogram(
        x=probabilities,
        nbins=50,
        title="Distribution of Churn Probabilities",
        labels={'x': 'Churn Probability', 'y': 'Count'}
    )
    fig.update_layout(
        xaxis_title="Churn Probability",
        yaxis_title="Number of Customers"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_feature_analysis(model, X):
    """Feature Analysis Page"""
    st.markdown("## 🔍 Feature Analysis")
    
    # Get feature importances
    feature_importances = pd.DataFrame({
        'features': X.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=True).reset_index(drop=True)
    
    # Feature importance plot
    st.markdown("### 📊 Feature Importance Ranking")
    fig = create_feature_importance_plot(feature_importances)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top features analysis
    st.markdown("### 🏆 Top 10 Most Important Features")
    
    top_10 = feature_importances.tail(10)
    
    for idx, row in top_10.iterrows():
        importance_pct = (row['importance'] / feature_importances['importance'].sum()) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h4>{row['features']}</h4>
            <p><strong>Importance:</strong> {row['importance']:.4f} ({importance_pct:.1f}% of total)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature categories
    st.markdown("### 📋 Feature Categories Analysis")
    
    # Categorize features
    financial_features = [col for col in X.columns if any(word in col.lower() for word in ['margin', 'price', 'cost', 'revenue'])]
    consumption_features = [col for col in X.columns if any(word in col.lower() for word in ['consumption', 'usage', 'kwh'])]
    temporal_features = [col for col in X.columns if any(word in col.lower() for word in ['month', 'year', 'tenure', 'duration'])]
    contract_features = [col for col in X.columns if any(word in col.lower() for word in ['contract', 'agreement'])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💰 Financial Features")
        financial_importance = feature_importances[feature_importances['features'].isin(financial_features)]['importance'].sum()
        st.metric("Total Importance", f"{financial_importance:.4f}")
        
        st.markdown("#### ⚡ Consumption Features")
        consumption_importance = feature_importances[feature_importances['features'].isin(consumption_features)]['importance'].sum()
        st.metric("Total Importance", f"{consumption_importance:.4f}")
    
    with col2:
        st.markdown("#### ⏰ Temporal Features")
        temporal_importance = feature_importances[feature_importances['features'].isin(temporal_features)]['importance'].sum()
        st.metric("Total Importance", f"{temporal_importance:.4f}")
        
        st.markdown("#### 📄 Contract Features")
        contract_importance = feature_importances[feature_importances['features'].isin(contract_features)]['importance'].sum()
        st.metric("Total Importance", f"{contract_importance:.4f}")

def show_customer_insights(df):
    """Customer Insights Page"""
    st.markdown("## 👥 Customer Insights")
    
    # Load additional data
    clean_data = load_clean_data()
    client_data = load_client_data()
    
    # Churn distribution
    st.markdown("### 📊 Customer Churn Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        churn_counts = df['churn'].value_counts()
        fig = px.pie(
            values=churn_counts.values,
            names=['No Churn', 'Churn'],
            title="Churn vs No Churn Distribution",
            color_discrete_sequence=[COLORS['success'], COLORS['danger']]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        churn_rate = (df['churn'].sum() / len(df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>Overall Churn Rate</h3>
            <h2>{churn_rate:.1f}%</h2>
            <p>Out of {len(df):,} total customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key insights
    st.markdown("### 🔍 Key Customer Insights")
    
    st.markdown("""
    <div class="insight-box">
        <h4>Customer Behavior Insights</h4>
        <ul style="list-style-type: none; padding-left: 0;">
            <li style="margin-bottom: 0.5rem;"><strong>Financial Health:</strong> Net margin is the strongest predictor of churn</li>
            <li style="margin-bottom: 0.5rem;"><strong>Usage Patterns:</strong> 12-month consumption history reveals churn risk</li>
            <li style="margin-bottom: 0.5rem;"><strong>Relationship Duration:</strong> Longer customer relationships show lower churn risk</li>
            <li style="margin-bottom: 0.5rem;"><strong>Contract Engagement:</strong> Recent contract updates indicate higher retention</li>
            <li style="margin-bottom: 0.5rem;"><strong>Price Sensitivity:</strong> <span style="color: #dc2626; font-weight: bold;">Not the primary driver</span> - analysis contradicts initial hypothesis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("### 💼 RETENTION STRATEGIES")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>High-Value Customer Focus</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li style="margin-bottom: 0.5rem;">• Monitor net margin trends</li>
                <li style="margin-bottom: 0.5rem;">• Proactive margin optimization</li>
                <li style="margin-bottom: 0.5rem;">• Premium service offerings</li>
                <li style="margin-bottom: 0.5rem;">• Dedicated account management</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>At-Risk Customer Identification</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li style="margin-bottom: 0.5rem;">• Track consumption declines</li>
                <li style="margin-bottom: 0.5rem;">• Monitor contract renewal dates</li>
                <li style="margin-bottom: 0.5rem;">• Identify price sensitivity signals</li>
                <li style="margin-bottom: 0.5rem;">• Engage before churn events</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_prediction_tool(model, feature_names):
    """Prediction Tool Page"""
    st.markdown("## 🔮 Churn Prediction Tool")
    
    st.markdown("### 📝 Enter Customer Information")
    st.markdown("Use this tool to predict the churn probability for a specific customer.")
    
    # Create input form
    with st.form("prediction_form"):
        st.markdown("#### Customer Financial Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            net_margin = st.number_input("Net Margin", value=0.0, help="Customer's net margin")
            margin_power_subscription = st.number_input("Margin on Power Subscription", value=0.0)
            price_sensitivity = st.number_input("Price Sensitivity Score", value=0.0, min_value=0.0, max_value=10.0)
        
        with col2:
            consumption_12m = st.number_input("12-Month Consumption (kWh)", value=0.0)
            monthly_consumption = st.number_input("Monthly Consumption (kWh)", value=0.0)
            avg_monthly_bill = st.number_input("Average Monthly Bill", value=0.0)
        
        st.markdown("#### Customer Relationship Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tenure_months = st.number_input("Tenure (Months)", value=0, min_value=0)
            months_active = st.number_input("Months Active", value=0, min_value=0)
            months_since_contract_update = st.number_input("Months Since Contract Update", value=0, min_value=0)
        
        with col2:
            contract_type = st.selectbox("Contract Type", ["Standard", "Premium", "Basic"])
            payment_method = st.selectbox("Payment Method", ["Direct Debit", "Credit Card", "Bank Transfer", "Other"])
        
        submitted = st.form_submit_button("🔮 Predict Churn Probability")
    
    if submitted:
        # Create feature vector (simplified - you'd need to match your actual feature engineering)
        # This is a simplified version - you'd need to create the exact same features as in your model
        
        st.markdown("### ⚠️ Note")
        st.markdown("""
        <div class="warning-box">
            <p><strong>Important:</strong> This is a simplified prediction tool. The actual model requires 61 engineered features 
            that match the exact feature engineering process used during training. To get accurate predictions, 
            you would need to implement the complete feature engineering pipeline.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Placeholder prediction
        st.markdown("### 📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", "N/A", help="Requires complete feature engineering")
        
        with col2:
            st.metric("Risk Level", "N/A", help="Based on probability threshold")
        
        with col3:
            st.metric("Recommendation", "N/A", help="Suggested action based on prediction")
        
        st.markdown("### 🔧 To Implement Full Prediction")
        st.markdown("""
        To make this tool fully functional, you would need to:
        1. Implement the complete feature engineering pipeline from your notebooks
        2. Create all 61 features used in the model
        3. Apply the same preprocessing steps
        4. Use the trained model to make predictions
        """)

def show_data_explorer():
    """Data Explorer Page"""
    st.markdown("## 📋 Data Explorer")
    
    # Load different datasets
    df = load_data()
    clean_data = load_clean_data()
    client_data = load_client_data()
    
    # Dataset selector
    dataset_choice = st.selectbox(
        "Choose a dataset to explore:",
        ["Modeling Dataset (61 features)", "Cleaned Data", "Client Data"]
    )
    
    if dataset_choice == "Modeling Dataset (61 features)" and df is not None:
        st.markdown("### 📊 Modeling Dataset Overview")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Columns:** {len(df.columns)}")
        
        # Show first few rows
        st.markdown("#### First 10 Rows")
        st.dataframe(df.head(10))
        
        # Show data info
        st.markdown("#### Dataset Information")
        st.write(df.info())
        
        # Show statistics
        st.markdown("#### Statistical Summary")
        st.dataframe(df.describe())
        
        # Show missing values
        st.markdown("#### Missing Values")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.write(missing_data[missing_data > 0])
        else:
            st.write("No missing values found!")
    
    elif dataset_choice == "Cleaned Data" and clean_data is not None:
        st.markdown("### 📊 Cleaned Data Overview")
        st.write(f"**Shape:** {clean_data.shape}")
        st.write(f"**Columns:** {len(clean_data.columns)}")
        
        st.markdown("#### First 10 Rows")
        st.dataframe(clean_data.head(10))
        
        st.markdown("#### Dataset Information")
        st.write(clean_data.info())
    
    elif dataset_choice == "Client Data" and client_data is not None:
        st.markdown("### 📊 Client Data Overview")
        st.write(f"**Shape:** {client_data.shape}")
        st.write(f"**Columns:** {len(client_data.columns)}")
        
        st.markdown("#### First 10 Rows")
        st.dataframe(client_data.head(10))
        
        st.markdown("#### Dataset Information")
        st.write(client_data.info())
    
    else:
        st.warning("Selected dataset not available. Please ensure all data files are present.")

if __name__ == "__main__":
    main()
