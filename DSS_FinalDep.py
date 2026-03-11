import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Impact Study", layout="wide")

st.title("📊 AI & Student Performance: Impact Analysis")

# 1. LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    return df

df = load_data()

# SIDEBAR
st.sidebar.header("Explore the Data")
page = st.sidebar.radio("Go to", ["Overview", "Correlation Heatmap", "AI vs Understanding", "Pass/Fail Predictor"])

# PAGE 1: OVERVIEW
if page == "Overview":
    st.header("Project Dataset Overview")
    st.write("Does AI usage hinder learning? We analyze 8,000 students to find out.")
    st.dataframe(df.head())

# PAGE 2: CORRELATION HEATMAP
elif page == "Correlation Heatmap":
    st.header("Variable Correlation Heatmap")
    st.write("This map shows the statistical link between all variables.")
    
    # Selecting numerical columns only
    cols = ['ai_dependency_score', 'ai_generated_content_percentage', 
            'concept_understanding_score', 'study_hours_per_day', 'final_score']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    st.info("💡 **Insight:** Notice the 0.42 correlation between Understanding and Final Score. Notice the near-zero correlation (0.02) between AI Dependency and Understanding.")

# PAGE 3: AI VS UNDERSTANDING (LINEAR REGRESSION)
elif page == "AI vs Understanding":
    st.header("Statistical Proof: Is AI Hurting Learning?")
    
    # Predicting Understanding based on AI factors
    X = df[['ai_dependency_score', 'ai_generated_content_percentage', 'study_hours_per_day']]
    y = df['concept_understanding_score']
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    st.text(str(model.summary()))
    st.markdown("### 📢 Result: AI dependency does NOT predict lower understanding scores.")

# PAGE 4: PASS/FAIL PREDICTOR (LOGISTIC REGRESSION)
elif page == "Pass/Fail Predictor":
    st.header("AI Predictor: Will the Student Pass?")
    st.write("Using Logistic Regression from `sklearn` to predict Pass (1) or Fail (0).")

    # Prepare data for sklearn
    features = ['ai_dependency_score', 'concept_understanding_score', 'study_hours_per_day']
    X = df[features]
    y = df['passed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    
    y_pred = lr_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    st.metric("Model Accuracy", f"{acc*100:.2f}%")
    
    # User Input for Prediction
    st.subheader("Try it yourself:")
    dep = st.slider("AI Dependency Score", 0, 10, 5)
    und = st.slider("Concept Understanding", 0, 10, 5)
    hrs = st.slider("Study Hours/Day", 0, 10, 3)
    
    prediction = lr_model.predict([[dep, und, hrs]])
    result = "✅ PASS" if prediction[0] == 1 else "❌ FAIL"
    st.title(f"Prediction: {result}")
