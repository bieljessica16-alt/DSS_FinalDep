import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. PAGE SETUP
st.set_page_config(page_title="AI & Student Success Predictor", layout="wide")
st.title("🤖 AI & Student Performance: Machine Learning Dashboard")

# 2. LOAD DATA
@st.cache_data
def load_data():
    # Make sure this filename matches your CSV exactly
    df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    return df

df = load_data()

# 3. SIDEBAR NAVIGATION
page = st.sidebar.radio("Go to", ["Project Overview", "Correlation Heatmap", "Success Predictor (ML)"])

if page == "Project Overview":
    st.header("📋 Project Overview")
    st.write("Does AI usage impact a student's ability to pass? This study analyzes 8,000 students.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Final Score", f"{round(df['final_score'].mean(), 1)}%")
    col2.metric("Pass Rate", f"{round(df['passed'].mean() * 100, 1)}%")
    col3.metric("AI Users", len(df[df['uses_ai'] == 1]))
    
    st.dataframe(df.head(10))

elif page == "Correlation Heatmap":
    st.header("🔥 Correlation Heatmap")
    st.write("This map identifies the strongest links between variables.")
    
    # Select only numeric columns for the heatmap
    numeric_df = df.select_dtypes(include=['number'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    st.info("**Key Finding:** Understanding Score is the #1 driver of grades (0.42). AI usage metrics have near-zero correlation with grades, meaning they don't harm performance.")

elif page == "Success Predictor (ML)":
    st.header("🧠 Predictor: Will the student PASS?")
    st.write("Using Logistic Regression to predict success based on behavior.")
    
    # MODEL TRAINING
    # Features: What we use to predict
    features = ['ai_dependency_score', 'ai_generated_content_percentage', 'study_hours_per_day', 'concept_understanding_score']
    X = df[features]
    y = df['passed']  # Target: 1 (Pass) or 0 (Fail)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model Training Complete! Accuracy: **{round(acc * 100, 2)}%**")
    
    st.divider()
    
    # LIVE PREDICTION TOOL
    st.subheader("🔮 Try the Predictor")
    st.write("Enter data to see if a hypothetical student would pass:")
    
    col_a, col_b = st.columns(2)
    with col_a:
        u_score = st.slider("Concept Understanding Score (1-10)", 1, 10, 7)
        hrs = st.slider("Study Hours Per Day", 0.0, 12.0, 4.0)
    with col_b:
        ai_dep = st.slider("AI Dependency (1-10)", 1, 10, 3)
        ai_perc = st.slider("AI Generated Content %", 0, 100, 15)
        
    # Make prediction
    input_data = [[ai_dep, ai_perc, hrs, u_score]]
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]
    
    if prediction[0] == 1:
        st.success(f"Prediction: **PASS** (Probability: {round(prob*100, 1)}%)")
    else:
        st.error(f"Prediction: **FAIL** (Probability: {round((1-prob)*100, 1)}%)")
