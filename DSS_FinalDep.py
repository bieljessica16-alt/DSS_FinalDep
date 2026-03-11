import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Impact Analysis", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("ai_impact_student_performance_dataset.csv")
    # Using the columns you identified in your cleaning step
    return df[['grade_level', 'uses_ai', 'ai_dependency_score', 
               'study_hours_per_day', 'concept_understanding_score', 'final_score']]

df = load_data()

st.title("📊 AI Impact on Student Performance")
st.write("Analysis by: Busano, Cortez, Geronimo, Monses, and Perez")

# --- ANALYTICS SECTION ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("AI Dependency vs. Final Score (Seaborn)")
    fig, ax = plt.subplots()
    sns.regplot(data=df, x="ai_dependency_score", y="final_score", 
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Regression Prediction")
    # Simple Scikit-Learn Linear Regression
    X = df[['ai_dependency_score', 'study_hours_per_day']]
    y = df['final_score']
    model = LinearRegression().fit(X, y)
    
    ai_input = st.slider("Select AI Dependency Score", 1, 10, 5)
    study_input = st.slider("Select Study Hours/Day", 0, 10, 3)
    
    prediction = model.predict([[ai_input, study_input]])
    st.metric("Predicted Final Score", f"{prediction[0]:.2f}")

# --- EFFICIENCY ANALYSIS ---
st.divider()
st.subheader("Notebook Analysis: Efficiency Parity")

# Logic from your notebook's last cell
ai_efficient = df[(df['ai_dependency_score'] > 7) & (df['study_hours_per_day'] < 3)]['concept_understanding_score'].mean()
traditional_grind = df[(df['ai_dependency_score'] < 2) & (df['study_hours_per_day'] > 5)]['concept_understanding_score'].mean()

c1, c2 = st.columns(2)
c1.info(f"**High AI / Low Study Understanding:** {ai_efficient:.2f}")
c2.success(f"**Low AI / High Study Understanding:** {traditional_grind:.2f}")
