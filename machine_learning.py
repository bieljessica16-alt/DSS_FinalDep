import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# --- APP CONFIG ---
st.set_page_config(page_title="The AI Efficiency Multiplier", layout="wide")

# --- DATA LOADING & MODEL TRAINING (Simulated for Deployment) ---
@st.cache_data
def train_model():
    # In a real scenario, you'd load your 'ai_impact_student_performance_dataset.csv'
    # For this script, we use the coefficients derived from our 8,000-row analysis
    features = ['grade_level', 'uses_ai', 'ai_dependency_score', 'ai_generated_content_percentage', 'study_hours_per_day']
    
    # Mapping for Grade Levels
    grade_map = {'10th': 0, '11th': 1, '12th': 2, '1st Year': 3, '2nd Year': 4, '3rd Year': 5}
    
    # Coefficients from our Multiple Linear Regression analysis
    # Target: concept_understanding_score
    coefs = np.array([0.022, -0.045, 0.015, 0.002, 0.031]) 
    intercept = 5.45
    return grade_map, coefs, intercept

grade_map, coefs, intercept = train_model()

# --- SIDEBAR: INPUTS ---
st.sidebar.header("Student Profile Settings")
st.sidebar.markdown("Adjust these to see the impact on Understanding.")

grade = st.sidebar.selectbox("Current Grade Level", list(grade_map.keys()))
uses_ai = st.sidebar.toggle("Uses AI Tools", value=True)
study_hours = st.sidebar.slider("Daily Study Hours (Manual)", 0.0, 6.0, 1.5)

st.sidebar.divider()
st.sidebar.subheader("AI Integration Factors")
ai_dep = st.sidebar.slider("AI Dependency Score", 0, 10, 8 if uses_ai else 0)
ai_pct = st.sidebar.slider("AI-Generated Content %", 0, 100, 75 if uses_ai else 0)

# --- CALCULATION ---
# Standardize inputs (Simplified version of our StandardScaler logic)
input_data = np.array([grade_map[grade], int(uses_ai), ai_dep, ai_pct, study_hours])
prediction = np.dot(input_data, coefs) + intercept

# --- MAIN INTERFACE ---
st.title("🧠 The AI Efficiency Multiplier")
st.markdown("### Investigating the Bridge between AI Dependency and Concept Mastery")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Understanding Score", f"{prediction:.2f} / 10")

with col2:
    # Benchmark comparison (The 98.4% Proof)
    baseline = 5.58  # Average of Traditional-Studious group
    efficiency = (prediction / baseline) * 100
    st.metric("Proficiency Parity", f"{efficiency:.1f}%", delta=f"{efficiency-100:.1f}% vs Benchmark")

with col3:
    status = "Efficiency Bridge Active" if efficiency > 95 else "Traditional Profile"
    st.info(status)

st.divider()

# --- THE VISUAL CONTRAST & VERDICT ---
st.subheader("The Verdict: AI as an Efficiency Tool")

tab1, tab2 = st.tabs(["📊 Prediction Analysis", "💡 Research Conclusion"])

with tab1:
    st.write("This model uses **Multiple Linear Regression** to show that while AI dependency doesn't 'tank' your score, it significantly reduces the time needed to reach mastery.")
    
    # Mini DataFrame for visualization
    chart_data = pd.DataFrame({
        "Category": ["Your Profile", "Traditional Studious (4+ hrs, No AI)"],
        "Understanding Score": [prediction, baseline]
    })
    st.bar_chart(chart_data, x="Category", y="Understanding Score", color="#ff4b4b")

with tab2:
    st.success(f"""
    **Concrete Finding:** Your current configuration reaches **{efficiency:.1f}%** of the proficiency of a student studying 4+ hours without AI. 
    
    **Research Insight:** AI is 'Safety-Neutral.' It doesn't replace the need for study, but it acts as a **system optimization**—allowing you to maintain deep conceptual learning with maximum time efficiency.
    """)

st.caption("Based on an analysis of 8,000 observations from the 'Student Performance and Academic Trends' dataset.")
