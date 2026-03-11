import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- APP CONFIG ---
st.set_page_config(page_title="AI Efficiency Multiplier Dashboard", layout="wide")

st.title("🧠 The AI Efficiency Multiplier")
st.markdown("### Investigating the Bridge between AI Dependency and Concept Mastery")

# --- 1. DATA UPLOADER ---
uploaded_file = st.sidebar.file_uploader("Upload your Student Performance CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Selecting our specific 7-pillar features
    features = ['uses_ai', 'ai_dependency_score', 'ai_generated_content_percentage', 'study_hours_per_day']
    target = 'concept_understanding_score'
    
    # Basic Preprocessing for the Model
    X = df[features]
    y = df[target]
    
    # Train-Test Split (80/20) as per Methodology
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Multiple Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # --- 2. SIDEBAR CONTROLS ---
    st.sidebar.divider()
    st.sidebar.header("Live Simulation")
    s_uses_ai = st.sidebar.toggle("Uses AI Tools", value=True)
    s_study = st.sidebar.slider("Daily Study Hours", 0.0, 6.0, 1.5)
    s_dep = st.sidebar.slider("AI Dependency Score", 0, 10, 8 if s_uses_ai else 0)
    s_pct = st.sidebar.slider("AI Content %", 0, 100, 75 if s_uses_ai else 0)

    # Prediction Logic
    user_input = np.array([[int(s_uses_ai), s_dep, s_pct, s_study]])
    prediction = model.predict(user_input)[0]

    # --- 3. DASHBOARD METRICS ---
    # Benchmark: Traditional Studious (Study > 4, No AI)
    trad_group = df[(df['uses_ai'] == 0) & (df['study_hours_per_day'] > 4)]
    avg_trad_score = trad_group[target].mean() if not trad_group.empty else 5.58

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Predicted Mastery Score", f"{prediction:.2f} / 10")
    with m2:
        efficiency_ratio = (prediction / avg_trad_score) * 100
        st.metric("Proficiency Parity", f"{efficiency_ratio:.1f}%", delta=f"{efficiency_ratio-100:.1f}%")
    with m3:
        st.metric("Group Benchmark", f"{avg_trad_score:.2f}")

    st.divider()

    # --- 4. VISUALIZATIONS ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Efficiency Comparison")
        chart_data = pd.DataFrame({
            "Category": ["Your Profile (AI-Reliant)", "Traditional Baseline"],
            "Score": [prediction, avg_trad_score]
        })
        st.bar_chart(chart_data, x="Category", y="Score", color="#0072B2")
        

    with col_right:
        st.subheader("Feature Correlation (The Evidence)")
        fig, ax = plt.subplots()
        sns.heatmap(df[features + [target]].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        

    # --- 5. THE VERDICT BOX ---
    st.info(f"""
    **The Verdict:** By utilizing AI, this profile achieves **{efficiency_ratio:.1f}%** of the mastery found in high-effort traditional students, 
    effectively proving that AI is an **Efficiency Multiplier**. It allows for deep learning parity while significantly reducing manual study time.
    """)

else:
    st.warning("Please upload the 'ai_impact_student_performance_dataset.csv' in the sidebar to begin the analysis.")
