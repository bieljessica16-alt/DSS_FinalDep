import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Efficiency Study", layout="wide")

# 1. LOAD DATA
@st.cache_data
def load_data():
    # Ensure your CSV is named exactly this in your GitHub repo
    df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    return df

df = load_data()

# 2. TRAIN THE ENGINE
# Fixed: Explicit feature list to ensure order is always the same
features = ['ai_dependency_score', 'study_hours_per_day', 'ai_generated_content_percentage', 'attendance_percentage']
X = df[features]
y_concept = df['concept_understanding_score']
y_final = df['final_score']

# Training simple models for the simulator
model_concept = LinearRegression().fit(X, y_concept)
model_final = LinearRegression().fit(X, y_final)

# --- SIDEBAR: THE SIMULATOR INPUTS ---
st.sidebar.header("🕹️ Learning Style Simulator")
st.sidebar.markdown("Adjust these to see the 'AI Paradox' in action.")

hrs = st.sidebar.slider("Study Hours Per Day", 0.5, 10.0, 3.0)
attendance = st.sidebar.slider("Class Attendance (%)", 0, 100, 90)
ai_dep = st.sidebar.slider("AI Dependency (1-10)", 1, 10, 5)
ai_pct = st.sidebar.slider("AI Content Percentage", 0, 100, 30)

# --- MAIN PAGE ---
st.title("📊 AI & Student Performance: The Efficiency Multiplier")

tabs = st.tabs(["The Simulator", "Statistical Proof", "Correlation Heatmap"])

# TAB 1: THE INTERACTIVE STORY
with tabs[0]:
    st.header("The Learning Style Simulator")
    st.write("Does using AI lower your actual understanding? Test a profile below.")

    # Calculations - Order must match 'features' list exactly
    user_input = np.array([[ai_dep, hrs, ai_pct, attendance]])
    pred_understanding = model_concept.predict(user_input)[0]
    pred_final = model_final.predict(user_input)[0]
    
    # Define Pass/Fail status for the metric (Using 50% as the threshold)
    pass_status = "✅ PASS" if pred_final >= 50 else "❌ FAIL"
    
    # Traditional Benchmark (High-effort student with NO AI)
    benchmark_score = df[(df['study_hours_per_day'] >= 5) & (df['ai_dependency_score'] < 2)]['final_score'].mean()
    efficiency_gap = (pred_final / benchmark_score) * 100

    # Main Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Concept Understanding", f"{pred_understanding:.2f}/10")
    # Using delta for the Pass/Fail label
    col2.metric("Predicted Final Score", f"{pred_final:.1f}%", delta=pass_status, delta_color="normal")
    col3.metric("Proficiency Parity", f"{efficiency_gap:.1f}%")

    # --- DESCRIPTION BOXES ---
    st.markdown("### 🔍 Why are my results like this?")
    exp1, exp2, exp3 = st.columns(3)
    
    with exp1:
        st.info("**Why only ~5.5 Understanding?**\n\nThe data shows an 'Understanding Plateau.' Adding more hours only adds tiny fractions to understanding. This proves quality matters more than quantity.")
    
    with exp2:
        st.info("**What is a Passing Grade?**\n\nIn this study, **50% is a Pass.** If you see 55%, you have successfully passed! To reach 80%+, you usually need higher Attendance.")
    
    with exp3:
        st.info("**What is Proficiency Parity?**\n\n**100% means you are performing exactly as well as a 'Traditional Student'** (5+ hrs, No AI). High parity proves AI efficiency.")

    # Visual Comparison
    st.divider()
    st.markdown("### Your Profile vs. Traditional Gold Standard")
    chart_data = pd.DataFrame({
        "Category": ["Your Profile", "Traditional (High Effort, No AI)"],
        "Final Score": [pred_final, benchmark_score]
    })
    
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(data=chart_data, x="Final Score", y="Category", palette=["#2E86C1", "#ABB2B9"], ax=ax)
    ax.set_xlim(0, 100)
    st.pyplot(fig)

    # THE VERDICT
    if efficiency_gap >= 95:
        st.success(f"### ✅ Verdict: Efficient Mastery\nYou are achieving {efficiency_gap:.1
