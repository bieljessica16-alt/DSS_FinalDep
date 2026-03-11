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
    df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    return df

df = load_data()

# 2. TRAIN THE ENGINE
# We include Attendance to show the balance between AI and showing up to class
features = ['ai_dependency_score', 'study_hours_per_day', 'ai_generated_content_percentage', 'attendance_percentage']
X = df[features]
y_concept = df['concept_understanding_score']
y_final = df['final_score']

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

    # Calculations
    user_input = np.array([[ai_dep, hrs, ai_pct, attendance]])
    pred_understanding = model_concept.predict(user_input)[0]
    pred_final = model_final.predict(user_input)[0]
    
    # Traditional Benchmark (High-effort student with NO AI)
    benchmark_score = df[(df['study_hours_per_day'] >= 5) & (df['ai_dependency_score'] < 2)]['final_score'].mean()
    efficiency_gap = (pred_final / benchmark_score) * 100

    # Main Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Concept Understanding", f"{pred_understanding:.2f}/10")
    col2.metric("Predicted Final Score", f"{pred_final:.1f}%")
    col3.metric("Proficiency Parity", f"{efficiency_gap:.1f}%")

    # --- NEW: DESCRIPTION BOXES TO EXPLAIN PERCENTAGES ---
    st.markdown("### 🔍 What do these results mean?")
    exp1, exp2, exp3 = st.columns(3)
    
    with exp1:
        st.info("**Concept Understanding**\n\nThis is the student's actual 'brain power.' A score above 7.0 means they truly grasp the lesson, regardless of using AI tools.")
    
    with exp2:
        st.info("**Predicted Final Score**\n\nThis is the estimated grade on an exam. It shows that even with AI, the final score depends heavily on attendance and study hours.")
    
    with exp3:
        st.info("**Proficiency Parity**\n\nThis compares you to a 'Traditional Student.' **100% means you are performing exactly as well as someone who studies 5+ hours with NO AI.**")

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
        st.success(f"### ✅ Verdict: Efficient Mastery\nYou are achieving {efficiency_gap:.1f}% of traditional mastery. This proves you can use AI to save time without losing performance.")
    elif efficiency_gap >= 85:
        st.info("### ℹ️ Verdict: Balanced Hybrid\nYou are maintaining solid performance. AI is acting as a helpful tool rather than a replacement for learning.")
    else:
        st.warning("### ⚠️ Verdict: Risk of Learning Loss\nYour current habits suggest your understanding is dipping. Try increasing attendance or manual study hours.")

# TAB 2: STATISTICAL PROOF
with tabs[1]:
    st.header("Does AI 'Tank' Brainpower?")
    st.write("Regression analysis showing AI variables do not negatively impact understanding.")
    
    X_stats = sm.add_constant(X)
    stats_model = sm.OLS(y_concept, X_stats).fit()
    st.text(str(stats_model.summary().tables[1]))
    
    st.markdown("> **Statistical Insight:** Look at the 'P>|t|' column. Numbers higher than 0.05 mean the variable (like AI Dependency) has no significant negative effect on understanding.")

# TAB 3: HEATMAP
with tabs[2]:
    st.header("Variable Correlation")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[features + ['concept_understanding_score', 'final_score']].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
