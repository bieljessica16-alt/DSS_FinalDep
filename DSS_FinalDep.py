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

# 2. TRAIN THE ENGINE (We added Attendance to the model)
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
attendance = st.sidebar.slider("Class Attendance (%)", 0, 100, 90) # NEW METRIC
ai_dep = st.sidebar.slider("AI Dependency (1-10)", 1, 10, 5)
ai_pct = st.sidebar.slider("AI Content Percentage", 0, 100, 30)

# --- MAIN PAGE ---
st.title("📊 AI & Student Performance: The Efficiency Multiplier")

tabs = st.tabs(["The Simulator", "Statistical Proof", "Correlation"])

# TAB 1: THE INTERACTIVE STORY
with tabs[0]:
    st.header("The Learning Style Simulator")
    st.write("Does using AI lower your actual understanding? Test a profile below.")

    # Calculations
    user_input = np.array([[ai_dep, hrs, ai_pct, attendance]])
    pred_understanding = model_concept.predict(user_input)[0]
    pred_final = model_final.predict(user_input)[0]
    
    # Traditional Benchmark (Hard-working, non-AI student)
    benchmark_score = df[(df['study_hours_per_day'] >= 5) & (df['ai_dependency_score'] < 2)]['final_score'].mean()
    efficiency_gap = (pred_final / benchmark_score) * 100

    # Layout Columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Concept Understanding", f"{pred_understanding:.2f}/10")
    col2.metric("Predicted Final Score", f"{pred_final:.1f}%")
    col3.metric("Proficiency Parity", f"{efficiency_gap:.1f}%")

    # --- THE "WHAT DOES THIS MEAN?" SECTION ---
    st.markdown("### 🔍 How to read these results:")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.help("**Concept Understanding:** This is your actual 'brain power' score. A score of 7+ means you truly understand the material, regardless of using AI.")
    with c2:
        st.help("**Final Score:** This is your predicted grade on an exam. It's what the school sees.")
    with c3:
        st.help("**Proficiency Parity:** This is a comparison. **100%** means you are performing exactly as well as a 'Traditional' student who studies 5+ hours a day with NO AI.")

    # Visual Comparison
    st.markdown("### Your Profile vs. Traditional Gold Standard")
    chart_data = pd.DataFrame({
        "Category": ["Your Profile", "Traditional (High Effort, No AI)"],
        "Final Score": [pred_final, benchmark_score]
    })
    
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(data=chart_data, x="Final Score", y="Category", palette=["#2E86C1", "#ABB2B9"], ax=ax)
    ax.set_xlim(0, 100)
    st.pyplot(fig)

    # THE VERDICT BOX
    st.divider()
    if efficiency_gap >= 95:
        st.success(f"### ✅ Verdict: Efficient Mastery\nEven with AI use, you are achieving {efficiency_gap:.1f}% of the mastery seen in high-effort traditional students. You are working smarter, not harder.")
    elif efficiency_gap >= 85:
        st.info("### ℹ️ Verdict: Balanced Learning\nYou are maintaining solid performance. Your AI usage is acting as a helpful tool rather than a crutch.")
    else:
        st.warning("### ⚠️ Verdict: Risk of Learning Loss\nYour current habits suggest that your understanding is dipping. Try increasing attendance or manual study hours.")

# TAB 2: STATISTICAL PROOF
with tabs[1]:
    st.header("Does AI 'Tank' Brainpower?")
    st.write("Regression analysis showing AI variables do not negatively impact understanding.")
    
    X_stats = sm.add_constant(X)
    stats_model = sm.OLS(y_concept, X_stats).fit()
    st.text(str(stats_model.summary().tables[1]))
    
    st.markdown("""
    > **The Data Proof:** Look at the 'P>|t|' column. If the number is greater than 0.05, it means AI has **no significant negative effect** on understanding.
    """)

# TAB 3: HEATMAP
with tabs[2]:
    st.header("Variable Correlation")
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(df[features + ['concept_understanding_score', 'final_score']].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
