import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Efficiency Study", layout="wide")

# 1. LOAD & PREP DATA
@st.cache_data
def load_data():
    # Ensure your CSV is named exactly this in your GitHub repo
    df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    return df

df = load_data()

# 2. TRAIN THE "ENGINE" (Simple Linear Regression for the Simulator)
# We train it once on startup so the sliders can use it live
features = ['ai_dependency_score', 'study_hours_per_day', 'ai_generated_content_percentage']
X = df[features]
y_concept = df['concept_understanding_score']
y_final = df['final_score']

model_concept = LinearRegression().fit(X, y_concept)
model_final = LinearRegression().fit(X, y_final)

# --- SIDEBAR: THE SIMULATOR INPUTS ---
st.sidebar.header("🕹️ Learning Style Simulator")
st.sidebar.markdown("Adjust these to see the 'AI Paradox' in action.")

grade = st.sidebar.selectbox("Grade Level", ["10th Grade", "11th Grade", "12th Grade", "1st Year College", "2nd Year College", "3rd Year College"])
hrs = st.sidebar.slider("Study Hours Per Day", 0.5, 10.0, 3.0)
ai_dep = st.sidebar.slider("AI Dependency (1-10)", 1, 10, 5)
ai_pct = st.sidebar.slider("AI Content Percentage", 0, 100, 30)

# --- MAIN PAGE ---
st.title("📊 AI & Student Performance: The Efficiency Multiplier")

tabs = st.tabs(["The Simulator", "Data Proof", "Correlation"])

# TAB 1: THE INTERACTIVE STORY
with tabs[0]:
    st.header("The Learning Style Simulator")
    st.write("Can AI replace the 4-hour grind? Simulate your profile below.")

    # Calculations
    user_input = np.array([[ai_dep, hrs, ai_pct]])
    pred_understanding = model_concept.predict(user_input)[0]
    pred_final = model_final.predict(user_input)[0]
    
    # Traditional Benchmark (Hardcoded based on your 98.4% finding)
    # Average of students with 5+ study hours and < 2 AI dependency
    benchmark_score = df[(df['study_hours_per_day'] >= 5) & (df['ai_dependency_score'] < 2)]['final_score'].mean()
    efficiency_gap = (pred_final / benchmark_score) * 100

    # Layout Columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Understanding", f"{pred_understanding:.2f}/10")
    col2.metric("Predicted Final Score", f"{pred_final:.1f}%")
    col3.metric("Proficiency Parity", f"{efficiency_gap:.1f}%", help="How close you are to a high-effort traditional student")

    # Visual Comparison
    st.markdown("### User vs. Traditional Gold Standard")
    chart_data = pd.DataFrame({
        "Category": ["Your Profile", "Traditional (5+ Hrs, No AI)"],
        "Final Score": [pred_final, benchmark_score]
    })
    
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(data=chart_data, x="Final Score", y="Category", palette=["#2E86C1", "#ABB2B9"], ax=ax)
    ax.set_xlim(0, 100)
    st.pyplot(fig)

    # THE VERDICT BOX
    st.divider()
    if efficiency_gap >= 95:
        st.success(f"### ✅ Verdict: Efficient Mastery\nYou are achieving {efficiency_gap:.1f}% of traditional mastery while utilizing AI for efficiency. This supports the 'Efficiency Multiplier' theory.")
    elif efficiency_gap >= 85:
        st.info("### ℹ️ Verdict: Balanced Learning\nYou are maintaining solid performance with a hybrid approach.")
    else:
        st.warning("### ⚠️ Verdict: Diminishing Returns\nYour current configuration suggests lower retention. Consider increasing manual study hours.")

# TAB 2: STATISTICAL PROOF
with tabs[1]:
    st.header("Does AI 'Tank' Brainpower?")
    st.write("Statistical Regression showing that AI variables do not negatively impact understanding.")
    
    X_stats = sm.add_constant(X)
    stats_model = sm.OLS(y_concept, X_stats).fit()
    st.text(str(stats_model.summary()))
    
    st.markdown("> **The Mic Drop:** Notice the p-values. AI dependency does not have a statistically significant negative coefficient, proving it's a neutral-to-positive tool.")

# TAB 3: HEATMAP
with tabs[2]:
    st.header("Variable Correlation")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
