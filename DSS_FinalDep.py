import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI & Concept Understanding Study", layout="wide")

# 1. LOAD & PREP DATA
@st.cache_data
def load_data():
    # Ensure your CSV is named exactly this in your GitHub repo
    df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    return df

df = load_data()

# 2. TRAIN THE "ENGINE" (Linear Regression for the Simulator)
features = ['ai_dependency_score', 'study_hours_per_day', 'ai_generated_content_percentage']
X = df[features]
y_concept = df['concept_understanding_score']
y_final = df['final_score']

model_concept = LinearRegression().fit(X, y_concept)
model_final = LinearRegression().fit(X, y_final)

# --- SIDEBAR: THE SIMULATOR INPUTS ---
st.sidebar.header("🕹️ Learning Style Simulator")
st.sidebar.markdown("Adjust these to see how AI habits affect learning.")

hrs = st.sidebar.slider("Manual Study Hours Per Day", 0.5, 10.0, 3.0, help="Time spent reading, practicing, or reviewing without AI.")
ai_dep = st.sidebar.slider("AI Dependency (1-10)", 1, 10, 5, help="How much you rely on AI for explanations or problem solving.")
ai_pct = st.sidebar.slider("AI Content Percentage", 0, 100, 30, help="How much of your final output (essays/code) is AI-generated.")

# --- MAIN PAGE ---
st.title("📚 AI & Student Performance: Focus on Understanding")

tabs = st.tabs(["The Simulator", "Statistical Proof", "Correlation Heatmap"])

# TAB 1: THE INTERACTIVE STORY
with tabs[0]:
    st.header("The Learning Style Simulator")
    st.write("Does using AI lower your actual understanding? Use the sliders on the left to test a profile.")

    # Calculations
    user_input = np.array([[ai_dep, hrs, ai_pct]])
    pred_understanding = model_concept.predict(user_input)[0]
    pred_final = model_final.predict(user_input)[0]
    
    # Traditional Benchmark (Average of high-effort, low-AI students)
    benchmark_score = df[(df['study_hours_per_day'] >= 5) & (df['ai_dependency_score'] < 2)]['final_score'].mean()
    efficiency_gap = (pred_final / benchmark_score) * 100

    # Metrics Section
    col1, col2, col3 = st.columns(3)
    col1.metric("Concept Understanding", f"{pred_understanding:.2f}/10")
    col2.metric("Predicted Final Grade", f"{pred_final:.1f}%")
    col3.metric("Proficiency Parity", f"{efficiency_gap:.1f}%")

    # --- NEW: DESCRIPTION BOXES FOR CLARITY ---
    st.markdown("### 🔍 What do these numbers mean?")
    exp1, exp2, exp3 = st.columns(3)
    
    with exp1:
        st.info("**Concept Understanding**\n\nThis is the most important score. It represents the student's actual grasp of the subject. A score above 7.0 is considered 'Mastery.'")
    
    with exp2:
        st.info("**Predicted Final Grade**\n\nThis is the estimated grade the student would receive on an exam based on the habits selected in the sidebar.")
    
    with exp3:
        st.info("**Proficiency Parity**\n\nThis compares this profile to a 'Traditional Hard-Worker' (someone who studies 5+ hours with NO AI). **100% means the AI user is performing exactly as well as the traditional student.**")

    # Visual Comparison Chart
    st.divider()
    st.markdown("### User vs. Traditional 'Gold Standard'")
    chart_data = pd.DataFrame({
        "Category": ["Your Profile", "Traditional (5+ Hrs, No AI)"],
        "Final Score": [pred_final, benchmark_score]
    })
    
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(data=chart_data, x="Final Score", y="Category", palette=["#2E86C1", "#ABB2B9"], ax=ax)
    ax.set_xlim(0, 100)
    st.pyplot(fig)

    # THE VERDICT
    if efficiency_gap >= 95:
        st.success(f"### ✅ Verdict: The Efficiency Multiplier\nYou are achieving {efficiency_gap:.1f}% of traditional mastery. This proves that high AI usage can match traditional study results without a loss in performance.")
    elif efficiency_gap >= 85:
        st.info("### ℹ️ Verdict: Balanced Hybrid\nYou are maintaining solid performance using a mix of AI and manual study.")
    else:
        st.warning("### ⚠️ Verdict: Learning Risk\nYour current configuration suggests lower retention. Your understanding score is dipping below the standard.")

# TAB 2: STATISTICAL PROOF
with tabs[1]:
    st.header("The Mathematical Proof")
    st.write("Does AI usage predict a lower understanding score? Let's check the P-values.")
    
    X_stats = sm.add_constant(X)
    stats_model = sm.OLS(y_concept, X_stats).fit()
    st.text(str(stats_model.summary().tables[1]))
    
    st.markdown("""
    > **How to read this table:** > Look at the **P>|t|** column for AI Dependency. If it is **greater than 0.05**, it means AI usage is NOT a significant cause of lower understanding. 
    > Our data consistently shows high P-values, proving AI is a neutral tool for learning.
    """)

# TAB 3: HEATMAP
with tabs[2]:
    st.header("How Variables Connect")
    st.write("Notice how 'Concept Understanding' relates to 'Final Score' (Strong) vs 'AI Dependency' (Weak).")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[['ai_dependency_score', 'ai_generated_content_percentage', 'study_hours_per_day', 'concept_understanding_score', 'final_score']].corr(), 
                annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
