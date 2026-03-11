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
    st.header("⚖️ The Statistical Reality")
    
    # Summary Statistics
    r_squared = 0.0007  # From your results
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("Model Predictive Power (R²)", f"{r_squared:.4f}")
        st.info("""
        **The "Noise" Factor:** An R² of 0.0007 indicates that AI usage and study hours currently explain almost none of the variance in scores. 
        This suggests that **individual talent, prior knowledge, or teacher quality** are likely much bigger factors than the tools used.
        """)

    with col2:
        st.subheader("Impact Coefficients")
        # Creating a clean table for the coefficients
        coef_data = {
            "Variable": ["Study Hours", "AI Content %", "AI Dependency", "Grade Level", "Uses AI"],
            "Weight": [0.0491, 0.0473, 0.0441, 0.0390, -0.0454]
        }
        coef_df = pd.DataFrame(coef_data)
        st.table(coef_df)

    st.divider()

    # --- VISUALIZING THE IMPACT ---
    st.subheader("Visualizing Feature Importance")
    fig_coef, ax_coef = plt.subplots(figsize=(8, 4))
    
    # Color coding: Green for positive impact, Red for negative
    colors = ['#27AE60' if x > 0 else '#E74C3C' for x in coef_df['Weight']]
    
    sns.barplot(x='Weight', y='Variable', data=coef_df.sort_values('Weight', ascending=False), palette=colors, ax=ax_coef)
    ax_coef.set_title("Which factors actually move the needle?")
    st.pyplot(fig_coef)

    # --- THE "SO WHAT?" SECTION ---
    st.markdown("### 🔍 What does this tell us?")
    
    # Logic based on your negative 'uses_ai' vs positive 'ai_dependency'
    st.warning("""
    **The AI Paradox:** Simply "using AI" (`-0.0454`) has a slight negative correlation with performance. However, **AI Dependency** (`+0.0441`) and **AI Content %** (`+0.0473`) are positive. 
    
    **Translation:** It's not *whether* you use AI, but *how* you use it. Students who integrate AI deeply (high dependency/content) might be using it more effectively than those who just "dabble" in it.
    """)
    
# TAB 3: HEATMAP
with tabs[2]:
    st.header("Variable Correlation")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[features + ['concept_understanding_score', 'final_score']].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)


