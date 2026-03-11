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
features = ['ai_dependency_score', 'study_hours_per_day', 'ai_generated_content_percentage', 'attendance_percentage']
X = df[features]
y_concept = df['concept_understanding_score']
y_final = df['final_score']

model_concept = LinearRegression().fit(X, y_concept)
model_final = LinearRegression().fit(X, y_final)

# --- SIDEBAR: THE SIMULATOR INPUTS ---
st.sidebar.header("🕹️ Learning Style Simulator")
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
    user_input = np.array([[ai_dep, hrs, ai_pct, attendance]])
    pred_understanding = model_concept.predict(user_input)[0]
    pred_final = model_final.predict(user_input)[0]
    
    pass_status = "✅ PASS" if pred_final >= 50 else "❌ FAIL"
    benchmark_score = df[(df['study_hours_per_day'] >= 5) & (df['ai_dependency_score'] < 2)]['final_score'].mean()
    efficiency_gap = (pred_final / benchmark_score) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Concept Understanding", f"{pred_understanding:.2f}/10")
    col2.metric("Predicted Final Score", f"{pred_final:.1f}%", delta=pass_status, delta_color="normal")
    col3.metric("Proficiency Parity", f"{efficiency_gap:.1f}%")

    st.markdown("### 🔍 Why are my results like this?")
    exp1, exp2, exp3 = st.columns(3)
    with exp1: st.info("**Understanding Plateau:** Adding more hours only adds tiny fractions to understanding in our data.")
    with exp2: st.info("**Pass Mark:** 50% is a Pass. To get higher, Attendance is usually the key driver.")
    with exp3: st.info("**Efficiency:** 100% means you match a 'Traditional Student' who studies 5+ hrs with no AI.")

# TAB 2: STATISTICAL PROOF (Regression & Feature Importance)
with tabs[1]:
    st.header("⚖️ Does AI 'Tank' Brainpower?")
    
    # 1. Run the model for stats
    X_stats = sm.add_constant(X)
    stats_model = sm.OLS(y_concept, X_stats).fit()
    
    # 2. Display Feature Importance Chart with Labels
    st.subheader("Visualizing Feature Impact")
    st.write("This chart shows how much each 'knob' moves the Understanding Score. Larger numbers = bigger impact.")
    
    # Prepare data for chart
    importance_df = pd.DataFrame({
        'Feature': ['AI Dependency', 'Study Hours', 'AI Content %', 'Attendance'],
        'Impact (Weight)': stats_model.params[1:].values # Exclude constant
    }).sort_values(by='Impact (Weight)', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#E74C3C' if x < 0 else '#2ECC71' for x in importance_df['Impact (Weight)']]
    bars = ax.barh(importance_df['Feature'], importance_df['Impact (Weight)'], color=colors)
    
    # --- ADDING NUMBERS/LABELS TO BARS ---
    ax.bar_label(bars, fmt='%.4f', padding=5, fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Weight (Coefficient)')
    ax.set_title('Impact on Concept Understanding Score')
    st.pyplot(fig)

    # 3. Display Detailed Regression Table
    st.subheader("Detailed Regression Table")
    results_df = pd.DataFrame({
        "Weight": stats_model.params,
        "P-Value": stats_model.pvalues,
        "Verdict": ["-" if i==0 else "✅ SAFE" if p > 0.05 else "⚠️ Significant" for i, p in enumerate(stats_model.pvalues)]
    })
    st.dataframe(results_df.style.format({"Weight": "{:.4f}", "P-Value": "{:.4f}"}), use_container_width=True)

# TAB 3: HEATMAP
with tabs[2]:
    st.header("Variable Correlation")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[features + ['concept_understanding_score', 'final_score']].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
