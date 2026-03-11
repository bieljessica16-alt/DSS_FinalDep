import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="AI Efficiency Study", layout="wide")

# --- 1. CLEANING & TRAINING ENGINE ---
@st.cache_data
def get_trained_models():
    # Load raw data
    df_raw = pd.read_csv('ai_impact_student_performance_dataset.csv')
    
    # DATA CLEANING: Retain only needed columns as per notebook 
    # We include 'attendance_percentage' for the simulator logic
    df_cleaned = df_raw[[
        'grade_level', 'uses_ai', 'ai_dependency_score',
        'ai_generated_content_percentage', 'study_hours_per_day',
        'attendance_percentage', 'concept_understanding_score', 'final_score'
    ]].copy()

    # MODEL TRAINING
    features = ['ai_dependency_score', 'study_hours_per_day', 'ai_generated_content_percentage', 'attendance_percentage']
    X = df_cleaned[features]
    
    # Targets
    y_concept = df_cleaned['concept_understanding_score']
    y_final = df_cleaned['final_score']

    # Train Models
    model_concept = LinearRegression().fit(X, y_concept)
    model_final = LinearRegression().fit(X, y_final)

    # EVALUATION METRICS (Calculated on the full dataset for the "Proof" tab)
    preds_final = model_final.predict(X)
    metrics = {
        "r2": r2_score(y_final, preds_final),
        "mse": mean_squared_error(y_final, preds_final)
    }

    return df_cleaned, model_concept, model_final, metrics, features

# Initialize everything
df, model_concept, model_final, stats, feature_list = get_trained_models()

# --- 2. SIDEBAR: THE SIMULATOR INPUTS ---
st.sidebar.header("🕹️ Learning Style Simulator")
hrs = st.sidebar.slider("Study Hours Per Day", 0.5, 10.0, 3.0)
attendance = st.sidebar.slider("Class Attendance (%)", 0, 100, 90)
ai_dep = st.sidebar.slider("AI Dependency (1-10)", 1, 10, 5)
ai_pct = st.sidebar.slider("AI Content Percentage", 0, 100, 30)

# --- 3. MAIN PAGE ---
st.title("📊 AI & Student Performance: The Efficiency Multiplier")
tabs = st.tabs(["The Simulator", "Statistical Proof", "Correlation Heatmap"])

# TAB 1: THE INTERACTIVE STORY
with tabs[0]:
    st.header("The Learning Style Simulator")
    
    # Predict using the input order matching the trained features
    user_input = np.array([[ai_dep, hrs, ai_pct, attendance]])
    pred_understanding = model_concept.predict(user_input)[0]
    pred_final = model_final.predict(user_input)[0]
    
    pass_status = "✅ PASS" if pred_final >= 50 else "❌ FAIL"
    
    # Benchmark Logic from Notebook: High effort, No AI 
    benchmark_df = df[(df['study_hours_per_day'] >= 5) & (df['ai_dependency_score'] < 2)]
    benchmark_score = benchmark_df['final_score'].mean() if not benchmark_df.empty else 65.0
    efficiency_gap = (pred_final / benchmark_score) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Concept Understanding", f"{pred_understanding:.2f}/10")
    col2.metric("Predicted Final Score", f"{pred_final:.1f}%", delta=pass_status)
    col3.metric("Proficiency Parity", f"{efficiency_gap:.1f}%")

    st.divider()
    st.markdown("### Your Profile vs. Traditional Gold Standard")
    chart_data = pd.DataFrame({
        "Category": ["Your Profile", "Traditional (High Effort, No AI)"],
        "Final Score": [pred_final, benchmark_score]
    })
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(data=chart_data, x="Final Score", y="Category", palette=["#2E86C1", "#ABB2B9"], ax=ax)
    st.pyplot(fig)

# TAB 2: STATISTICAL PROOF
with tabs[1]:
    st.header("⚖️ The Statistical Reality")
    
    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric("Model Predictive Power (R²)", f"{stats['r2']:.4f}")
    with m2:
        st.info(f"An R² of {stats['r2']:.4f} means these factors explain {stats['r2']*100:.2f}% of the score variance.")

    # Show Feature Importance (Coefficients)
    st.subheader("Which factors move the needle?")
    coef_df = pd.DataFrame({
        "Variable": feature_list,
        "Weight": model_final.coef_
    }).sort_values(by="Weight", ascending=False)
    
    fig_coef, ax_coef = plt.subplots()
    colors = ['#27AE60' if x > 0 else '#E74C3C' for x in coef_df['Weight']]
    sns.barplot(x='Weight', y='Variable', data=coef_df, palette=colors, ax=ax_coef)
    st.pyplot(fig_coef)

# TAB 3: HEATMAP
with tabs[2]:
    st.header("Variable Correlation")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # Core correlations identified in cleaning 
    corr_cols = feature_list + ['concept_understanding_score', 'final_score']
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
