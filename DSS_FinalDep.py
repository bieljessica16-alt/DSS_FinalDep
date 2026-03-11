import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="AI Efficiency Study", layout="wide")

# --- Title ---
st.title("Measuring Concept Understanding and Final Score Prediction")

# --- 1. Load and Preprocess Data ---
@st.cache_data
def load_and_clean_data():
    # DIRECT LOAD: No checks, assumes file is present and correct
    df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    
    # Normalize headers to lowercase 
    df.columns = df.columns.str.lower()
    
    # Select columns and drop missing values
    # Note: We use the lowercase names here
    required_cols = [
        'ai_dependency_score', 'study_hours_per_day', 
        'ai_generated_content_percentage', 'attendance_percentage',
        'concept_understanding_score', 'final_score'
    ]
    df = df[required_cols]
    df = df.dropna()
    
    return df

df = load_and_clean_data()

# --- 2. Build the Model ---
@st.cache_resource
def train_engines(df):
    features = ['ai_dependency_score', 'study_hours_per_day', 'ai_generated_content_percentage', 'attendance_percentage']
    X = df[features]
    y_final = df['final_score']
    y_concept = df['concept_understanding_score']

    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.2, random_state=42)

    # Train Linear Regression Models
    model_final = LinearRegression().fit(X_train, y_train)
    model_concept = LinearRegression().fit(X, y_concept) # Trained on all for simulator accuracy

    # Calculate real accuracy metrics on the Test Set
    y_pred = model_final.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred)
    }
    
    # Return values must match the initialization call below
    return df, model_concept, model_final, metrics, features

# Initialize everything - Calling the correct function name
df, model_concept, model_final, stats, feature_list = train_engines(df)

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
    # Order: ai_dependency_score, study_hours_per_day, ai_generated_content_percentage, attendance_percentage
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
