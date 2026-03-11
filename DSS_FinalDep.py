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

    # Train Models
    model_final = LinearRegression().fit(X_train, y_train)
    model_concept = LinearRegression().fit(X, y_concept) # Trained on all for simulator accuracy

    # Calculate real accuracy metrics on the Test Set
    y_pred = model_final.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred)
    }
    
    return model_final, model_concept, metrics, features

# Initialize everything
model_final, model_concept, stats, feature_list = train_engines(df)

# --- SIDEBAR: THE SIMULATOR INPUTS ---
st.sidebar.header("🕹️ Learning Style Simulator")
st.sidebar.markdown("Adjust these to see the 'AI Paradox' in action.")

hrs = st.sidebar.slider("Study Hours Per Day", 0.5, 10.0, 3.0)
attendance = st.sidebar.slider("Class Attendance (%)", 0, 100, 90)
ai_dep = st.sidebar.slider("AI Dependency (1-10)", 1, 10, 5)
ai_pct = st.sidebar.slider("AI Content Percentage", 0, 100, 30)

# --- MAIN PAGE ---
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
    benchmark_df = df[(df['study_hours_per_day'] >= 5) & (df['ai_dependency_score'] < 2)]
    benchmark_score = benchmark_df['final_score'].mean() if not benchmark_df.empty else 65.0
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
        st.success(f"### ✅ Verdict: Efficient Mastery\nYou are achieving {efficiency_gap:.1f}% of traditional mastery. You are using AI as an efficiency tool without losing performance.")
    elif efficiency_gap >= 85:
        st.info("### ℹ️ Verdict: Balanced Hybrid\nYou are maintaining solid performance. AI is acting as a tool, not a crutch.")
    else:
        st.warning("### ⚠️ Verdict: Risk of Learning Loss\nYour current habits suggest your understanding is dipping. Increase manual study or attendance.")

# TAB 2: STATISTICAL PROOF (Regression)
with tabs[1]:
    st.header("⚖️ The Statistical Reality")
    st.write("We used **Linear Regression** to see if AI variables 'tank' a student's actual understanding.")
    
    # 1. Top Level Metrics
    r_squared = stats['r2']  # Dynamic R2 from your dataset
    m1, m2 = st.columns([1, 2])
    
    with m1:
        st.metric("Model Predictive Power (R²)", f"{r_squared:.4f}")
    with m2:
        st.info(f"**Insight:** An R² of {r_squared:.4f} means these factors only explain {r_squared*100:.2f}% of the score. This proves that AI isn't the 'grade-killer' people fear; performance is likely driven by external factors like prior knowledge or teaching quality.")

    st.divider()

    # 2. Data & Visualization
    col_table, col_viz = st.columns([2, 3])

    with col_table:
        st.subheader("Impact Coefficients")
        # Building the dataframe from the trained model's coefficients
        coef_df = pd.DataFrame({
            "Variable": feature_list,
            "Impact Weight": model_final.coef_
        }).sort_values(by="Impact Weight", ascending=False)
        
        # Displaying a styled table
        st.table(coef_df)

    with col_viz:
        st.subheader("Visualizing Feature Importance")
        fig_coef, ax_coef = plt.subplots(figsize=(8, 5))
        
        # Logic: Green for positive impact, Red for negative
        colors = ['#27AE60' if x > 0 else '#E74C3C' for x in coef_df['Impact Weight']]
        
        sns.barplot(x='Impact Weight', y='Variable', data=coef_df, palette=colors, ax=ax_coef)
        ax_coef.set_title("Which factors move the needle?")
        ax_coef.set_xlabel("Coefficient Value (Direction of Impact)")
        st.pyplot(fig_coef)

    st.divider()

    # 3. The "So What?" Section
    st.markdown("### 🔍 What does this tell us?")
    
    st.warning(f"""
    **The AI Paradox:** Notice that in the underlying data, simply *using* AI often shows a tiny negative weight, 
    but **AI Dependency** and **Content %** coefficients are often *positive* or near-neutral. 

    **The Verdict:** It's not *if* you use AI, but *how* you use it. Deep integration (High Dependency) 
    correlates with slightly better understanding than just using it as a surface-level shortcut.
    """)

    st.success("**Mic Drop Conclusion:** With coefficients this close to zero, AI is statistically 'neutral.' It is a tool that depends entirely on the user's intent.")
    
# TAB 3: HEATMAP
with tabs[2]:
    st.header("Variable Correlation")
    st.write("Visual proof of how Understanding relates to Grades vs AI.")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[feature_list + ['concept_understanding_score', 'final_score']].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
