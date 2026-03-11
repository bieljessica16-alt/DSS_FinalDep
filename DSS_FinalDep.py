import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Page Config ---
st.set_page_config(page_title="AI Efficiency Study", layout="wide")

# --- Title ---
st.title("#🤖 Measuring Concept Understanding and Final Score")
st.markdown("### **DSS110 Final Project | Group Members:**")
st.write("Busano, Cortez, Geronimo, Monses, & Perez")
st.divider()

# --- 1. Load and Preprocess Data ---
@st.cache_data
def load_and_clean_data():
    # DIRECT LOAD: Assumes file is in the same directory
    df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    
    # Normalize headers to lowercase (Prof's safety step)
    df.columns = df.columns.str.lower()
    
    # RETAIN ONLY NEEDED COLUMNS (As per Notebook cleaning step)
    required_cols = [
        'grade_level',
        'uses_ai',
        'ai_dependency_score',
        'ai_generated_content_percentage',
        'study_hours_per_day',
        'concept_understanding_score',
        'final_score'
    ]
    df = df[required_cols].copy()
    
    # DROP MISSING VALUES (As per Notebook check)
    df = df.dropna()
    
    # 1. For grade_level (Label encoder as per Notebook)
    # It transforms ordered categorical data into numerical values
    le = LabelEncoder()
    df['grade_level'] = le.fit_transform(df['grade_level'])
    
    return df, le

df, le = load_and_clean_data()

# --- 2. Build the Model (Calculates the REAL R^2) ---
@st.cache_resource
def train_engines(df):
    # Features (Independent Variables) - factors that PREDICT understanding (As per Notebook)
    features = ['grade_level', 'uses_ai', 'ai_dependency_score', 'ai_generated_content_percentage', 'study_hours_per_day']
    X = df[features]
    
    # 2. Scaling Numerical Features (As per Notebook)
    # Scaling these helps you compare which one has a bigger effect
    scaler = StandardScaler()
    features_to_scale = ['grade_level', 'ai_dependency_score', 'ai_generated_content_percentage', 'study_hours_per_day']
    
    # We fit the scaler on the whole dataset to be consistent with notebook workflow
    X_scaled = X.copy()
    X_scaled[features_to_scale] = scaler.fit_transform(X[features_to_scale])
    
    # Target Variable (As per Notebook Research Objective 3)
    y_concept = df['concept_understanding_score']
    y_final = df['final_score']

    # Split data for validation (As per Notebook)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_concept, test_size=0.2, random_state=42)

    # Training the Models
    model_concept = LinearRegression().fit(X_train, y_train)
    
    # Also training a final score model for the simulator
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_scaled, y_final, test_size=0.2, random_state=42)
    model_final = LinearRegression().fit(X_train_f, y_train_f)

    # CALCULATING THE ACTUAL R^2 (Based on Concept Understanding as per Notebook evaluation)
    y_pred = model_concept.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred)
    }
    
    return model_final, model_concept, metrics, features, scaler

model_final, model_concept, stats, feature_list, scaler = train_engines(df)

# --- SIDEBAR: SIMULATOR ---
st.sidebar.header("🕹️ Learning Style Simulator")
st.sidebar.markdown("Adjust these to see the 'AI Paradox' in action.")

# Adjusted Sidebar inputs to match Notebook features
grade_opt = st.sidebar.selectbox("Select Grade Level", options=le.classes_, index=3) # Default to 1st Year
grade_val = le.transform([grade_opt])[0]

hrs = st.sidebar.slider("Study Hours Per Day", 0.5, 10.0, 3.0)
ai_dep = st.sidebar.slider("AI Dependency (1-10)", 1, 10, 5)
ai_pct = st.sidebar.slider("AI Content Percentage", 0, 100, 30)
uses_ai_val = 1 if st.sidebar.checkbox("Uses AI Tools", value=True) else 0

# --- MAIN PAGE ---
tabs = st.tabs(["The Simulator", "Statistical Proof", "Correlation Heatmap"])

# TAB 1: THE INTERACTIVE STORY
with tabs[0]:
    st.header("The Learning Style Simulator")
    st.write("Does using AI lower your actual understanding? Test a profile below.")
    
    # Scaling Input for Prediction (Must match training preprocessing)
    # Feature order: ['grade_level', 'uses_ai', 'ai_dependency_score', 'ai_generated_content_percentage', 'study_hours_per_day']
    raw_input = pd.DataFrame([[grade_val, uses_ai_val, ai_dep, ai_pct, hrs]], columns=feature_list)
    
    features_to_scale = ['grade_level', 'ai_dependency_score', 'ai_generated_content_percentage', 'study_hours_per_day']
    scaled_input = raw_input.copy()
    scaled_input[features_to_scale] = scaler.transform(raw_input[features_to_scale])
    
    # Prediction using Real Data
    pred_understanding = model_concept.predict(scaled_input)[0]
    pred_final = model_final.predict(scaled_input)[0]
    
    # Define Pass/Fail status for the metric (Using 50% as the threshold)
    pass_status = "✅ PASS" if pred_final >= 50 else "❌ FAIL"
    
    # Benchmark Logic (As per Notebook: Comparing AI-reliant students to traditional high-study students)
    # Reloading raw for benchmark to avoid encoded values in comparison logic
    raw_df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    raw_df.columns = raw_df.columns.str.lower()
    
    ai_efficient = raw_df[(raw_df['ai_dependency_score'] > 7) & (raw_df['study_hours_per_day'] < 3)]['concept_understanding_score'].mean()
    traditional_grind = raw_df[(raw_df['ai_dependency_score'] < 2) & (raw_df['study_hours_per_day'] > 5)]['concept_understanding_score'].mean()
    
    # Local efficiency gap for the simulated user vs the traditional benchmark from data
    efficiency_gap = (pred_understanding / traditional_grind) * 100 if traditional_grind > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Concept Understanding", f"{pred_understanding:.2f}/10")
    col2.metric("Predicted Final Score", f"{pred_final:.1f}%", delta=pass_status, delta_color="normal")
    col3.metric("Proficiency Parity", f"{efficiency_gap:.1f}%")

    # --- DESCRIPTION BOXES ---
    st.markdown("### 🔍 Why are my results like this?")
    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        st.info("**Why only ~5.5 Understanding?**\n\nThe data shows an 'Understanding Plateau.' Adding more hours only adds tiny fractions to understanding. This proves quality matters more than quantity.")
    
    with exp2:
        st.info("**What is a Passing Grade?**\n\nIn this study, **50% is a Pass.** If you see 55%, you have successfully passed! To reach 80%+, you usually need higher consistency.")
    
    with exp3:
        st.info("**What is Proficiency Parity?**\n\n**100% means you are performing exactly as well as a 'Traditional Student'** (5+ hrs, No AI). High parity proves AI efficiency.")

     # THE VERDICT
    st.divider()
    if efficiency_gap >= 95:
        st.success(f"### ✅ Verdict: Efficient Mastery\nYou are achieving {efficiency_gap:.1f}% of traditional mastery. You are using AI as an efficiency tool without losing performance.")
    elif efficiency_gap >= 85:
        st.info("### ℹ️ Verdict: Balanced Hybrid\nYou are maintaining solid performance. AI is acting as a tool, not a crutch.")
    else:
        st.warning("### ⚠️ Verdict: Risk of Learning Loss\nYour current habits suggest your understanding is dipping. Increase manual study.")

   
# TAB 2: STATISTICAL PROOF (Regression)
with tabs[1]:
    st.header("⚖️ The Statistical Reality")
    st.write("We used **Linear Regression** to see if AI variables 'tank' a student's actual understanding.")

    # 1. Top Level Metrics
    r_squared = stats['r2']  # Dynamic R2 from your dataset as calculated in the file
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
        # Building the dataframe dynamically from the trained model (Matching Notebook output)
        coef_data = {
            "Variable": ["Grade Level", "Uses AI", "AI Dependency", "AI Content %", "Study Hours"],
            "Impact Weight": model_concept.coef_
        }
        coef_df = pd.DataFrame(coef_data).sort_values(by="Impact Weight", ascending=False)
        
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
    **The AI Paradox:** Notice that simply *using* AI (`uses_ai`: {model_concept.coef_[1]:.4f}) has a tiny negative weight, 
    but **AI Dependency** and **Content %** are both *positive*. 

    **The Verdict:** It's not *if* you use AI, but *how* you use it. Deep integration (High Dependency) 
    correlates with slightly better understanding than just using it as a surface-level shortcut.
    """)

    st.success("**Mic Drop Conclusion:** With coefficients this close to zero, AI is statistically 'neutral.' It is a tool that depends entirely on the user's intent.")
    
# TAB 3: HEATMAP
with tabs[2]:
    st.header("Variable Correlation")
    st.write("Visual proof of how Understanding relates to Grades vs AI.")
    # Calculate the correlation matrix (Matching Notebook logic)
    corr_matrix = df.corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax2)
    st.pyplot(fig2)


