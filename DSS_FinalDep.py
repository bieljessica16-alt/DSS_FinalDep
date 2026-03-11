import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set page layout
st.set_page_config(page_title="AI Impact Study", layout="wide")

# Title and Context
st.title("📊 Study: The Impact of AI on Student Learning")
st.markdown("""
This dashboard presents the findings on whether AI dependency affects conceptual understanding 
and academic success. Use the sections below to explore the data and statistical proofs.
""")

# 1. DATA LOADING
@st.cache_data
def load_data():
    # Make sure this filename matches your file on GitHub
    df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    return df

df = load_data()

# 2. KEY METRICS
st.header("📌 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", len(df))
col2.metric("Avg Understanding", f"{df['concept_understanding_score'].mean():.2f}/10")
col3.metric("Avg AI Dependency", f"{df['ai_dependency_score'].mean():.2f}/10")
col4.metric("Passing Rate", f"{(df['passed'].mean()*100):.1f}%")

st.divider()

# 3. CORRELATION HEATMAP (Visual Evidence)
st.header("📉 Statistical Relationships (Heatmap)")
st.write("We use this to see if AI usage (Dependency/Content) correlates with lower understanding.")

# Select relevant columns for the heatmap
corr_cols = ['ai_dependency_score', 'ai_generated_content_percentage', 
             'concept_understanding_score', 'study_hours_per_day', 'final_score']
corr_matrix = df[corr_cols].corr()

fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt=".2f", ax=ax_heat)
st.pyplot(fig_heat)

st.info("💡 **Key Finding:** There is almost **zero correlation** between AI Dependency and Understanding. This suggests that using AI tools does not inherently decrease a student's grasp of the material.")

st.divider()

# 4. STATISTICAL PROOF (Multiple Linear Regression)
st.header("⚖️ Does AI Predict Understanding?")
st.write("This regression model checks if AI usage is a significant 'predictor' of learning loss.")

X_lin = df[['ai_dependency_score', 'ai_generated_content_percentage', 'study_hours_per_day']]
y_lin = df['concept_understanding_score']
X_lin = sm.add_constant(X_lin)
lin_model = sm.OLS(y_lin, X_lin).fit()

# Displaying results in a clean way
st.text(str(lin_model.summary().tables[1]))
st.write("**Conclusion:** High P-values (P > |t|) for AI scores confirm that AI usage is **not** a statistically significant cause of lower understanding.")

st.divider()

# 5. PASS/FAIL PREDICTOR (Logistic Regression)
st.header("🤖 Live Predictor: Will a Student Pass?")
st.write("Using `scikit-learn` to predict if a student will pass based on their habits.")

# Prepare Logistic Regression
features = ['ai_dependency_score', 'concept_understanding_score', 'study_hours_per_day']
X_log = df[features]
y_log = df['passed']

X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.2, random_state=42)
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Interactive UI for User
col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("Adjust Student Stats")
    user_dep = st.slider("AI Dependency (0-10)", 0, 10, 5)
    user_und = st.slider("Concept Understanding (0-10)", 0, 10, 7)
    user_hrs = st.slider("Study Hours Per Day", 0, 10, 3)

with col_result:
    prediction = lr_model.predict([[user_dep, user_und, user_hrs]])
    prob = lr_model.predict_proba([[user_dep, user_und, user_hrs]])[0][1]
    
    st.subheader("Model Result")
    if prediction[0] == 1:
        st.success(f"PREDICTION: PASS (Confidence: {prob*100:.1f}%)")
    else:
        st.error(f"PREDICTION: FAIL (Confidence: {(1-prob)*100:.1f}%)")

st.caption(f"Model Accuracy Score: {accuracy_score(y_test, lr_model.predict(X_test))*100:.2f}%")
