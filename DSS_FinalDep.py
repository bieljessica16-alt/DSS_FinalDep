import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(page_title="AI & Concept Understanding Study", layout="wide")

# Title
st.title("📚 Study: Does AI Usage Impact Concept Understanding?")
st.markdown("""
This application focuses on **Concept Understanding** as the target. We want to see if
AI dependency and AI-generated content actually hurt a student's grasp of their lessons.
""")

# Load data
@st.cache_data
def load_data():
    # Make sure this filename matches your file on GitHub
    df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    return df

df = load_data()

# 1. THE DATASET AT A GLANCE
st.header("📌 Overview of Learning Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Concept Understanding", f"{df['concept_understanding_score'].mean():.2f}/10")
col2.metric("Avg AI Dependency", f"{df['ai_dependency_score'].mean():.2f}/10")
col3.metric("Avg Study Hours/Day", f"{df['study_hours_per_day'].mean():.2f} hrs")

st.divider()

# 2. THE VISUAL PROOF (HEATMAP)
st.header("📉 Statistical Correlation (Focus on Understanding)")
st.write("This map shows if AI usage has a negative relationship (red/blue) with understanding.")

# Focus columns
cols = ['ai_dependency_score', 'ai_generated_content_percentage', 
        'study_hours_per_day', 'concept_understanding_score']
corr_matrix = df[cols].corr()

fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt=".2f", ax=ax_heat)
st.pyplot(fig_heat)

st.info("""
**How to read this:** Look at the 'concept_understanding_score' row. 
Notice that AI Dependency (0.02) and AI Content (0.02) are almost zero. 
This means as AI usage goes up, understanding DOES NOT go down.
""")

st.divider()

# 3. THE MATHEMATICAL PROOF (REGRESSION)
st.header("⚖️ Statistical Analysis")
st.write("We use Multiple Linear Regression to see if AI predicts a drop in understanding.")

X = df[['ai_dependency_score', 'ai_generated_content_percentage', 'study_hours_per_day']]
y = df['concept_understanding_score']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

st.text(str(model.summary().tables[1]))

st.markdown("""
**The Result:** The 'P-values' (P>|t|) for AI scores are all higher than 0.05. 
In statistics, this means AI has **no significant impact** on understanding.
""")

st.divider()

# 4. THE SIMULATOR (HIGH UNDERSTANDING PREDICTOR)
st.header("🤖 Simulator: Will AI usage lead to High Understanding?")
st.write("""
To make this easy to understand, we defined **'High Understanding'** as a score of 7/10 or higher.
The AI model below predicts if a student will reach that level based on their habits.
""")

# Create a binary target for 'High Understanding'
df['high_understanding'] = (df['concept_understanding_score'] >= 7).astype(int)

# Logistic Regression setup
features = ['ai_dependency_score', 'ai_generated_content_percentage', 'study_hours_per_day']
X_sim = df[features]
y_sim = df['high_understanding']

X_train, X_test, y_train, y_test = train_test_split(X_sim, y_sim, test_size=0.2, random_state=42)
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Simulator UI
col_in, col_out = st.columns([1, 1])

with col_in:
    st.subheader("1. Adjust Student Habits")
    u_dep = st.slider("AI Dependency Score (0-10)", 0, 10, 5)
    u_content = st.slider("AI Content Percentage (0-100%)", 0, 100, 30)
    u_hours = st.slider("Manual Study Hours/Day", 0, 10, 3)

with col_out:
    prediction = lr_model.predict([[u_dep, u_content, u_hours]])
    # Get the probability for the 'High Understanding' class
    prob = lr_model.predict_proba([[u_dep, u_content, u_hours]])[0][1]
    
    st.subheader("2. Prediction Result")
    if prediction[0] == 1:
        st.success("**PREDICTION: HIGH UNDERSTANDING (>= 7/10)**")
        st.write(f"**Confidence Level: {prob*100
