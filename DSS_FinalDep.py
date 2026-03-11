import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

st.set_page_config(page_title="AI & Student Performance Study", layout="wide")

st.title("📊 Impact of AI on Student Performance & Understanding")
st.markdown("This application analyzes whether AI usage (dependency and content generation) affects students' **Conceptual Understanding** and their **Final Grades**.")

# Load data
@st.cache_data
def load_data():
    # Ensure this matches your filename on GitHub
    df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    cols = [
        'grade_level', 'uses_ai', 'ai_dependency_score', 
        'ai_generated_content_percentage', 'study_hours_per_day', 
        'concept_understanding_score', 'final_score'
    ]
    return df[cols].copy()

df = load_data()

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Correlation Analysis", "Regression Model"])

if page == "Overview":
    st.header("Project Overview")
    st.write("Does AI usage hinder learning? Let's look at the data.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Understanding", round(df['concept_understanding_score'].mean(), 2))
    col2.metric("Avg AI Dependency", round(df['ai_dependency_score'].mean(), 2))
    col3.metric("Avg Final Score", round(df['final_score'].mean(), 2))

    st.subheader("Data Preview")
    st.dataframe(df.head(10))

elif page == "Correlation Analysis":
    st.header("The Correlation Heatmap")
    st.write("This map shows how variables relate to each other. Values near 1.0 indicate a strong positive link.")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.select_dtypes(include=['number']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    st.info("**Key Insight:** Notice that **Concept Understanding** has a strong link (0.42) to **Final Score**, while **AI Dependency** has almost zero correlation (0.02) with understanding. This suggests AI use doesn't decrease understanding.")

elif page == "Regression Model":
    st.header("Statistical Proof: Multiple Linear Regression")
    
    target = st.selectbox("Select Target Variable", ["concept_understanding_score", "final_score"])
    
    features = ['ai_dependency_score', 'ai_generated_content_percentage', 'study_hours_per_day']
    if target == "final_score":
        features.append('concept_understanding_score')
        
    X = df[features]
    y = df[target]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    
    st.write(f"### Predicting {target.replace('_', ' ').title()}")
    st.text(str(model.summary()))
    
    st.subheader("Interpretation")
    if target == "concept_understanding_score":
        st.write("The **P-values** for AI metrics are high (above 0.05), which means **AI usage is not a significant predictor of understanding levels.**")
