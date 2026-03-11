import streamlit as st
import pandas as pd
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Impact on Student Performance", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # Load your dataset (ensure the CSV is in the same folder)
    df = pd.read_csv("ai_impact_student_performance_dataset.csv")
    
    # Cleaning steps from your notebook
    df_cleaned = df[[
        'grade_level', 'uses_ai', 'ai_dependency_score',
        'ai_generated_content_percentage', 'study_hours_per_day',
        'concept_understanding_score', 'final_score'
    ]].copy()
    return df_cleaned

df = load_data()

# --- SIDEBAR ---
st.sidebar.header("Filter Data")
grade = st.sidebar.multiselect("Select Grade Level:", 
                               options=df["grade_level"].unique(),
                               default=df["grade_level"].unique())

df_selection = df[df["grade_level"].isin(grade)]

# --- MAIN PAGE ---
st.title("📊 DSS110 Final Project: AI Impact Analysis")
st.markdown("## Group Members: Busano, Cortez, Geronimo, Monses, Perez") #

# Top Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Avg Final Score", f"{df_selection['final_score'].mean():.2f}")
m2.metric("Avg AI Dependency", f"{df_selection['ai_dependency_score'].mean():.2f}")
m3.metric("Avg Study Hours", f"{df_selection['study_hours_per_day'].mean():.2f}")

st.divider()

# --- VISUALIZATIONS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("AI Dependency vs. Final Score")
    fig_scatter = px.scatter(df_selection, x="ai_dependency_score", y="final_score", 
                             color="grade_level", trendline="ols")
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("Distribution of Study Hours")
    fig_hist = px.histogram(df_selection, x="study_hours_per_day", nbins=20, color="uses_ai")
    st.plotly_chart(fig_hist, use_container_width=True)

# Efficiency Analysis from your notebook
st.subheader("Efficiency Analysis")
ai_efficient = df_selection[(df_selection['ai_dependency_score'] > 7) & 
                            (df_selection['study_hours_per_day'] < 3)]['concept_understanding_score'].mean()
st.write(f"**Average Understanding (High AI / Low Study):** {ai_efficient:.2f}")

# Display raw data
if st.checkbox("Show Raw Data"):
    st.dataframe(df_selection)
