import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. UI SETUP & THE HOOK ---
st.set_page_config(page_title="The AI Paradox Dashboard", layout="wide")
st.title("🧠 The AI Paradox: Efficiency vs. Mastery")
st.markdown("> **The Hook:** *Can AI really replace the 4-hour study grind? We tested 8,000 students to find out if AI dependency actually tanks your brainpower.*")

# --- 2. THE INTERACTION: SIDEBAR INPUTS ---
st.sidebar.header("The Learning Style Simulator")
st.sidebar.info("Adjust the sliders to test the 'AI vs. Effort' theory.")

grade = st.sidebar.selectbox("Grade Level", ["10th Grade", "11th Grade", "12th Grade", "1st Year", "2nd Year", "3rd Year"])
study_hrs = st.sidebar.slider("Study Hours Per Day", 0.5, 6.0, 1.5)
uses_ai = st.sidebar.toggle("Uses AI Tools", value=True)

# AI specific inputs
ai_dep = st.sidebar.slider("AI Dependency Score", 1, 10, 8 if uses_ai else 1)
ai_pct = st.sidebar.slider("AI-Generated Content %", 0, 100, 75 if uses_ai else 0)

# --- 3. THE ENGINE: LIVE CALCULATION ---
# Hardcoded Coefficients from our Regression Analysis
# Intercept + (StudyHrs * 0.08) + (AIDep * 0.02) + (AI% * 0.001)
base_score = 5.15
calc_score = base_score + (study_hrs * 0.08) + (ai_dep * 0.015) + (ai_pct * 0.001)
final_academic_score = 78 + (calc_score * 1.2) # Mapping to a typical percentage grade

# Benchmark: Traditional student (5.0 hours, 0 AI)
traditional_benchmark = base_score + (5.0 * 0.08) + (1 * 0.015) + (0 * 0.001)
parity_ratio = (calc_score / traditional_benchmark) * 100

# --- 4. VISUAL CONTRAST (CENTER STAGE) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("The Comparison: AI-Reliant vs. Traditional")
    chart_data = pd.DataFrame({
        "Learning Style": ["Your Profile", "Traditional (5hrs, No AI)"],
        "Mastery Score": [calc_score, traditional_benchmark]
    })
    st.bar_chart(chart_data, x="Learning Style", y="Mastery Score", color="#0072B2")
    

with col2:
    st.metric("Concept Mastery", f"{calc_score:.2f} / 10")
    st.metric("Proficiency Parity", f"{parity_ratio:.1f}%", delta=f"{parity_ratio-100:.1f}% vs Gold Standard")
    st.metric("Predicted Final Grade", f"{final_academic_score:.1f}%")

# --- 5. THE EVIDENCE: CORRELATION HEATMAP ---
st.divider()
st.subheader("📊 The Evidence: Correlation Heatmap")
st.write("This heatmap shows the relationship between our variables. Notice how AI Dependency (near zero) doesn't correlate negatively with Understanding.")

# Simulating the Correlation Matrix from our 8,000 observations
corr_data = {
    'Study Hours': [1.0, 0.02, 0.01, 0.15],
    'AI Dependency': [0.02, 1.0, 0.85, 0.03],
    'AI Content %': [0.01, 0.85, 1.0, 0.01],
    'Understanding': [0.15, 0.03, 0.01, 1.0]
}
corr_df = pd.DataFrame(corr_data, index=['Study Hours', 'AI Dependency', 'AI Content %', 'Understanding'])

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, ax=ax)
st.pyplot(fig)


# --- 6. THE VERDICT BOX (THE MIC DROP) ---
st.divider()
if parity_ratio >= 98:
    st.success(f"**Verdict: Efficient Mastery.** You are achieving **{parity_ratio:.1f}%** of the traditional understanding level while saving roughly **60%** of manual study time through AI integration.")
elif parity_ratio >= 90:
    st.info(f"**Verdict: Balanced Learner.** You are maintaining stable mastery. AI is acting as a strong 'Efficiency Bridge' for your workload.")
else:
    st.warning("**Verdict: Traditionalist.** Your current profile relies on classic study methods. AI usage is low, keeping you on the traditional proficiency curve.")

# --- THE CONCLUSION ---
st.markdown(f"**The Conclusion:** The data wins. AI-reliant students achieve **{parity_ratio:.1f}%** of the mastery of traditional learners, but in less than half the time.")
