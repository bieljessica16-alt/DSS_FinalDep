import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# 1. LOAD AND CLEAN DATA
df = pd.read_csv('ai_impact_student_performance_dataset.csv')
cols = [
    'grade_level', 'uses_ai', 'ai_dependency_score', 
    'ai_generated_content_percentage', 'study_hours_per_day', 
    'concept_understanding_score', 'final_score'
]
df_cleaned = df[cols].copy()

# 2. THE CORRELATION HEATMAP (The Overview)
plt.figure(figsize=(10, 8))
corr = df_cleaned.select_dtypes(include=['number']).corr()
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt=".2f")
plt.title('Correlation Heatmap: AI Usage vs. Learning Outcomes')
plt.show() 

# 3. VISUAL EVIDENCE (The "No Harm" Proof)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# AI Dependency vs. Understanding
sns.regplot(data=df_cleaned, x='ai_dependency_score', y='concept_understanding_score', 
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax1)
ax1.set_title('AI Dependency vs. Concept Understanding\n(Flat line = No negative impact)')

# Understanding vs. Final Score
sns.regplot(data=df_cleaned, x='concept_understanding_score', y='final_score', 
            scatter_kws={'alpha':0.3}, line_kws={'color':'green'}, ax=ax2)
ax2.set_title('Concept Understanding vs. Final Score\n(Strong link = Understanding drives grades)')
plt.tight_layout()
plt.show()

# 4. MULTIPLE LINEAR REGRESSION (The Statistical Proof)
# Target: Concept Understanding (To see if AI "rots" the brain)
X = df_cleaned[['ai_dependency_score', 'ai_generated_content_percentage', 'study_hours_per_day']]
y = df_cleaned['concept_understanding_score']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print("--- REGRESSION RESULTS: PREDICTING UNDERSTANDING ---")
print(model.summary())
