import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. PREPARE DATA
# Map text grade levels to numbers so the model can process them
grade_map = {
    '10th': 10, '11th': 11, '12th': 12, 
    '1st Year': 13, '2nd Year': 14, '3rd Year': 15
}
df_cleaned['grade_level_num'] = df_cleaned['grade_level'].map(grade_map)

# 2. SELECT FEATURES
# We use the new numerical grade column
features = ['grade_level_num', 'uses_ai', 'ai_dependency_score', 
            'ai_generated_content_percentage', 'study_hours_per_day']

X = df_cleaned[features]
y = df_cleaned['concept_understanding_score']

# 3. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. TRAIN MODEL (Scikit-Learn for Prediction)
model = LinearRegression()
model.fit(X_train, y_train)

# 5. STATISTICAL PROOF (Statsmodels for P-values)
# This part is crucial for your conclusion!
X_const = sm.add_constant(X_train)
stats_model = sm.OLS(y_train, X_const).fit()

# --- OUTPUT RESULTS ---
print(f"Model Accuracy (R-squared): {r2_score(y_test, model.predict(X_test)):.4f}")
print("\n--- Coefficients (The Impact of Each Factor) ---")
for col, coef in zip(features, model.coef_):
    print(f"{col}: {coef:.4f}")

print("\n--- P-Values (Statistical Significance) ---")
print(stats_model.pvalues)
