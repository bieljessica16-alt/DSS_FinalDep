# TAB 2: STATISTICAL PROOF
with tabs[1]:
    st.header("⚖️ The Statistical Reality")
    
    # Summary Statistics
    r_squared = 0.0007  # From your results
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("Model Predictive Power (R²)", f"{r_squared:.4f}")
        st.info("""
        **The "Noise" Factor:** An R² of 0.0007 indicates that AI usage and study hours currently explain almost none of the variance in scores. 
        This suggests that **individual talent, prior knowledge, or teacher quality** are likely much bigger factors than the tools used.
        """)

    with col2:
        st.subheader("Impact Coefficients")
        # Creating a clean table for the coefficients
        coef_data = {
            "Variable": ["Study Hours", "AI Content %", "AI Dependency", "Grade Level", "Uses AI"],
            "Weight": [0.0491, 0.0473, 0.0441, 0.0390, -0.0454]
        }
        coef_df = pd.DataFrame(coef_data)
        st.table(coef_df)

    st.divider()

    # --- VISUALIZING THE IMPACT ---
    st.subheader("Visualizing Feature Importance")
    fig_coef, ax_coef = plt.subplots(figsize=(8, 4))
    
    # Color coding: Green for positive impact, Red for negative
    colors = ['#27AE60' if x > 0 else '#E74C3C' for x in coef_df['Weight']]
    
    sns.barplot(x='Weight', y='Variable', data=coef_df.sort_values('Weight', ascending=False), palette=colors, ax=ax_coef)
    ax_coef.set_title("Which factors actually move the needle?")
    st.pyplot(fig_coef)

    # --- THE "SO WHAT?" SECTION ---
    st.markdown("### 🔍 What does this tell us?")
    
    # Logic based on your negative 'uses_ai' vs positive 'ai_dependency'
    st.warning("""
    **The AI Paradox:** Simply "using AI" (`-0.0454`) has a slight negative correlation with performance. However, **AI Dependency** (`+0.0441`) and **AI Content %** (`+0.0473`) are positive. 
    
    **Translation:** It's not *whether* you use AI, but *how* you use it. Students who integrate AI deeply (high dependency/content) might be using it more effectively than those who just "dabble" in it.
    """)
