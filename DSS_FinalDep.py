import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI 学习效率研究", layout="wide")

# 1. 加载数据
@st.cache_data
def load_data():
    # 请确保您的 GitHub 仓库中 CSV 文件名准确无误
    df = pd.read_csv('ai_impact_student_performance_dataset.csv')
    return df

df = load_data()

# 2. 训练预测模型 (加入出勤率以平衡 AI 指标)
features = ['ai_dependency_score', 'study_hours_per_day', 'ai_generated_content_percentage', 'attendance_percentage']
X = df[features]
y_concept = df['concept_understanding_score']
y_final = df['final_score']

model_concept = LinearRegression().fit(X, y_concept)
model_final = LinearRegression().fit(X, y_final)

# --- 侧边栏：模拟器输入 ---
st.sidebar.header("🕹️ 学习模式模拟器")
st.sidebar.markdown("调整以下参数，观察“AI 悖论”如何运作。")

hrs = st.sidebar.slider("每日手动学习时长", 0.5, 10.0, 3.0, help="指不使用 AI 的阅读、练习或复习时间。")
attendance = st.sidebar.slider("课堂出勤率 (%)", 0, 100, 90, help="这是影响理解力最传统的关键因素。")
ai_dep = st.sidebar.slider("AI 依赖度 (1-10)", 1, 10, 5, help="你在解决问题或寻求解释时对 AI 的依赖程度。")
ai_pct = st.sidebar.slider("AI 生成内容占比 (%)", 0, 100, 30, help="你的作业或论文中 AI 生成的比例。")

# --- 主页面 ---
st.title("📊 AI 与学生表现：效率倍增器研究")

tabs = st.tabs(["交互模拟器", "统计学证明", "相关性分析"])

# 标签页 1：交互故事
with tabs[0]:
    st.header("学习风格模拟器")
    st.write("使用 AI 会降低真实理解力吗？在左侧调整你的配置进行测试。")

    # 模型预测计算
    user_input = np.array([[ai_dep, hrs, ai_pct, attendance]])
    pred_understanding = model_concept.predict(user_input)[0]
    pred_final = model_final.predict(user_input)[0]
    
    # 设定合格线与状态
    pass_status = "✅ 及格 (PASS)" if pred_final >= 50 else "❌ 未及格 (FAIL)"
    
    # 传统基准 (高投入、无 AI 的学生平均分)
    benchmark_score = df[(df['study_hours_per_day'] >= 5) & (df['ai_dependency_score'] < 2)]['final_score'].mean()
    efficiency_gap = (pred_final / benchmark_score) * 100

    # 核心指标展示
    col1, col2, col3 = st.columns(3)
    col1.metric("核心概念理解力", f"{pred_understanding:.2f}/10")
    col2.metric("预测最终得分", f"{pred_final:.1f}%", delta=pass_status, delta_color="normal")
    col3.metric("效率对等率", f"{efficiency_gap:.1f}%")

    # --- 关键：解释卡片 ---
    st.markdown("### 🔍 为什么我的分数是这样？")
    exp1, exp2, exp3 = st.columns(3)
    
    with exp1:
        st.info("**关于理解力 (5.5/10)**\n\n数据证明，单纯增加“学习时长”对理解力的提升有瓶颈。即使你学习 10 小时，如果缺乏互动，理解力也会停留在基准线。这证明 AI 并未让你变笨，而是揭示了‘死记硬背’的局限性。")
    
    with exp2:
        st.info("**关于及格线 (50%)**\n\n在本研究中，50% 即为及格。55% 的得分意味着你已掌握核心，但若想冲击高分，单靠 AI 或时长是不够的，还需要提高出勤率和学习质量。")
    
    with exp3:
        st.info("**效率对等率 (Proficiency Parity)**\n\n**100% 代表你的表现与完全不使用 AI 且每天苦读 5 小时的学生完全一致。** 如果你的得分接近 100%，说明你成功利用 AI 实现了‘降本增效’。")

    # 视觉对比图
    st.divider()
    st.markdown("### 你的配置 vs. 传统高投入模式")
    chart_data = pd.DataFrame({
        "类别": ["当前配置", "传统模式 (5+小时, 无AI)"],
        "最终得分": [pred_final, benchmark_score]
    })
    
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(data=chart_data, x="最终得分", y="类别", palette=["#2E86C1", "#ABB2B9"], ax=ax)
    ax.set_xlim(0, 100)
    st.pyplot(fig)

# 标签页 2：统计学证明
with tabs[1]:
    st.header("数学层面的真相")
    st.write("多元线性回归分析显示：AI 变量对理解力没有显著的负面影响。")
    
    X_stats = sm.add_constant(X)
    stats_model = sm.OLS(y_concept, X_stats).fit()
    st.text(str(stats_model.summary().tables[1]))
    
    st.markdown("> **统计学提示：** 请观察 `P>|t|` 列。如果数值大于 0.05（如 AI 依赖度），则意味着该因素在统计上**不会**导致理解力下降。")

# 标签页 3：热力图
with tabs[2]:
    st.header("变量间的关联性")
    st.write("观察“概念理解力”与“最终得分”的强相关，以及与“AI 依赖度”的极弱相关。")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[features + ['concept_understanding_score', 'final_score']].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
