import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------- 页面基础设置 --------------------------
st.set_page_config(page_title="学业压力智能分析与预警系统", layout="wide")
st.title("面向高校学生的学业压力智能分析与预警系统")
st.markdown("> 大数据赛道创意组竞赛演示Demo | 基于脱敏校园数据，K-Means聚类实现学业压力分级预警")

# -------------------------- 侧边栏 --------------------------
st.sidebar.header("⚙️ 参数设置")
n_cluster = st.sidebar.slider("预警等级划分数量", min_value=2, max_value=4, value=3, step=1)
select_level = st.sidebar.multiselect("筛选预警等级", ["低风险","中风险","高风险","极高风险"], default=["低风险","中风险","高风险"])
st.sidebar.markdown("---")
st.sidebar.info("说明：本Demo使用模拟脱敏数据，仅用于竞赛展示，真实落地严格遵守学生隐私保护规范")

# -------------------------- 生成模拟脱敏数据（这里全部统一60条！） --------------------------
np.random.seed(42)
student_count = 60
data = {
    "学生编号": [f"S{i:03d}" for i in range(1, student_count+1)],
    "绩点下滑幅度": np.random.uniform(0, 2.8, student_count),
    "缺勤次数": np.random.randint(0, 14, student_count),
    "线上学习缺课时长": np.random.uniform(0, 20, student_count),
    "学业压力自评分(1-5)": np.random.randint(1, 6, student_count)
}
df = pd.DataFrame(data)

# -------------------------- 模型计算 --------------------------
feature_cols = ["绩点下滑幅度", "缺勤次数", "线上学习缺课时长", "学业压力自评分(1-5)"]
X = df[feature_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init="auto")
df["cluster"] = kmeans.fit_predict(X_scaled)

# 按簇得分排序，映射风险等级
centers = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)
centers["总分"] = centers.sum(axis=1)
sorted_idx = centers.sort_values("总分").index
level_name = ["低风险","中风险","高风险","极高风险"]
level_map = {}
for i, idx in enumerate(sorted_idx):
    level_map[idx] = level_name[i]
df["压力预警等级"] = df["cluster"].map(level_map)

# -------------------------- 统计卡片 --------------------------
st.subheader("📊 整体风险统计")
col1, col2, col3, col4 = st.columns(4)
total = len(df)
high = len(df[df["压力预警等级"]=="高风险"])
mid = len(df[df["压力预警等级"]=="中风险"])
low = len(df[df["压力预警等级"]=="低风险"])
col1.metric("学生总人数", total)
col2.metric("高风险人数", high)
col3.metric("中风险人数", mid)
col4.metric("低风险人数", low)

# -------------------------- 表格 + 导出Excel --------------------------
st.subheader("📋 学生预警详情表")
show_df = df[df["压力预警等级"].isin(select_level)]
show_table = show_df[["学生编号","绩点下滑幅度","缺勤次数","线上学习缺课时长","学业压力自评分(1-5)","压力预警等级"]]
st.dataframe(show_table, use_container_width=True)

# 导出Excel
def export_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        dataframe.to_excel(writer, index=False, sheet_name="预警名单")
    return output.getvalue()
excel_data = export_excel(show_table)
st.download_button(
    label="📥 导出当前筛选预警名单（Excel）",
    data=excel_data,
    file_name="学业压力预警名单.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# -------------------------- 可视化图表 --------------------------
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("散点图：学业压力分布")
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig1, ax1 = plt.subplots(figsize=(7,5))
    color_dict = {"低风险":"#4CAF50", "中风险":"#FFC107", "高风险":"#F44336", "极高风险":"#9C27B0"}
    for level, c in color_dict.items():
        sub = df[df["压力预警等级"] == level]
        ax1.scatter(sub["绩点下滑幅度"], sub["缺勤次数"], c=c, label=level, s=70)
    ax1.set_xlabel("绩点下滑幅度")
    ax1.set_ylabel("缺勤次数")
    ax1.set_title("学业风险聚类分布")
    ax1.legend()
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

with col_b:
    st.subheader("饼图：各风险占比")
    fig2, ax2 = plt.subplots(figsize=(7,5))
    pie_data = df["压力预警等级"].value_counts()
    colors = ["#4CAF50","#FFC107","#F44336","#9C27B0"]
    ax2.pie(pie_data.values, labels=pie_data.index, autopct="%1.1f%%", colors=colors[:len(pie_data)], startangle=90)
    ax2.set_title("风险等级占比")
    st.pyplot(fig2)

# -------------------------- 项目介绍板块（答辩加分） --------------------------
st.markdown("---")
st.subheader("📖 项目说明与技术方案")
st.markdown("""
1. **项目目标**：融合多源校园学习数据，智能识别学生学业压力，提前预警，辅助辅导员精准帮扶，实现从被动补救转向事前预防。
2. **核心技术**：Python + K-Means无监督聚类算法、数据标准化、特征工程、Web可视化。
3. **数据特征**：绩点波动、考勤记录、线上学习时长、匿名压力自评。
4. **创新点**：客观学习数据+主观问卷结合，可灵活调整预警分级，轻量化、隐私友好，适合高校推广。
""")

