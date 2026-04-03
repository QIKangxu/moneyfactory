# 文件名: app.py
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# 1️⃣ 页面标题
st.set_page_config(page_title="行业拥挤度分析", layout="wide")
st.title("行业拥挤度分析 (近20天滚动)")

# -----------------------------
# 2️⃣ 读取 Excel
file_path = "yongjidu.xlsx"
df = pd.read_excel(file_path, sheet_name="data")

# 假设第一列是日期
df.rename(columns={df.columns[0]: "日期"}, inplace=True)
df.set_index("日期", inplace=True)

# 列名处理
Ashare_col = "成交额-万得全A"
industry_cols = [col for col in df.columns if col.startswith("成交额-")]

# -----------------------------
# 3️⃣ 计算拥挤度
window = 20
Ashare_rolling = df[Ashare_col].rolling(window).sum()

crowding = pd.DataFrame(index=df.index[window-1:])
for col in industry_cols:
    if col == Ashare_col:
        continue
    crowding[col] = df[col].rolling(window).sum() / Ashare_rolling

# -----------------------------
# 4️⃣ 网页侧边栏选择行业
selected_industry = st.sidebar.selectbox(
    "请选择行业查看拥挤度",
    options=[col for col in crowding.columns]
)

# -----------------------------
# 5️⃣ 表格 + 折线图左右并排显示
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{selected_industry} 拥挤度表格（最近20行）")
    st.dataframe(crowding[selected_industry].tail(20))

with col2:
    st.subheader(f"{selected_industry} 拥挤度折线图")
    fig = px.line(
        crowding,
        x=crowding.index,
        y=selected_industry,
        title=f"{selected_industry} 拥挤度折线图（近20天滚动）",
        labels={selected_industry: "拥挤度", "index": "日期"}
    )
    st.plotly_chart(fig, width="stretch")
