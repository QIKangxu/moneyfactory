import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import os

st.sidebar.markdown("### 🔥 版本: 4月10日")
# =============================
# 页面配置（必须放最前面）
# =============================
st.set_page_config(
    page_title="工厂 | 行业分析系统",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# 自定义CSS样式
# =============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    .chart-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .chart-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    .industry-label {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 15px;
    }

    .legend-box {
        background: #f9fafb;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 24px;
        border-left: 4px solid #667eea;
    }

    .stMultiSelect [data-baseweb="select"] {
        border-radius: 10px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
        margin: 2rem 0;
    }

    .stSpinner > div {
        border-top-color: #667eea !important;
    }

    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border: 1px solid #667eea30;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .welcome-container {
        text-align: center;
        padding: 60px 20px;
    }

    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 20px;
    }

    .menu-icon {
        font-size: 3rem;
        margin-bottom: 16px;
    }

    .menu-title {
        color: #1f2937;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# 初始化 session_state
# =============================
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "show_icvr_submenu" not in st.session_state:
    st.session_state.show_icvr_submenu = False


# =============================
# 数据加载函数（Market Overview）
# =============================
@st.cache_data(ttl=3600)
def load_market_overview_data(file_path, sheet_name="ov"):
    """加载市场概览数据（指数行情）"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 获取数据日期（从列名中提取，如"2026/4/3 当日涨跌幅"）
    date_cols = [col for col in df.columns if '当日涨跌幅' in str(col)]
    latest_date_str = "未知日期"

    if date_cols:
        col_str = str(date_cols[0])
        if ' ' in col_str:
            date_part = col_str.split(' ')[0]
            try:
                latest_date = pd.to_datetime(date_part)
                latest_date_str = latest_date.strftime("%Y年%m月%d日")
            except:
                latest_date_str = date_part

    return df, latest_date_str


# =============================
# 数据加载函数（ICVR）
# =============================
@st.cache_data(ttl=3600)
def load_icvr_data(file_path):
    df = pd.read_csv(file_path, header=[0, 1, 2, 3])

    new_columns = []
    for col in df.columns:
        if col[0] == '日期' or str(col[0]).startswith('Unnamed'):
            new_columns.append('日期')
        else:
            new_col = f"{col[0]}_{col[1]}_{col[2]}_{col[3]}"
            new_columns.append(new_col)

    df.columns = new_columns
    df = df.loc[:, ~df.columns.duplicated()]
    df.rename(columns={df.columns[0]: "日期"}, inplace=True)
    df["日期"] = pd.to_datetime(df["日期"])
    df.set_index("日期", inplace=True)

    latest_date = df.index.max()
    latest_date_str = latest_date.strftime("%Y年%m月%d日")

    return df, latest_date_str


# =============================
# 数据加载函数（Earning）
# =============================
@st.cache_data(ttl=3600)
def load_earning_data(file_path):
    df = pd.read_csv(file_path)

    fund_cols = [col for col in df.columns if isinstance(col, str) and len(col) == 8 and col.isdigit()]
    for col in fund_cols:
        if df[col].dtype == object:
            df[col] = df[col].str.replace('%', '').astype(float) / 100

    # 三分类：上调、下调、未调整
    threshold = 0
    df['净利润26E变化'] = df['T日预测2026年净利润中值'] - df['T-1日预测2026年净利润中值']
    df['净利润27E变化'] = df['T日预测2027年净利润中值'] - df['T-1日预测2027年净利润中值']

    df['业绩调整26E'] = df['净利润26E变化'].apply(
        lambda x: '上调' if x > threshold else ('下调' if x < -threshold else '未调整')
    )
    df['业绩调整27E'] = df['净利润27E变化'].apply(
        lambda x: '上调' if x > threshold else ('下调' if x < -threshold else '未调整')
    )

    return df


# =============================
# ICVR核心计算函数
# =============================
def identify_icvr_columns(df, category_filter=None):
    """按分类识别ICVR列"""
    all_cols = df.columns.tolist()
    all_vol_cols = [col for col in all_cols if '_波动率_' in col]

    filtered_vol_cols = []
    filtered_industries = []

    for col in all_vol_cols:
        parts = col.split('_')
        if len(parts) >= 4:
            category = parts[2]
            industry_name = parts[3]
            if category_filter is None or category == category_filter:
                filtered_vol_cols.append(col)
                filtered_industries.append(industry_name)

    Ashare_vol_col = None
    Ashare_name = None
    for col in all_vol_cols:
        parts = col.split('_')
        if len(parts) >= 4 and parts[3] == '万得全A':
            Ashare_vol_col = col
            Ashare_name = '万得全A'
            break

    if Ashare_vol_col is None and len(all_vol_cols) > 0:
        Ashare_vol_col = all_vol_cols[0]
        Ashare_name = all_vol_cols[0].split('_')[-1]

    industry_names = [name for name in filtered_industries if name != Ashare_name]

    amt_dict, ret_dict, vol_dict = {}, {}, {}
    for name in industry_names:
        amt_dict[name] = next((col for col in all_cols if f'_成交额_' in col and col.endswith(f'_{name}')), None)
        ret_dict[name] = next((col for col in all_cols if f'_收益率_' in col and col.endswith(f'_{name}')), None)
        vol_dict[name] = next((col for col in filtered_vol_cols if col.endswith(f'_{name}')), None)

    Ashare_amt_col = next((col for col in all_cols if f'_成交额_' in col and '万得全A' in col), None)
    Ashare_ret_col = next((col for col in all_cols if f'_收益率_' in col and '万得全A' in col), None)

    return {
        "industry_names": industry_names,
        "Ashare_amt_col": Ashare_amt_col,
        "Ashare_ret_col": Ashare_ret_col,
        "amt_dict": amt_dict,
        "ret_dict": ret_dict,
        "vol_dict": vol_dict
    }


def calculate_icvr_indicators(df, col_info, window_crowd=20, window_ret=55):
    """计算ICVR指标"""
    industry_names = col_info["industry_names"]
    Ashare_amt_col = col_info["Ashare_amt_col"]
    Ashare_ret_col = col_info["Ashare_ret_col"]
    amt_dict = col_info["amt_dict"]
    ret_dict = col_info["ret_dict"]
    vol_dict = col_info["vol_dict"]

    if Ashare_amt_col is None or Ashare_amt_col not in df.columns:
        raise ValueError(f"万得全A成交额列未找到")
    if Ashare_ret_col is None or Ashare_ret_col not in df.columns:
        raise ValueError(f"万得全A收益率列未找到")

    Ashare_rolling = df[Ashare_amt_col].rolling(window_crowd).sum()
    crowding = pd.DataFrame(index=df.index[window_crowd - 1:])

    for name in industry_names:
        amt_col = amt_dict.get(name)
        if amt_col and amt_col in df.columns:
            crowding[name] = df[amt_col].rolling(window_crowd).sum() / Ashare_rolling
        else:
            crowding[name] = 0

    relative_returns = pd.DataFrame(index=df.index)
    for name in industry_names:
        ret_col = ret_dict.get(name)
        if ret_col and ret_col in df.columns:
            relative_returns[name] = df[ret_col] - df[Ashare_ret_col]
        else:
            relative_returns[name] = 0
    relative_returns_sum = relative_returns.rolling(window_ret).sum()

    volatility = pd.DataFrame(index=crowding.index)
    for name in industry_names:
        vol_col = vol_dict.get(name)
        if vol_col and vol_col in df.columns:
            volatility[name] = df[vol_col].iloc[window_crowd - 1:]
        else:
            volatility[name] = 0

    return crowding, relative_returns_sum, volatility


def standardize_icvr_data(crowding, volatility, relative_returns_sum, industry_names,
                          window_crowd=20, window_ret=55):
    """ICVR数据标准化"""
    scaled_data = {}
    max_window = max(window_crowd, window_ret)

    for name in industry_names:
        tmp = pd.DataFrame({
            "拥挤度": crowding[name],
            "超额收益": relative_returns_sum[name].iloc[window_crowd - 1:],
            "波动率": volatility[name]
        })

        tmp_valid = tmp.iloc[max_window - 1:]
        tmp_valid = tmp_valid.replace([float('inf'), -float('inf')], pd.NA).dropna()
        if tmp_valid.empty:
            continue

        result = pd.DataFrame(index=tmp_valid.index)
        scaler = MinMaxScaler()
        result["拥挤度"] = scaler.fit_transform(tmp_valid[["拥挤度"]])
        result["波动率"] = scaler.fit_transform(tmp_valid[["波动率"]])
        result["超额收益"] = tmp_valid["超额收益"]
        scaled_data[name] = result

    return scaled_data


def create_icvr_chart(name, data, params_label, height=320):
    """创建ICVR图表"""
    fig = go.Figure()

    colors = {
        '拥挤度': '#6366f1',
        '波动率': '#10b981',
        '超额收益': '#f43f5e'
    }

    fig.add_trace(go.Scatter(
        x=data.index, y=data["拥挤度"], mode='lines',
        name='拥挤度', line=dict(color=colors['拥挤度'], width=1),
        yaxis='y', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>拥挤度: %{y:.2%}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data["波动率"], mode='lines',
        name='波动率', line=dict(color=colors['波动率'], width=1),
        yaxis='y', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>波动率: %{y:.2%}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data["超额收益"], mode='lines',
        name='超额收益', line=dict(color=colors['超额收益'], width=1),
        yaxis='y2', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>超额收益: %{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{name}</b>",
            font=dict(size=16, color='#1f2937'),
            x=0.5, xanchor='center',
            y=0.98, yanchor='top'
        ),
        annotations=[dict(
            text=f"<span style='color:#6b7280;font-size:11px;'>{params_label}</span>",
            xref='paper', yref='paper',
            x=0, y=1.12,
            xanchor='left', yanchor='top',
            showarrow=False
        )],
        xaxis=dict(
            showline=True, linecolor='#e5e7eb', linewidth=1,
            tickfont=dict(color='#6b7280', size=10),
            zeroline=False, showgrid=True, gridcolor='#f3f4f6',
            tickformat='%Y-%m'
        ),
        yaxis=dict(
            side='left', range=[0, 1], tickmode='array', tickvals=[0, 0.5, 1],
            ticktext=['0%', '50%', '100%'], tickfont=dict(color='#6b7280', size=10),
            showgrid=True, gridcolor='#f3f4f6',
            showline=True, linecolor='#e5e7eb', linewidth=1, zeroline=False,
            title=dict(text='拥挤度/波动率', font=dict(size=10, color='#9ca3af'))
        ),
        yaxis2=dict(
            overlaying='y', side='right', tickformat='.0%',
            tickfont=dict(color='#6b7280', size=10),
            showline=True, linecolor='#e5e7eb', linewidth=1,
            showgrid=False, zeroline=False,
            title=dict(text='超额收益', font=dict(size=10, color='#9ca3af'))
        ),
        legend=dict(
            orientation='h',
            yanchor='top', y=1.12,
            xanchor='right', x=1,
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
            font=dict(size=10),
            itemsizing='constant'
        ),
        margin=dict(l=60, r=60, t=80, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        hovermode='x unified'
    )
    return fig


# =============================
# 页面渲染函数
# =============================

def render_sidebar():
    """渲染侧边栏导航"""
    st.sidebar.title("🏭 工厂")

    # 添加清除缓存按钮
    if st.sidebar.button("🔄 刷新数据"):
        st.cache_data.clear()
        st.rerun()

    # 一级菜单：大盘速览
    with st.sidebar.expander("📊 大盘速览", expanded=False):
        # 修改：市场概览改为指数复盘，删除指数行情
        if st.button("📈 指数复盘", key="nav_market_overview", use_container_width=True):
            st.session_state.page = "market_overview"
            st.session_state.show_icvr_submenu = False

    # 一级菜单：行业状态 - ICVR（默认折叠）
    with st.sidebar.expander("🔥 行业状态", expanded=False):
        # 二级菜单：ICVR 主按钮（紫色）
        if st.button("🔥 拥挤、波动与超额（ICVR）", key="nav_icvr_main",
                     use_container_width=True, type="primary"):
            st.session_state.show_icvr_submenu = not st.session_state.show_icvr_submenu
            st.rerun()

        # 三级菜单（子菜单）- 紧凑缩进，默认灰色按钮（与紫色区分）
        if st.session_state.show_icvr_submenu:
            # 用极窄列减少留白
            _, btn_col = st.columns([0.08, 0.92])
            with btn_col:
                # 三级菜单：默认按钮样式（灰色），与二级紫色明显区分
                if st.button("📈 一级行业概览", key="nav_icvr_overview", use_container_width=True):
                    st.session_state.page = "icvr_overview"
                if st.button("🔍 细分行业筛选", key="nav_icvr_filter", use_container_width=True):
                    st.session_state.page = "icvr_filter"

    # 一级菜单：发现牛牛 - 业绩上修
    with st.sidebar.expander("🐮 发现牛牛", expanded=False):
        if st.button("📊 业绩上修", key="nav_earning", use_container_width=True, type="primary"):
            st.session_state.page = "earning_revision"
            st.session_state.show_icvr_submenu = False

    # 底部信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style='text-align:center;color:#9ca3af;font-size:0.75rem;'>
            系统版本 v1.4<br>
            © 2026 工厂
        </div>
    """, unsafe_allow_html=True)


def render_welcome():
    """初始欢迎界面 - 简化版"""
    st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">🏭</div>
            <h1 class="main-title">欢迎使用工厂</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div style='background:white;padding:32px 24px;border-radius:16px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);text-align:center;'>
                <div class="menu-icon">📊</div>
                <div class="menu-title">大盘速览</div>
            </div>
        """, unsafe_allow_html=True)
        # 修改：大盘速览功能已上线
        if st.button("进入", key="welcome_market", use_container_width=True):
            st.session_state.page = "market_overview"
            st.rerun()

    with col2:
        st.markdown("""
            <div style='background:white;padding:32px 24px;border-radius:16px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);text-align:center;'>
                <div class="menu-icon">🔥</div>
                <div class="menu-title">行业状态</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("进入", key="welcome_icvr", use_container_width=True):
            st.session_state.show_icvr_submenu = True
            st.session_state.page = "icvr_overview"
            st.rerun()

    with col3:
        st.markdown("""
            <div style='background:white;padding:32px 24px;border-radius:16px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);text-align:center;'>
                <div class="menu-icon">🐮</div>
                <div class="menu-title">发现牛牛</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("进入", key="welcome_earning", use_container_width=True):
            st.session_state.page = "earning_revision"
            st.rerun()


def render_market_overview():
    """大盘速览 - 市场概览"""
    st.markdown(f'<h1 class="main-title">📈 市场概览</h1>', unsafe_allow_html=True)

    try:
        # 只读取第2行作为列名
        df = pd.read_csv(
            st.session_state.data_paths['market_overview_file'],
            header=1
        )

        df_header = pd.read_csv(
            st.session_state.data_paths['market_overview_file'],
            header=None,
            nrows=1
        )

        latest_date_str = "未知日期"
        for val in df_header.iloc[0]:
            if pd.notna(val) and isinstance(val, (str, pd.Timestamp)):
                val_str = str(val)
                if '2026' in val_str or '2025' in val_str:
                    try:
                        if isinstance(val, pd.Timestamp):
                            latest_date_str = val.strftime("%Y年%m月%d日")
                        else:
                            date_obj = pd.to_datetime(val)
                            latest_date_str = date_obj.strftime("%Y年%m月%d日")
                    except:
                        latest_date_str = val_str
                    break

        st.markdown(f'<p class="subtitle">数据截至 {latest_date_str}</p>', unsafe_allow_html=True)

        # 自动识别列名
        index_code_col = None
        index_name_col = None
        category_col = None

        for col in df.columns:
            col_str = str(col).strip()
            if any(x in col_str for x in ['指数代码', '代码']):
                index_code_col = col
            if any(x in col_str for x in ['指数名称', '名称']):
                index_name_col = col
            if any(x in col_str for x in ['板块', '所属板块', '指数所属板块']):
                category_col = col

        if category_col is None and len(df.columns) >= 1:
            category_col = df.columns[0]
        if index_code_col is None and len(df.columns) >= 2:
            index_code_col = df.columns[1]
        if index_name_col is None and len(df.columns) >= 3:
            index_name_col = df.columns[2]

        change_col = [c for c in df.columns if '当日涨跌幅' in str(c)]
        change_col = change_col[0] if change_col else None

        # ========== 板块分类筛选器（默认显示全部）==========
        if category_col and category_col in df.columns:
            categories = df[category_col].dropna().unique().tolist()
            st.markdown("### 🗂️ 板块筛选")

            # 修改：默认不选择任何分类，显示全部
            selected_categories = st.multiselect(
                "选择板块分类（不选则显示全部）",
                options=categories,
                default=[],  # 默认空，显示全部
                placeholder="请选择板块...",
                key="category_filter"
            )
            if selected_categories:
                df_filtered = df[df[category_col].isin(selected_categories)].copy()
            else:
                df_filtered = df.copy()
        else:
            df_filtered = df.copy()

        # ========== 主要指数卡片 - 显示筛选板块中涨最多和跌最多的3个 ==========
        st.markdown("---")

        # 获取当日涨跌幅列
        change_col = [c for c in df.columns if '当日涨跌幅' in str(c)]
        change_col = change_col[0] if change_col else None

        if change_col and index_name_col:
            # 按涨跌幅排序
            df_sorted = df_filtered.sort_values(by=change_col, ascending=False)

            # 获取涨最多的3个和跌最多的3个
            top3_up = df_sorted.head(3)  # 涨最多
            top3_down = df_sorted.tail(3)  # 跌最多

            # 合并并去重（防止数据少于6个时重复）
            display_indices = pd.concat([top3_up, top3_down]).drop_duplicates(subset=[index_name_col])

            # 显示为卡片
            cols = st.columns(min(len(display_indices), 6))

            for idx, (_, row) in enumerate(display_indices.iterrows()):
                if idx >= 6:
                    break

                index_name = row[index_name_col]
                change_val = row[change_col]

                # 红涨绿跌
                color = "🔴" if change_val > 0 else "🟢" if change_val < 0 else "⚪"

                with cols[idx]:
                    st.metric(
                        label=f"{color} {index_name}",
                        value=f"{change_val:+.2f}%"
                    )
        else:
            # 如果无法获取涨跌幅列，显示默认的4个主要指数
            main_indices = ['万得全A', '上证指数', '创业板指', '科创50']
            cols = st.columns(4)
            for idx, index_name in enumerate(main_indices):
                row = df_filtered[df_filtered[index_name_col] == index_name]
                if not row.empty and change_col:
                    change_val = row[change_col].values[0]
                    color = "🔴" if change_val > 0 else "🟢" if change_val < 0 else "⚪"
                    with cols[idx]:
                        st.metric(label=f"{color} {index_name}", value=f"{change_val:+.2f}%")

        st.markdown("---")

        # ========== 格式化数据表格 ==========
        display_df = df_filtered.copy()

        for col in display_df.columns:
            col_str = str(col)
            # 修改1：添加"近一周"、"近一月"关键词
            if any(kw in col_str for kw in ['涨跌幅', '近5日', '近20日', '近一周', '近一月', '年初至今', '上年全年']):
                display_df[col] = display_df[col].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
            elif any(kw in col_str for kw in ['PE', 'PB', '股息率', '百分位数', '分位数']):
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

        # ========== 按分类分组显示表格 ==========
        st.markdown("### 📊 指数行情数据")

        # 隐藏分类列和代码列，只保留指数名称和数据列
        exclude_cols = []
        if category_col:
            exclude_cols.append(category_col)
        if index_code_col:
            exclude_cols.append(index_code_col)

        display_cols = [c for c in display_df.columns if c not in exclude_cols]
        show_df = display_df[display_cols].copy()

        # ========== 关键修改：计算每列的最大值用于相对条形图 ==========
        # 获取原始数值（格式化前的）用于计算最大值
        raw_df = df_filtered[[c for c in display_cols if c in df_filtered.columns]].copy()

        # 计算每列的最大绝对值
        col_max_values = {}
        for col in raw_df.columns:
            col_str = str(col)
            # 修改2：添加"近一周"、"近一月"关键词
            if any(kw in col_str for kw in ['涨跌幅', '近5日', '近20日', '近一周', '近一月', '年初至今', '上年全年']):
                max_val = raw_df[col].abs().max()
                col_max_values[col] = max_val if max_val > 0 else 1

        # ========== 红涨绿跌 + 相对条形图 ==========
        def colorize_value(val, col_name, is_percent=True):
            """红涨绿跌 + 相对条形图"""
            if not isinstance(val, str) or val == "-":
                return val, 0
            try:
                num_str = val.replace('%', '').replace('+', '')
                num = float(num_str)

                if is_percent and col_name in col_max_values:
                    # 使用当前筛选数据的最大值计算条形图宽度
                    max_val = col_max_values[col_name]
                    bar_width = min(abs(num) / max_val * 100, 100) if max_val > 0 else 0
                    bar_color = "#ef4444" if num > 0 else "#10b981" if num < 0 else "#6b7280"

                    # 创建条形图HTML
                    bar_html = f'<div style="display:inline-flex;align-items:center;width:100%;"><div style="display:inline-block;width:50px;height:6px;background:#e5e7eb;border-radius:3px;overflow:hidden;margin-right:6px;flex-shrink:0;"><div style="width:{bar_width}%;height:100%;background:{bar_color};"></div></div>'

                    if num > 0:
                        return f'{bar_html}<span style="color:#ef4444;font-weight:600;">{val}</span></div>', num
                    elif num < 0:
                        return f'{bar_html}<span style="color:#10b981;font-weight:600;">{val}</span></div>', num
                    else:
                        return f'{bar_html}<span style="color:#6b7280;">{val}</span></div>', num
                else:
                    return val, 0
            except:
                return val, 0

        # 对百分比列应用颜色和条形图
        styled_df = show_df.copy()
        for col in styled_df.columns:
            col_str = str(col)
            # 修改3：添加"近一周"、"近一月"关键词
            if any(kw in col_str for kw in ['涨跌幅', '近5日', '近20日', '近一周', '近一月', '年初至今', '上年全年']):
                styled_df[col] = styled_df[col].apply(lambda x: colorize_value(x, col, True)[0])

        # ========== 固定列宽（恢复原来的设置）==========
        st.markdown("""
        <style>
        .market-table-container { overflow-x: auto; }
        .market-table {
            font-size: 13px;
            border-collapse: collapse;
            width: 100%;
            table-layout: fixed;
        }
        .market-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 6px;
            text-align: center;
            font-weight: 600;
            border: none;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .market-table td {
            padding: 8px 6px;
            border-bottom: 1px solid #e5e7eb;
            text-align: center;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .market-table tr:nth-child(even) { background-color: #f9fafb; }
        .market-table tr:hover { background-color: #f3f4f6; }
        .market-table td:first-child {
            font-weight: 600;
            color: #374151;
            text-align: left;
        }
        .market-table th:nth-child(1), .market-table td:nth-child(1) { width: 90px; }
        .market-table th:nth-child(2), .market-table td:nth-child(2) { width: 130px; }
        .market-table th:nth-child(3), .market-table td:nth-child(3) { width: 130px; }
        .market-table th:nth-child(4), .market-table td:nth-child(4) { width: 100px; }
        .market-table th:nth-child(5), .market-table td:nth-child(5) { width: 100px; }
        .market-table th:nth-child(6), .market-table td:nth-child(6) { width: 100px; }
        .market-table th:nth-child(7), .market-table td:nth-child(7) { width: 100px; }
        .market-table th:nth-child(8), .market-table td:nth-child(8) { width: 80px; }
        .market-table th:nth-child(9), .market-table td:nth-child(9) { width: 90px; }
        .market-table th:nth-child(10), .market-table td:nth-child(10) { width: 80px; }
        .market-table th:nth-child(11), .market-table td:nth-child(11) { width: 90px; }
        .market-table th:nth-child(12), .market-table td:nth-child(12) { width: 70px; }
        .market-table th:nth-child(13), .market-table td:nth-child(13) { width: 100px; }
        </style>
        """, unsafe_allow_html=True)

        # ========== 按分类分组显示 ==========
        if category_col and category_col in df_filtered.columns:
            # 获取所有分类（保持原始顺序）
            categories_in_data = df_filtered[category_col].dropna().unique().tolist()

            for category in categories_in_data:
                # 显示分类标题
                st.markdown(f"""
                <div style="margin-top: 24px; margin-bottom: 12px; padding: 8px 16px; 
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white; border-radius: 8px; font-weight: 600; 
                            display: inline-block; font-size: 14px;">
                    {category}
                </div>
                """, unsafe_allow_html=True)

                # 获取该分类的数据
                category_df = styled_df[df_filtered[category_col] == category].copy()

                # 生成该分类的表格
                html_table = category_df.to_html(index=False, escape=False, classes='market-table')
                st.markdown(f'<div class="market-table-container">{html_table}</div>', unsafe_allow_html=True)
        else:
            # 没有分类列，直接显示全部
            html_table = styled_df.to_html(index=False, escape=False, classes='market-table')
            st.markdown(f'<div class="market-table-container">{html_table}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"市场概览数据加载失败: {e}")
        import traceback
        st.error(traceback.format_exc())


def render_index_quote():
    """大盘速览 - 指数行情（预留）"""
    st.markdown(f'<h1 class="main-title">📊 指数行情</h1>', unsafe_allow_html=True)
    st.info("🚧 功能开发中，敬请期待...")

def render_icvr_overview(df, latest_date_str):
    """ICVR - 一级行业概览"""
    st.markdown(f'<h1 class="main-title">🔥 ICVR 一级行业概览</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">数据截至 {latest_date_str}</p>', unsafe_allow_html=True)

    st.markdown("""
        <div class="legend-box">
            <span style='font-weight:600;color:#374151;'>ICVR 图例：</span>
            <span style='color:#6366f1;font-weight:500;'>● 拥挤度(C)</span>（标准化）
            <span style='margin:0 8px;color:#d1d5db;'>|</span>
            <span style='color:#10b981;font-weight:500;'>● 波动率(V)</span>（标准化）
            <span style='margin:0 8px;color:#d1d5db;'>|</span>
            <span style='color:#f43f5e;font-weight:500;'>● 超额收益(R)</span>（右轴，绝对值）
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("正在计算 ICVR 指标..."):
        col_info = identify_icvr_columns(df, category_filter="一级行业")

        if len(col_info["industry_names"]) == 0:
            st.error("未找到一级行业数据，请检查数据源")
            return

        c15, r15, v20 = calculate_icvr_indicators(df, col_info, 15, 15)
        data_15 = standardize_icvr_data(c15, v20, r15, col_info["industry_names"], 15, 15)

        c20, r55, v20 = calculate_icvr_indicators(df, col_info, 20, 55)
        data_20 = standardize_icvr_data(c20, v20, r55, col_info["industry_names"], 20, 55)

    total_count = len([n for n in col_info["industry_names"] if n in data_15 and n in data_20])
    st.markdown(f"<p style='color:#6b7280;margin-bottom:20px;'>共 <b>{total_count}</b> 个一级行业</p>",
                unsafe_allow_html=True)

    for name in col_info["industry_names"]:
        if name not in data_15 or name not in data_20:
            continue

        st.markdown(f"""
            <div class="chart-card">
                <span class="industry-label">{name}</span>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            fig_15 = create_icvr_chart(name, data_15[name], "C:15天 | V:20天 | R:15天")
            st.plotly_chart(fig_15, use_container_width=True, key=f"icvr_{name}_15")

        with col2:
            fig_20 = create_icvr_chart(name, data_20[name], "C:20天 | V:20天 | R:55天")
            st.plotly_chart(fig_20, use_container_width=True, key=f"icvr_{name}_20")

        st.markdown("<hr>", unsafe_allow_html=True)


def render_icvr_filter(df, latest_date_str):
    """ICVR - 细分行业筛选（四种筛选方式：股票、一级行业、细分行业、历史分位数）"""
    st.markdown(f'<h1 class="main-title">🔍 ICVR 细分行业筛选</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">数据截至 {latest_date_str}</p>', unsafe_allow_html=True)

    # ========== 加载数据（分别处理股票映射和一级行业映射）==========
    try:
        stock_info_df = pd.read_csv(
            st.session_state.data_paths['stock_info_file']
        )

        # 1. 股票筛选映射：A列(代码) -> C列(细分行业)
        stock_to_industry = {}
        for _, row in stock_info_df.iterrows():
            code = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""  # A列：证券代码
            name = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""  # B列：证券简称
            sub_industry = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""  # C列：细分行业

            if code and sub_industry and sub_industry.lower() not in ['nan', 'none']:
                stock_to_industry[code] = sub_industry
            if name and sub_industry and sub_industry.lower() not in ['nan', 'none']:
                stock_to_industry[name] = sub_industry

        # 2. 一级行业筛选映射：E列(一级行业) -> F列(细分行业)
        primary_to_sub = {}
        for _, row in stock_info_df.iterrows():
            primary_industry = str(row.iloc[4]).strip() if pd.notna(row.iloc[4]) else ""  # E列：一级行业
            sub_industry = str(row.iloc[5]).strip() if pd.notna(row.iloc[5]) else ""  # F列：细分行业

            if (
                primary_industry and sub_industry
                and primary_industry.lower() not in ['nan', 'none']
                and sub_industry.lower() not in ['nan', 'none']
            ):
                if primary_industry not in primary_to_sub:
                    primary_to_sub[primary_industry] = set()
                primary_to_sub[primary_industry].add(sub_industry)

        # 转换为排序列表
        for key in primary_to_sub:
            primary_to_sub[key] = sorted(list(primary_to_sub[key]))

    except Exception as e:
        stock_info_df = None
        stock_to_industry = {}
        primary_to_sub = {}
        st.error(f"加载行业映射数据失败: {e}")

    col_info_all = identify_icvr_columns(df, category_filter="细分行业")
    industry_options = col_info_all["industry_names"]

    if len(industry_options) == 0:
        st.error("未找到细分行业数据，请检查数据源")
        return

    st.markdown(
        f"<p style='color:#6b7280;margin-bottom:16px;'>系统共有 <b>{len(industry_options)}</b> 个细分行业</p>",
        unsafe_allow_html=True
    )

    # ========== 初始化 session_state ==========
    if "filter_mode" not in st.session_state:
        st.session_state.filter_mode = None
    if "selected_industries" not in st.session_state:
        st.session_state.selected_industries = []
    if "show_percentile_filter" not in st.session_state:
        st.session_state.show_percentile_filter = False
    if "applied_percentile_count" not in st.session_state:
        st.session_state.applied_percentile_count = 0

    # 分位数筛选默认值
    if "pct_crowd_range" not in st.session_state:
        st.session_state.pct_crowd_range = (0, 100)
    if "pct_ret_range" not in st.session_state:
        st.session_state.pct_ret_range = (0, 100)
    if "pct_vol_range" not in st.session_state:
        st.session_state.pct_vol_range = (0, 100)

    # ========== 1. 股票搜索（A-C列，最上面）==========
    st.markdown("### 🔍 股票搜索")

    if stock_info_df is not None and not stock_info_df.empty:
        search_options = []
        for _, row in stock_info_df.iterrows():
            code = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""  # A列
            name = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""  # B列
            if code and name:
                search_options.append(f"{code} {name}")

        selected_stock = st.selectbox(
            "输入股票代码或名称搜索（选中后自动显示该股票所属行业）",
            options=[""] + sorted(list(set(search_options))),
            index=0,
            key="stock_search"
        )

        if selected_stock:
            stock_code = selected_stock.split(' ')[0]
            if stock_code in stock_to_industry:
                industry = stock_to_industry[stock_code]  # 来自C列
                if industry in industry_options:
                    st.session_state.filter_mode = "stock"
                    st.session_state.selected_industries = [industry]
                    st.success(f"✅ 股票 **{selected_stock}** 所属行业：**{industry}**")
                else:
                    st.warning(f"⚠️ 该股票所属行业 **{industry}** 暂无ICVR数据")
            else:
                st.error("❌ 未找到股票对应的行业信息")

    st.markdown("---")

    # ========== 2. 一级行业筛选（E-F列）==========
    st.markdown("### 🏭 一级行业筛选")

    if primary_to_sub:
        primary_list = sorted(list(primary_to_sub.keys()))
        primary_options = ["请选择一级行业..."] + primary_list

        selected_primary = st.selectbox(
            "选择一级行业（选中后自动显示该行业下所有细分行业）",
            options=primary_options,
            index=0,
            key="primary_industry_select"
        )

        if selected_primary != "请选择一级行业...":
            sub_under_primary = set(primary_to_sub[selected_primary])  # 来自F列
            matched_industries = [ind for ind in industry_options if ind in sub_under_primary]

            if len(matched_industries) == 0:
                st.warning(f"⚠️ **{selected_primary}** 下的细分行业暂无ICVR数据")
                st.session_state.selected_industries = []
            else:
                st.session_state.filter_mode = "primary"
                st.session_state.selected_industries = matched_industries
                st.success(f"✅ **{selected_primary}** 下共找到 **{len(matched_industries)}** 个细分行业有ICVR数据")

                with st.expander(f"查看 {selected_primary} 下的 {len(matched_industries)} 个细分行业"):
                    st.write(matched_industries)
        else:
            if st.session_state.filter_mode == "primary":
                st.session_state.filter_mode = None
                st.session_state.selected_industries = []

    st.markdown("---")

    # ========== 3. 细分行业多选 ==========
    st.markdown("### 📊 细分行业筛选")

    manual_selected = st.multiselect(
        "直接选择细分行业（支持多选，选择后覆盖上方筛选）",
        options=industry_options,
        default=[] if st.session_state.filter_mode != "manual" else st.session_state.selected_industries,
        placeholder="请选择细分行业...",
        key="manual_industry_select"
    )

    if manual_selected:
        st.session_state.filter_mode = "manual"
        st.session_state.selected_industries = manual_selected
        st.session_state.show_percentile_filter = False

    st.markdown("---")

    # ========== 4. 历史分位数筛选 ==========
    st.markdown("### 📈 历史分位数筛选")

    show_percentile = st.checkbox(
        "启用历史分位数筛选（基于最新日期的拥挤度、超额收益、波动率历史分位数）",
        value=st.session_state.show_percentile_filter,
        key="show_percentile_checkbox"
    )
    st.session_state.show_percentile_filter = show_percentile

    if st.session_state.filter_mode == "percentile" and not show_percentile:
        st.session_state.show_percentile_filter = True
        show_percentile = True
        st.rerun()

    if show_percentile:
        @st.cache_data(ttl=3600)
        def calculate_latest_percentiles_cached(df_cached, col_info_cached):
            return calculate_latest_percentiles(df_cached, col_info_cached)

        def apply_quick_filter(crowd_range, ret_range, vol_range):
            st.session_state.pct_crowd_range = crowd_range
            st.session_state.pct_ret_range = ret_range
            st.session_state.pct_vol_range = vol_range
            st.session_state.show_percentile_filter = True

        try:
            with st.spinner("正在计算历史分位数..."):
                percentile_df = calculate_latest_percentiles_cached(df, col_info_all)
        except Exception as e:
            st.error(f"分位数计算失败: {e}")
            percentile_df = pd.DataFrame()

        if not percentile_df.empty:
            st.caption("设置各指标的历史分位数范围（0-100），筛选出符合条件的细分行业")

            # ========== 快捷筛选按钮 ==========
            st.markdown("#### 🚀 快捷筛选")
            quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)

            with quick_col1:
                if st.button("📉 低超额（<5%）", use_container_width=True, key="btn_low_ret"):
                    apply_quick_filter((0, 100), (0, 5), (0, 100))
                    st.rerun()

            with quick_col2:
                if st.button("📉 低拥挤+低超额（<5%）", use_container_width=True, key="btn_low_all"):
                    apply_quick_filter((0, 5), (0, 5), (0, 5))
                    st.rerun()

            with quick_col3:
                if st.button("📈 高超额（>90%）", use_container_width=True, key="btn_high_ret"):
                    apply_quick_filter((0, 100), (90, 100), (0, 100))
                    st.rerun()

            with quick_col4:
                if st.button("🔥 高拥挤+高超额（>90%）", use_container_width=True, key="btn_high_all"):
                    apply_quick_filter((90, 100), (90, 100), (90, 100))
                    st.rerun()

            # ========== 三个 slider ==========
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**拥挤度分位数**")
                st.slider(
                    "范围",
                    min_value=0,
                    max_value=100,
                    key="pct_crowd_range"
                )

            with col2:
                st.markdown("**超额收益分位数**")
                st.slider(
                    "范围",
                    min_value=0,
                    max_value=100,
                    key="pct_ret_range"
                )

            with col3:
                st.markdown("**波动率分位数**")
                st.slider(
                    "范围",
                    min_value=0,
                    max_value=100,
                    key="pct_vol_range"
                )

            # ========== 应用分位数筛选 ==========
            filtered_percentile = percentile_df[
                (percentile_df['拥挤度分位数'] >= st.session_state.pct_crowd_range[0]) &
                (percentile_df['拥挤度分位数'] <= st.session_state.pct_crowd_range[1]) &
                (percentile_df['超额收益分位数'] >= st.session_state.pct_ret_range[0]) &
                (percentile_df['超额收益分位数'] <= st.session_state.pct_ret_range[1]) &
                (percentile_df['波动率分位数'] >= st.session_state.pct_vol_range[0]) &
                (percentile_df['波动率分位数'] <= st.session_state.pct_vol_range[1])
            ]

            st.markdown(f"**分位数筛选结果：共 {len(filtered_percentile)} 个行业**")

            if len(filtered_percentile) > 0:
                display_pct_df = filtered_percentile[
                    ['细分行业', '拥挤度分位数', '超额收益分位数', '波动率分位数']
                ].copy()
                display_pct_df['拥挤度分位数'] = display_pct_df['拥挤度分位数'].apply(lambda x: f"{x:.1f}%")
                display_pct_df['超额收益分位数'] = display_pct_df['超额收益分位数'].apply(lambda x: f"{x:.1f}%")
                display_pct_df['波动率分位数'] = display_pct_df['波动率分位数'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_pct_df, use_container_width=True, hide_index=True)

            st.caption(
                f"当前筛选范围 | "
                f"拥挤度: {st.session_state.pct_crowd_range[0]}-{st.session_state.pct_crowd_range[1]}% | "
                f"超额收益: {st.session_state.pct_ret_range[0]}-{st.session_state.pct_ret_range[1]}% | "
                f"波动率: {st.session_state.pct_vol_range[0]}-{st.session_state.pct_vol_range[1]}%"
            )

            if len(filtered_percentile) > 0:
                current_count = len(filtered_percentile)
                is_already_applied = (
                    st.session_state.filter_mode == "percentile"
                    and st.session_state.applied_percentile_count == current_count
                    and len(st.session_state.selected_industries) == current_count
                )

                if not is_already_applied:
                    if st.button("✅ 应用分位数筛选结果（覆盖上方选择）", type="primary", key="apply_percentile"):
                        st.session_state.filter_mode = "percentile"
                        st.session_state.selected_industries = filtered_percentile['细分行业'].tolist()
                        st.session_state.show_percentile_filter = True
                        st.session_state.applied_percentile_count = len(filtered_percentile)
                        st.rerun()
                else:
                    st.success(f"✅ 已应用分位数筛选结果：共 {current_count} 个行业")
            else:
                st.warning("⚠️ 没有符合分位数条件的行业，无法应用筛选")

        st.markdown("---")

    # ========== 显示图表 ==========
    all_selected = st.session_state.selected_industries

    if not all_selected:
        st.markdown("""
            <div class="info-box">
                <div style='font-size:2rem;margin-bottom:12px;'>👆</div>
                <p style='color:#6b7280;'>请通过以下任一方式筛选：</p>
                <p style='color:#9ca3af;font-size:0.9rem;margin-top:8px;'>
                    1. 股票搜索 | 2. 一级行业筛选 | 3. 细分行业多选 | 4. 历史分位数筛选
                </p>
            </div>
        """, unsafe_allow_html=True)
        return

    if st.session_state.filter_mode == "stock":
        st.caption(f"当前模式：股票搜索(A-C列) | 显示行业：{', '.join(all_selected)}")
    elif st.session_state.filter_mode == "primary":
        st.caption(f"当前模式：一级行业筛选(E-F列) | 共 {len(all_selected)} 个细分行业")
    elif st.session_state.filter_mode == "manual":
        st.caption(f"当前模式：手动选择 | 共 {len(all_selected)} 个细分行业")
    elif st.session_state.filter_mode == "percentile":
        st.caption(f"当前模式：历史分位数筛选 | 共 {len(all_selected)} 个细分行业")

    # 图表渲染
    st.markdown("""
        <div class="legend-box">
            <span style='font-weight:600;color:#374151;'>ICVR 图例：</span>
            <span style='color:#6366f1;font-weight:500;'>● 拥挤度(C)</span>（标准化）
            <span style='margin:0 8px;color:#d1d5db;'>|</span>
            <span style='color:#10b981;font-weight:500;'>● 波动率(V)</span>（标准化）
            <span style='margin:0 8px;color:#d1d5db;'>|</span>
            <span style='color:#f43f5e;font-weight:500;'>● 超额收益(R)</span>（右轴，绝对值）
        </div>
    """, unsafe_allow_html=True)

    with st.spinner(f"正在计算 {len(all_selected)} 个行业的 ICVR 指标..."):
        col_info = {
            "industry_names": all_selected,
            "Ashare_amt_col": col_info_all["Ashare_amt_col"],
            "Ashare_ret_col": col_info_all["Ashare_ret_col"],
            "amt_dict": {k: v for k, v in col_info_all["amt_dict"].items() if k in all_selected},
            "ret_dict": {k: v for k, v in col_info_all["ret_dict"].items() if k in all_selected},
            "vol_dict": {k: v for k, v in col_info_all["vol_dict"].items() if k in all_selected},
        }

        try:
            c15, r15, v20 = calculate_icvr_indicators(df, col_info, 15, 15)
            data_15 = standardize_icvr_data(c15, v20, r15, all_selected, 15, 15)

            c20, r55, v20 = calculate_icvr_indicators(df, col_info, 20, 55)
            data_20 = standardize_icvr_data(c20, v20, r55, all_selected, 20, 55)

        except Exception as e:
            st.error(f"计算出错: {e}")
            return

    for name in all_selected:
        if name not in data_15 or name not in data_20:
            st.warning(f"{name}: 数据不足，无法计算")
            continue

        st.markdown(f"""
            <div class="chart-card">
                <span class="industry-label">{name}</span>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            fig_15 = create_icvr_chart(name, data_15[name], "C:15天 | V:20天 | R:15天")
            st.plotly_chart(fig_15, use_container_width=True, key=f"icvr_filter_{name}_15")

        with col2:
            fig_20 = create_icvr_chart(name, data_20[name], "C:20天 | V:20天 | R:55天")
            st.plotly_chart(fig_20, use_container_width=True, key=f"icvr_filter_{name}_20")

        st.markdown("<hr>", unsafe_allow_html=True)

# ========== 辅助函数：计算历史分位数 ==========
def calculate_latest_percentiles(df, col_info_all, window_crowd=20, window_ret=55):
    """计算最新日期的ICVR指标历史分位数"""
    industry_names = col_info_all["industry_names"]
    Ashare_amt_col = col_info_all["Ashare_amt_col"]
    Ashare_ret_col = col_info_all["Ashare_ret_col"]
    amt_dict = col_info_all["amt_dict"]
    ret_dict = col_info_all["ret_dict"]
    vol_dict = col_info_all["vol_dict"]

    if Ashare_amt_col is None or Ashare_amt_col not in df.columns:
        raise ValueError(f"万得全A成交额列未找到")
    if Ashare_ret_col is None or Ashare_ret_col not in df.columns:
        raise ValueError(f"万得全A收益率列未找到")

    # 计算拥挤度
    Ashare_rolling = df[Ashare_amt_col].rolling(window_crowd).sum()
    crowding = pd.DataFrame(index=df.index[window_crowd - 1:])

    for name in industry_names:
        amt_col = amt_dict.get(name)
        if amt_col and amt_col in df.columns:
            crowding[name] = df[amt_col].rolling(window_crowd).sum() / Ashare_rolling
        else:
            crowding[name] = np.nan

    # 计算超额收益
    relative_returns = pd.DataFrame(index=df.index)
    for name in industry_names:
        ret_col = ret_dict.get(name)
        if ret_col and ret_col in df.columns:
            relative_returns[name] = df[ret_col] - df[Ashare_ret_col]
        else:
            relative_returns[name] = np.nan
    relative_returns_sum = relative_returns.rolling(window_ret).sum()

    # 计算波动率
    volatility = pd.DataFrame(index=crowding.index)
    for name in industry_names:
        vol_col = vol_dict.get(name)
        if vol_col and vol_col in df.columns:
            volatility[name] = df[vol_col].iloc[window_crowd - 1:]
        else:
            volatility[name] = np.nan

    # 获取最新日期
    latest_date = crowding.index[-1]

    # 计算每个行业的历史分位数
    percentile_data = []

    for name in industry_names:
        # 跳过数据不足的行业
        if name not in crowding.columns or name not in relative_returns_sum.columns or name not in volatility.columns:
            continue

        # 拥挤度历史分位数
        crowd_series = crowding[name].dropna()
        if len(crowd_series) > 0:
            latest_crowd = crowd_series.iloc[-1]
            crowd_pct = (crowd_series <= latest_crowd).mean() * 100
        else:
            latest_crowd = np.nan
            crowd_pct = np.nan

        # 超额收益历史分位数
        ret_series = relative_returns_sum[name].dropna()
        if len(ret_series) > 0:
            latest_ret = ret_series.iloc[-1]
            ret_pct = (ret_series <= latest_ret).mean() * 100
        else:
            latest_ret = np.nan
            ret_pct = np.nan

        # 波动率历史分位数
        vol_series = volatility[name].dropna()
        if len(vol_series) > 0:
            latest_vol = vol_series.iloc[-1]
            vol_pct = (vol_series <= latest_vol).mean() * 100
        else:
            latest_vol = np.nan
            vol_pct = np.nan

        percentile_data.append({
            '细分行业': name,
            '最新日期': latest_date.strftime('%Y-%m-%d'),
            '拥挤度': latest_crowd,
            '拥挤度分位数': crowd_pct,
            '超额收益': latest_ret,
            '超额收益分位数': ret_pct,
            '波动率': latest_vol,
            '波动率分位数': vol_pct
        })

    return pd.DataFrame(percentile_data)

def render_earning_revision():
    """发现牛牛 - 业绩上修（Earning Revision）"""
    st.markdown(f'<h1 class="main-title">🐮 业绩上修</h1>', unsafe_allow_html=True)

    from datetime import datetime
    today_str = datetime.now().strftime("%Y年%m月%d日")
    st.markdown(f'<p class="subtitle">数据截至 {today_str}</p>', unsafe_allow_html=True)

    try:
        df = load_earning_data(st.session_state.data_paths['earning_file'])
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return

    st.markdown("### ⚙️ 筛选条件")

    # 调整列宽比例，让布局更均衡
    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1])

    with col1:
        st.markdown("**总市值范围（亿元）**")
        cap_min = st.number_input("最小值", value=0.0, min_value=0.0, format="%.2f",
                                  key="cap_min_input", label_visibility="collapsed")
        cap_max = st.number_input("最大值", value=float(df['总市值'].max()), min_value=0.0, format="%.2f",
                                  key="cap_max_input", label_visibility="collapsed")

    with col2:
        st.markdown("**2026年业绩**")
        filter_26 = st.multiselect(
            "2026年业绩调整类型",
            options=["上调", "下调", "未调整"],
            default=[],
            key="filter_26",
            placeholder="选择调整类型...",
            label_visibility="collapsed"
        )

    with col3:
        st.markdown("**2027年业绩**")
        filter_27 = st.multiselect(
            "2027年业绩调整类型",
            options=["上调", "下调", "未调整"],
            default=[],
            key="filter_27",
            placeholder="选择调整类型...",
            label_visibility="collapsed"
        )

    with col4:
        st.markdown("**排序与数量**")
        sort_by = st.selectbox("排序方式", ['总市值', '机构数变化', 'PE(26E)',
                                            'T日预测2026年净利润中值', '净利润26E变化', '净利润27E变化'],
                               index=0, key="sort_select", label_visibility="collapsed")
        top_n = st.number_input("显示数量", min_value=5, max_value=100, value=20, step=5,
                                key="top_n_input", label_visibility="collapsed")

    # 查询按钮单独一行，居中或靠右
    _, _, _, btn_col = st.columns([1.2, 1.2, 1.2, 1])
    with btn_col:
        query_btn = st.button("🔍 查询", use_container_width=True, type="primary", key="query_btn")

    st.markdown("---")

    if not query_btn:
        st.markdown("""
            <div class="info-box">
                <div style='font-size:3rem;margin-bottom:16px;'>👆</div>
                <p style='color:#6b7280;font-size:1.1rem;'>请设置筛选条件后，点击右侧"🔍 查询"按钮</p>
                <p style='color:#9ca3af;font-size:0.9rem;margin-top:8px;'>支持：总市值范围 + 业绩调整类型筛选</p>
            </div>
        """, unsafe_allow_html=True)
        return

    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df['总市值'] >= cap_min) & (filtered_df['总市值'] <= cap_max)]

    # 2026年业绩筛选
    if filter_26:
        filtered_df = filtered_df[filtered_df['业绩调整26E'].isin(filter_26)]

    # 2027年业绩筛选
    if filter_27:
        filtered_df = filtered_df[filtered_df['业绩调整27E'].isin(filter_27)]

    # 排序
    filtered_df[sort_by] = filtered_df[sort_by].replace([np.inf, -np.inf], np.nan)
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=False, na_position='last')
    result_df = filtered_df.head(int(top_n))

    # 统计指标
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("📊 总标的数", f"{len(df):,}")
    col_stat2.metric("✅ 筛选通过", f"{len(filtered_df):,}")
    col_stat3.metric("🏆 当前展示", f"{len(result_df)}")

    # 修改统计指标：显示上调占比
    up_26_pct = (df['业绩调整26E'] == '上调').sum() / len(df) * 100 if len(df) > 0 else 0
    col_stat4.metric("📈 全市场上调26年占比", f"{up_26_pct:.1f}%")

    st.markdown("---")

    if len(result_df) == 0:
        st.warning("⚠️ 没有符合条件的标的，请调整筛选条件")
        return

    # 修改展示列：删除净利润26E_修正% 和 最新持仓比例，新增业绩调整列
    display_cols = ['证券代码', '证券简称', '所属一级行业', '总市值', 'T日预测2026年净利润中值',
                    'T-1日预测2026年净利润中值',
                    'T日预测2027年净利润中值', 'T-1日预测2027年净利润中值', 'PE(26E)',
                    '业绩调整26E', '业绩调整27E', '机构数变化']
    display_cols = [c for c in display_cols if c in result_df.columns]

    show_df = result_df[display_cols].copy()

    # 格式化机构数变化
    if '机构数变化' in show_df.columns:
        show_df['机构数变化'] = show_df['机构数变化'].apply(lambda x: f"+{int(x)}" if x > 0 else str(int(x)))

    row_count = len(show_df)
    table_height = min(max(35 + row_count * 35, 200), 800)

    st.dataframe(show_df, use_container_width=True, height=table_height, hide_index=True)

    st.markdown("---")
    st.subheader("📈 基金持仓比例趋势")

    fund_cols = []
    for col in result_df.columns:
        col_str = str(col)
        if len(col_str) == 8 and col_str.isdigit():
            fund_cols.append(col)
    fund_cols = sorted(fund_cols)

    if len(fund_cols) > 0 and len(result_df) > 0:
        for i in range(0, len(result_df), 2):
            cols = st.columns(2)

            with cols[0]:
                row = result_df.iloc[i]
                fig1 = go.Figure()

                fund_values = []
                dates = []
                for col in fund_cols:
                    val = row[col]
                    if pd.notna(val):
                        fund_values.append(val * 100)
                        col_str = str(col)
                        dates.append(f"{col_str[:4]}-{col_str[4:6]}")

                if len(fund_values) > 0:
                    fig1.add_trace(go.Scatter(
                        x=dates, y=fund_values, mode='lines+markers',
                        line=dict(width=2, color='#6366f1'),
                        marker=dict(size=6),
                        hovertemplate='%{x}<br>%{y:.2f}%<extra></extra>'
                    ))

                    fig1.update_layout(
                        title=f"{row['证券简称']} ({row['证券代码']})",
                        xaxis_title="时间",
                        yaxis_title="基金持仓比例(%)",
                        height=300,
                        showlegend=False,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(l=50, r=30, t=50, b=40),
                        xaxis=dict(showgrid=True, gridcolor='#f0f0f0', tickfont=dict(size=10)),
                        yaxis=dict(showgrid=True, gridcolor='#f0f0f0', tickfont=dict(size=10))
                    )
                    st.plotly_chart(fig1, use_container_width=True, key=f"fund_chart_{i}")

            if i + 1 < len(result_df):
                with cols[1]:
                    row = result_df.iloc[i + 1]
                    fig2 = go.Figure()

                    fund_values = []
                    dates = []
                    for col in fund_cols:
                        val = row[col]
                        if pd.notna(val):
                            fund_values.append(val * 100)
                            col_str = str(col)
                            dates.append(f"{col_str[:4]}-{col_str[4:6]}")

                    if len(fund_values) > 0:
                        fig2.add_trace(go.Scatter(
                            x=dates, y=fund_values, mode='lines+markers',
                            line=dict(width=2, color='#10b981'),
                            marker=dict(size=6),
                            hovertemplate='%{x}<br>%{y:.2f}%<extra></extra>'
                        ))

                        fig2.update_layout(
                            title=f"{row['证券简称']} ({row['证券代码']})",
                            xaxis_title="时间",
                            yaxis_title="基金持仓比例(%)",
                            height=300,
                            showlegend=False,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(l=50, r=30, t=50, b=40),
                            xaxis=dict(showgrid=True, gridcolor='#f0f0f0', tickfont=dict(size=10)),
                            yaxis=dict(showgrid=True, gridcolor='#f0f0f0', tickfont=dict(size=10))
                        )
                        st.plotly_chart(fig2, use_container_width=True, key=f"fund_chart_{i + 1}")
    else:
        st.info(f"暂无基金持仓数据")


def validate_data_paths(config):
    """验证数据文件路径是否存在"""
    missing_files = []

    for key, path in config.items():
        if 'file' in key and not os.path.exists(path):
            missing_files.append(f"{key}: {path}")

    if missing_files:
        st.error("以下数据文件未找到：")
        for f in missing_files:
            st.error(f"  • {f}")
        st.info("请在下方 main() 函数中修改 DATA_CONFIG 配置")
        return False
    return True

# =============================
# 主程序入口
# =============================

def main():
    import urllib.request
    import zipfile
    import socket
    
    # GitHub 上的 source.zip 直链
    SOURCE_ZIP_URL = "https://github.com/QIKangxu/moneyfactory/raw/main/source.zip"
    
    def download_and_extract_source(zip_url, extract_to="."):
        """自动下载并解压 source.zip"""
        zip_path = "source.zip"
        
        # 如果 source 文件夹已存在且包含所有必要文件，跳过下载
        required_files = ["source/data.csv", "source/search.csv", "source/ov.csv", "source/info.csv"]
        if all(os.path.exists(f) for f in required_files):
            return
        
        with st.spinner("首次运行，正在下载数据文件（约50MB，可能需要1-2分钟）..."):
            try:
                # 设置超时时间为 5 分钟
                socket.setdefaulttimeout(300)
                
                # 添加请求头，模拟浏览器
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                req = urllib.request.Request(zip_url, headers=headers)
                
                # 下载并显示进度
                def download_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(downloaded / total_size * 100, 100)
                    if block_num % 10 == 0:  # 每10个块更新一次
                        st.write(f"下载进度: {percent:.1f}%")
                
                urllib.request.urlretrieve(req, zip_path, reporthook=download_progress)
                
                st.write("下载完成，正在解压...")
                
                # 解压
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                
                # 删除zip文件节省空间
                os.remove(zip_path)
                
                st.success("✅ 数据下载完成！")
                
            except Exception as e:
                st.error(f"下载数据失败: {e}")
                st.info("请检查网络连接，或手动将数据文件放到 source/ 文件夹")
                # 如果下载失败，尝试继续运行（可能本地已有数据）
                if not all(os.path.exists(f) for f in required_files):
                    raise e
    
    # 自动下载解压（如果失败会报错）
    try:
        download_and_extract_source(SOURCE_ZIP_URL)
    except Exception as e:
        st.warning("自动下载失败，请检查网络或手动放置数据文件")
        # 检查本地是否已有数据
        required_files = ["source/data.csv", "source/search.csv", "source/ov.csv", "source/info.csv"]
        if not all(os.path.exists(f) for f in required_files):
            st.error("本地也没有数据文件，程序无法运行")
            return
    
    # 原有配置保持不变
    DATA_CONFIG = {
        'icvr_file': "source/data.csv",
        'earning_file': "source/search.csv",
        'market_overview_file': "source/ov.csv",
        'stock_info_file': "source/info.csv"
    }

    st.session_state.data_paths = DATA_CONFIG

    if not validate_data_paths(DATA_CONFIG):
        return

    render_sidebar()

    current_page = st.session_state.page

    if current_page == "welcome":
        render_welcome()

    elif current_page == "market_overview":
        render_market_overview()

    elif current_page == "index_quote":
        render_index_quote()

    elif current_page == "icvr_overview":
        try:
            df, latest_date_str = load_icvr_data(DATA_CONFIG['icvr_file'])
            render_icvr_overview(df, latest_date_str)
        except Exception as e:
            st.error(f"ICVR 数据加载失败: {e}")
            import traceback
            st.error(traceback.format_exc())

    elif current_page == "icvr_filter":
        try:
            df, latest_date_str = load_icvr_data(DATA_CONFIG['icvr_file'])
            render_icvr_filter(df, latest_date_str)
        except Exception as e:
            st.error(f"ICVR 数据加载失败: {e}")
            import traceback
            st.error(traceback.format_exc())

    elif current_page == "earning_revision":
        render_earning_revision()

    else:
        st.error("未知页面")


if __name__ == "__main__":
    main()
