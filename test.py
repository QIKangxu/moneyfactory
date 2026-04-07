import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import os

st.sidebar.markdown("### 🔥 版本: v1.6 - 测试标记")
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
# 数据加载函数（ICVR）
# =============================
@st.cache_data(ttl=3600)
def load_icvr_data(file_path, sheet_name="data"):
    """加载ICVR数据（拥挤度、波动率、超额收益）"""
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=[0, 1, 2, 3])

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
def load_earning_data(file_path, sheet_name="search"):
    """加载业绩上修数据（Earning Revision）"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)

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
        if st.button("📈 市场概览", key="nav_market_overview", use_container_width=True):
            st.session_state.page = "market_overview"
            st.session_state.show_icvr_submenu = False
        if st.button("📊 指数行情", key="nav_index_quote", use_container_width=True):
            st.session_state.page = "index_quote"
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
        st.button("即将上线", disabled=True, key="welcome_market", use_container_width=True)

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
    """大盘速览 - 市场概览（预留）"""
    st.markdown(f'<h1 class="main-title">📈 市场概览</h1>', unsafe_allow_html=True)
    st.info("🚧 功能开发中，敬请期待...")


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

        c15, r15, v15 = calculate_icvr_indicators(df, col_info, 15, 15)
        data_15 = standardize_icvr_data(c15, v15, r15, col_info["industry_names"], 15, 15)

        c20, r20, v20 = calculate_icvr_indicators(df, col_info, 20, 55)
        data_20 = standardize_icvr_data(c20, v20, r20, col_info["industry_names"], 20, 55)

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
            fig_15 = create_icvr_chart(name, data_15[name], "C:15天 | V:15天 | R:15天")
            st.plotly_chart(fig_15, use_container_width=True, key=f"icvr_{name}_15")

        with col2:
            fig_20 = create_icvr_chart(name, data_20[name], "C:20天 | V:20天 | R:55天")
            st.plotly_chart(fig_20, use_container_width=True, key=f"icvr_{name}_20")

        st.markdown("<hr>", unsafe_allow_html=True)


def render_icvr_filter(df, latest_date_str):
    """ICVR - 细分行业筛选"""
    st.markdown(f'<h1 class="main-title">🔍 ICVR 细分行业筛选</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">数据截至 {latest_date_str}</p>', unsafe_allow_html=True)

    col_info_all = identify_icvr_columns(df, category_filter="细分行业")
    industry_options = col_info_all["industry_names"]

    if len(industry_options) == 0:
        st.error("未找到细分行业数据，请检查数据源")
        return

    st.markdown(f"<p style='color:#6b7280;margin-bottom:16px;'>可选 <b>{len(industry_options)}</b> 个细分行业</p>",
                unsafe_allow_html=True)

    selected = st.multiselect(
        "选择要分析的细分行业（支持多选）",
        options=industry_options,
        default=[],
        placeholder="请选择行业..."
    )

    if not selected:
        st.markdown("""
            <div class="info-box">
                <div style='font-size:2rem;margin-bottom:12px;'>👆</div>
                <p style='color:#6b7280;'>请从上方选择至少一个细分行业进行分析</p>
            </div>
        """, unsafe_allow_html=True)
        return

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

    with st.spinner(f"正在计算 {len(selected)} 个行业的 ICVR 指标..."):
        col_info = {
            "industry_names": selected,
            "Ashare_amt_col": col_info_all["Ashare_amt_col"],
            "Ashare_ret_col": col_info_all["Ashare_ret_col"],
            "amt_dict": {k: v for k, v in col_info_all["amt_dict"].items() if k in selected},
            "ret_dict": {k: v for k, v in col_info_all["ret_dict"].items() if k in selected},
            "vol_dict": {k: v for k, v in col_info_all["vol_dict"].items() if k in selected},
        }

        try:
            c15, r15, v15 = calculate_icvr_indicators(df, col_info, 15, 15)
            data_15 = standardize_icvr_data(c15, v15, r15, selected, 15, 15)

            c20, r20, v20 = calculate_icvr_indicators(df, col_info, 20, 55)
            data_20 = standardize_icvr_data(c20, v20, r20, selected, 20, 55)

        except Exception as e:
            st.error(f"计算出错: {e}")
            return

    for name in selected:
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
            fig_15 = create_icvr_chart(name, data_15[name], "C:15天 | V:15天 | R:15天")
            st.plotly_chart(fig_15, use_container_width=True, key=f"icvr_filter_{name}_15")

        with col2:
            fig_20 = create_icvr_chart(name, data_20[name], "C:20天 | V:20天 | R:55天")
            st.plotly_chart(fig_20, use_container_width=True, key=f"icvr_filter_{name}_20")

        st.markdown("<hr>", unsafe_allow_html=True)


def render_earning_revision():
    """发现牛牛 - 业绩上修（Earning Revision）"""
    st.markdown(f'<h1 class="main-title">🐮 业绩上修</h1>', unsafe_allow_html=True)

    from datetime import datetime
    today_str = datetime.now().strftime("%Y年%m月%d日")
    st.markdown(f'<p class="subtitle">数据截至 {today_str}</p>', unsafe_allow_html=True)

    try:
        df = load_earning_data(st.session_state.data_paths['earning_file'],
                               st.session_state.data_paths['earning_sheet'])
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
    DATA_CONFIG = {
        'icvr_file': "data.xlsx",
        'icvr_sheet': "data",
        'earning_file': "search.xlsx",
        'earning_sheet': "search"
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
            df, latest_date_str = load_icvr_data(DATA_CONFIG['icvr_file'],
                                                 DATA_CONFIG['icvr_sheet'])
            render_icvr_overview(df, latest_date_str)
        except Exception as e:
            st.error(f"ICVR 数据加载失败: {e}")
            import traceback
            st.error(traceback.format_exc())

    elif current_page == "icvr_filter":
        try:
            df, latest_date_str = load_icvr_data(DATA_CONFIG['icvr_file'],
                                                 DATA_CONFIG['icvr_sheet'])
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
