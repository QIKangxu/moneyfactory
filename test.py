import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

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
# 自定义CSS样式（添加移动端适配）
# =============================
st.markdown("""
<style>
    /* 全局字体和颜色 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* 主标题样式 */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    /* 副标题样式 */
    .subtitle {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    
    /* 卡片容器样式 */
    .chart-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 16px;
    }
    
    /* 行业名称标签 */
    .industry-label {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 12px;
    }
    
    /* 图例说明样式 */
    .legend-box {
        background: #f9fafb;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 20px;
        border-left: 4px solid #667eea;
        font-size: 0.85rem;
    }
    
    /* 选择框样式优化 */
    .stMultiSelect [data-baseweb="select"] {
        border-radius: 10px;
    }
    
    /* 按钮样式 */
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
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* 分隔线样式 */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
        margin: 1.5rem 0;
    }
    
    /* 信息提示框 */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border: 1px solid #667eea30;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    /* 欢迎页面样式 */
    .welcome-container {
        text-align: center;
        padding: 40px 20px;
    }
    
    .welcome-icon {
        font-size: 3rem;
        margin-bottom: 16px;
    }
    
    /* 移动端适配 */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.5rem;
        }
        .subtitle {
            font-size: 0.8rem;
        }
        .chart-card {
            padding: 12px;
        }
        .industry-label {
            font-size: 0.8rem;
            padding: 4px 10px;
        }
        .legend-box {
            font-size: 0.75rem;
            padding: 10px 12px;
        }
    }
</style>
""", unsafe_allow_html=True)

# =============================
# 初始化 session_state
# =============================
if "page" not in st.session_state:
    st.session_state.page = "请选择"
if "selected_industries" not in st.session_state:
    st.session_state.selected_industries = []


# =============================
# 数据加载函数
# =============================
@st.cache_data
def load_data(file_path):
    """加载并缓存数据"""
    df = pd.read_excel(file_path, sheet_name="data", header=[0, 1, 2, 3])

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
# 核心计算函数
# =============================
def identify_columns_by_category(df, category_filter=None):
    """按分类识别列"""
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


def calculate_indicators(df, col_info, window_crowd=20, window_ret=55):
    """计算指标"""
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


def standardize_data(crowding, volatility, relative_returns_sum, industry_names,
                     window_crowd=20, window_ret=55):
    """标准化数据"""
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


def create_chart(name, data, params_label, height=380, is_mobile=False):
    """创建图表（支持移动端适配）"""
    fig = go.Figure()
    
    colors = {
        '拥挤度': '#6366f1',
        '波动率': '#10b981',
        '超额收益': '#f43f5e'
    }
    
    # 移动端调整字体大小
    font_size = 9 if is_mobile else 10
    title_size = 14 if is_mobile else 16
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data["拥挤度"], mode='lines',
        name='拥挤度', line=dict(color=colors['拥挤度'], width=2),
        yaxis='y', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>拥挤度: %{y:.1%}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data["波动率"], mode='lines',
        name='波动率', line=dict(color=colors['波动率'], width=2),
        yaxis='y', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>波动率: %{y:.1%}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data["超额收益"], mode='lines',
        name='超额收益', line=dict(color=colors['超额收益'], width=2),
        yaxis='y2', showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>超额收益: %{y:.1%}<extra></extra>'
    ))

    # 移动端调整边距
    margin_top = 70 if is_mobile else 80
    margin_lr = 40 if is_mobile else 60
    
    fig.update_layout(
        title=dict(
            text=f"<b>{name}</b>",
            font=dict(size=title_size, color='#1f2937'),
            x=0.5, xanchor='center',
            y=0.98, yanchor='top'
        ),
        annotations=[dict(
            text=f"<span style='color:#6b7280;font-size:{font_size}px;'>{params_label}</span>",
            xref='paper', yref='paper',
            x=0, y=1.08 if is_mobile else 1.12,
            xanchor='left', yanchor='top',
            showarrow=False
        )],
        xaxis=dict(
            showline=True, linecolor='#e5e7eb', linewidth=1,
            tickfont=dict(color='#6b7280', size=font_size),
            zeroline=False, showgrid=True, gridcolor='#f3f4f6',
            tickformat='%Y-%m' if not is_mobile else '%y-%m'
        ),
        yaxis=dict(
            side='left', range=[0, 1], tickmode='array', tickvals=[0, 0.5, 1],
            ticktext=['0%', '50%', '100%'], tickfont=dict(color='#6b7280', size=font_size),
            showgrid=True, gridcolor='#f3f4f6',
            showline=True, linecolor='#e5e7eb', linewidth=1, zeroline=False,
            title=dict(text='拥挤度/波动率', font=dict(size=font_size, color='#9ca3af'))
        ),
        yaxis2=dict(
            overlaying='y', side='right', tickformat='.0%',
            tickfont=dict(color='#6b7280', size=font_size),
            showline=True, linecolor='#e5e7eb', linewidth=1,
            showgrid=False, zeroline=False,
            title=dict(text='超额收益', font=dict(size=font_size, color='#9ca3af'))
        ),
        legend=dict(
            orientation='h',
            yanchor='top', y=1.08 if is_mobile else 1.12,
            xanchor='right', x=1,
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
            font=dict(size=font_size),
            itemsizing='constant'
        ),
        margin=dict(l=margin_lr, r=margin_lr, t=margin_top, b=40),
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
    
    with st.sidebar.expander("🔥 拥挤度", expanded=False):
        if st.button("📊 一级行业概览", key="nav_一级行业概览", use_container_width=True):
            st.session_state.page = "拥挤度_一级行业概览"
        if st.button("🔍 细分行业筛选", key="nav_细分行业筛选", use_container_width=True):
            st.session_state.page = "拥挤度_细分行业筛选"

    with st.sidebar.expander("📈 景气度", expanded=False):
        st.button("行业景气度", key="景气度_行业", disabled=True, use_container_width=True)
        st.button("个股景气度", key="景气度_个股", disabled=True, use_container_width=True)

    with st.sidebar.expander("💰 资金流向", expanded=False):
        st.button("北向资金", key="资金_北向", disabled=True, use_container_width=True)
        st.button("主力资金", key="资金_主力", disabled=True, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style='text-align:center;color:#9ca3af;font-size:0.75rem;'>
            系统版本 v1.2<br>
            © 2026 工厂
        </div>
    """, unsafe_allow_html=True)


def render_请选择():
    """初始欢迎界面"""
    st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">🏭</div>
            <h1 class="main-title">欢迎使用工厂</h1>
            <p class="subtitle">智能行业分析系统</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background:white;padding:20px;border-radius:16px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);text-align:center;'>
                <div style='font-size:2rem;margin-bottom:10px;'>🔥</div>
                <h3 style='color:#1f2937;margin-bottom:6px;font-size:1rem;'>拥挤度分析</h3>
                <p style='color:#6b7280;font-size:0.8rem;'>一级行业概览与细分行业筛选</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("进入拥挤度", key="welcome_拥挤度", use_container_width=True):
            st.session_state.page = "拥挤度_一级行业概览"
            st.rerun()
    
    with col2:
        st.markdown("""
            <div style='background:white;padding:20px;border-radius:16px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);text-align:center;opacity:0.6;'>
                <div style='font-size:2rem;margin-bottom:10px;'>📈</div>
                <h3 style='color:#1f2937;margin-bottom:6px;font-size:1rem;'>景气度分析</h3>
                <p style='color:#6b7280;font-size:0.8rem;'>即将上线</p>
            </div>
        """, unsafe_allow_html=True)
        st.button("敬请期待", key="welcome_景气度", disabled=True, use_container_width=True)
    
    with col3:
        st.markdown("""
            <div style='background:white;padding:20px;border-radius:16px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);text-align:center;opacity:0.6;'>
                <div style='font-size:2rem;margin-bottom:10px;'>💰</div>
                <h3 style='color:#1f2937;margin-bottom:6px;font-size:1rem;'>资金流向</h3>
                <p style='color:#6b7280;font-size:0.8rem;'>即将上线</p>
            </div>
        """, unsafe_allow_html=True)
        st.button("敬请期待", key="welcome_资金流向", disabled=True, use_container_width=True)


def render_一级行业概览(df, latest_date_str):
    """渲染一级行业概览页面（响应式）"""
    st.markdown(f'<h1 class="main-title">拥挤度 - 一级行业概览</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">数据截至 {latest_date_str}</p>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="legend-box">
            <span style='font-weight:600;color:#374151;'>图例：</span>
            <span style='color:#6366f1;'>● 拥挤度</span>
            <span style='margin:0 6px;color:#d1d5db;'>|</span>
            <span style='color:#10b981;'>● 波动率</span>
            <span style='margin:0 6px;color:#d1d5db;'>|</span>
            <span style='color:#f43f5e;'>● 超额收益</span>（右轴）
        </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("正在计算指标..."):
        col_info = identify_columns_by_category(df, category_filter="一级行业")
        
        if len(col_info["industry_names"]) == 0:
            st.error("未找到一级行业数据")
            return
        
        c15, r15, v15 = calculate_indicators(df, col_info, 15, 15)
        data_15 = standardize_data(c15, v15, r15, col_info["industry_names"], 15, 15)
        
        c20, r20, v20 = calculate_indicators(df, col_info, 20, 55)
        data_20 = standardize_data(c20, v20, r20, col_info["industry_names"], 20, 55)
    
    total_count = len([n for n in col_info["industry_names"] if n in data_15 and n in data_20])
    st.markdown(f"<p style='color:#6b7280;margin-bottom:16px;font-size:0.9rem;'>共 <b>{total_count}</b> 个一级行业</p>", unsafe_allow_html=True)
    
    # 检测是否为移动端（根据屏幕宽度判断）
    is_mobile = st.session_state.get('is_mobile', False)
    
    for name in col_info["industry_names"]:
        if name not in data_15 or name not in data_20:
            continue
        
        st.markdown(f"""
            <div class="chart-card">
                <span class="industry-label">{name}</span>
            </div>
        """, unsafe_allow_html=True)
        
        if is_mobile:
            # 移动端：上下排列，增大高度
            fig_15 = create_chart(name, data_15[name], "拥挤度15天 | 超额收益15天", height=320, is_mobile=True)
            st.plotly_chart(fig_15, use_container_width=True, key=f"{name}_15")
            
            fig_20 = create_chart(name, data_20[name], "拥挤度20天 | 超额收益55天", height=320, is_mobile=True)
            st.plotly_chart(fig_20, use_container_width=True, key=f"{name}_20")
        else:
            # PC端：左右排列
            col1, col2 = st.columns(2)
            
            with col1:
                fig_15 = create_chart(name, data_15[name], "拥挤度15天 | 超额收益15天", height=380, is_mobile=False)
                st.plotly_chart(fig_15, use_container_width=True, key=f"{name}_15")
            
            with col2:
                fig_20 = create_chart(name, data_20[name], "拥挤度20天 | 超额收益55天", height=380, is_mobile=False)
                st.plotly_chart(fig_20, use_container_width=True, key=f"{name}_20")
        
        st.markdown("<hr>", unsafe_allow_html=True)


def render_细分行业筛选(df, latest_date_str):
    """渲染细分行业筛选页面（响应式）"""
    st.markdown(f'<h1 class="main-title">拥挤度 - 细分行业筛选</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">数据截至 {latest_date_str}</p>', unsafe_allow_html=True)
    
    col_info_all = identify_columns_by_category(df, category_filter="细分行业")
    industry_options = col_info_all["industry_names"]
    
    if len(industry_options) == 0:
        st.error("未找到细分行业数据")
        return
    
    st.markdown(f"<p style='color:#6b7280;margin-bottom:12px;font-size:0.9rem;'>可选 <b>{len(industry_options)}</b> 个细分行业</p>", unsafe_allow_html=True)
    
    selected = st.multiselect(
        "🔍 选择要分析的细分行业（支持多选）",
        options=industry_options,
        default=[],
        placeholder="请选择行业..."
    )
    
    if not selected:
        st.markdown("""
            <div class="info-box">
                <div style='font-size:1.5rem;margin-bottom:8px;'>👆</div>
                <p style='color:#6b7280;font-size:0.9rem;'>请从上方选择至少一个细分行业</p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
        <div class="legend-box">
            <span style='font-weight:600;color:#374151;'>图例：</span>
            <span style='color:#6366f1;'>● 拥挤度</span>
            <span style='margin:0 6px;color:#d1d5db;'>|</span>
            <span style='color:#10b981;'>● 波动率</span>
            <span style='margin:0 6px;color:#d1d5db;'>|</span>
            <span style='color:#f43f5e;'>● 超额收益</span>（右轴）
        </div>
    """, unsafe_allow_html=True)
    
    is_mobile = st.session_state.get('is_mobile', False)
    
    with st.spinner(f"正在计算 {len(selected)} 个行业的指标..."):
        col_info = {
            "industry_names": selected,
            "Ashare_amt_col": col_info_all["Ashare_amt_col"],
            "Ashare_ret_col": col_info_all["Ashare_ret_col"],
            "amt_dict": {k: v for k, v in col_info_all["amt_dict"].items() if k in selected},
            "ret_dict": {k: v for k, v in col_info_all["ret_dict"].items() if k in selected},
            "vol_dict": {k: v for k, v in col_info_all["vol_dict"].items() if k in selected},
        }
        
        try:
            c15, r15, v15 = calculate_indicators(df, col_info, 15, 15)
            data_15 = standardize_data(c15, v15, r15, selected, 15, 15)
            
            c20, r20, v20 = calculate_indicators(df, col_info, 20, 55)
            data_20 = standardize_data(c20, v20, r20, selected, 20, 55)
            
        except Exception as e:
            st.error(f"计算出错: {e}")
            return
    
    for name in selected:
        if name not in data_15 or name not in data_20:
            st.warning(f"{name}: 数据不足")
            continue
        
        st.markdown(f"""
            <div class="chart-card">
                <span class="industry-label">{name}</span>
            </div>
        """, unsafe_allow_html=True)
        
        if is_mobile:
            fig_15 = create_chart(name, data_15[name], "拥挤度15天 | 超额收益15天", height=320, is_mobile=True)
            st.plotly_chart(fig_15, use_container_width=True, key=f"细分_{name}_15")
            
            fig_20 = create_chart(name, data_20[name], "拥挤度20天 | 超额收益55天", height=320, is_mobile=True)
            st.plotly_chart(fig_20, use_container_width=True, key=f"细分_{name}_20")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_15 = create_chart(name, data_15[name], "拥挤度15天 | 超额收益15天", height=380, is_mobile=False)
                st.plotly_chart(fig_15, use_container_width=True, key=f"细分_{name}_15")
            
            with col2:
                fig_20 = create_chart(name, data_20[name], "拥挤度20天 | 超额收益55天", height=380, is_mobile=False)
                st.plotly_chart(fig_20, use_container_width=True, key=f"细分_{name}_20")
        
        st.markdown("<hr>", unsafe_allow_html=True)


# =============================
# 主程序入口
# =============================

def main():
    render_sidebar()
    
    # 检测是否为移动端
    # 使用JavaScript检测屏幕宽度
    st.markdown("""
        <script>
            const width = window.innerWidth;
            const isMobile = width < 768;
            window.parent.postMessage({type: 'streamlit:setSessionState', isMobile: isMobile}, '*');
        </script>
    """, unsafe_allow_html=True)
    
    # 默认假设为移动端（保险起见）
    if 'is_mobile' not in st.session_state:
        st.session_state.is_mobile = True
    
    file_path = "data.xlsx"
    
    try:
        df, latest_date_str = load_data(file_path)
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        import traceback
        st.error(traceback.format_exc())
        return
    
    current_page = st.session_state.page
    
    if current_page == "请选择":
        render_请选择()
    elif current_page == "拥挤度_一级行业概览":
        render_一级行业概览(df, latest_date_str)
    elif current_page == "拥挤度_细分行业筛选":
        render_细分行业筛选(df, latest_date_str)
    else:
        st.error("未知页面")


if __name__ == "__main__":
    main()
