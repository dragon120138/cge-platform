# -*- coding: utf-8 -*-
"""
中国42部门CGE政策模拟平台 v2.0 — 专业经济分析面板

v2.0新增:
  - 12季度(3年)动态路径, 按季度展示
  - 基准年2025数据
  - 利率政策工具
  - TFP(全要素生产率)冲击, 支持全行业/定向行业
  - 定向产业投资支持
  - 财政盈余/赤字动态路径
  - 三部门信心指数(消费者/企业/投资者)动态路径
  - 分行业12季度产出动态钻取
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os, time, builtins
from pathlib import Path
from datetime import datetime

_orig_print = builtins.print
def _safe_print(*a, **k):
    try: _orig_print(*a, **k)
    except: pass
builtins.print = _safe_print

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cge_core.model_builder import ModelBuilder
from cge_core.sectors import SECTOR_NAMES_CN, SECTOR_CODES, SECTORS_42

# =====================================================================
#  页面配置
# =====================================================================
st.set_page_config(page_title="中国42部门CGE政策模拟平台 v2.0", page_icon="📊", layout="wide")

# =====================================================================
#  专业蓝调 CSS
# =====================================================================
st.markdown("""
<style>
.stApp {
    background: #f0f4f8;
    color: #1a2744;
    font-family: "Microsoft YaHei", "SimHei", sans-serif;
}
h1, h2, h3 { color: #1a3a6b !important; font-weight: 700; }
h4, h5, h6 { color: #2c5282 !important; }

/* 侧边栏 */
.stSidebar {
    background: #ffffff !important;
    border-right: 2px solid #3182ce;
}

/* 指标卡 */
.metric-card {
    background: #ffffff;
    border: 1px solid #cbd5e0;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.metric-label { color: #4a5568; font-size: 15px; margin-bottom: 6px; font-weight: 500; }
.metric-value { font-size: 28px; font-weight: 800; }
.metric-value.pos { color: #2f855a; }
.metric-value.neg { color: #c53030; }
.metric-value.neu { color: #3182ce; }

/* 按钮 */
.stButton > button {
    background: linear-gradient(180deg, #3182ce 0%, #2c5282 100%);
    color: #fff !important;
    border: none;
    border-radius: 8px;
    padding: 12px 32px;
    font-size: 16px !important;
    font-weight: 700;
    width: 100%;
    box-shadow: 0 2px 8px rgba(49,130,206,0.3);
}
.stButton > button:hover {
    background: linear-gradient(180deg, #4299e1 0%, #3182ce 100%);
    box-shadow: 0 4px 12px rgba(49,130,206,0.4);
}

/* Tab */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #e2e8f0; border-radius: 8px 8px 0 0; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    background: #ebf4ff;
    border: 1px solid #bee3f8;
    border-radius: 8px 8px 0 0;
    padding: 10px 24px;
    color: #4a5568;
    font-size: 15px;
    font-weight: 600;
}
.stTabs [data-baseweb="tab"]:hover { color: #2c5282; border-color: #3182ce; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, #3182ce 0%, #2c5282 100%) !important;
    color: #fff !important;
    border-color: #3182ce !important;
}

/* 信息框 */
.info-box {
    background: #ebf8ff;
    border-left: 4px solid #3182ce;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
    color: #2d3748;
}
.alert-box {
    background: #fff5f5;
    border-left: 4px solid #c53030;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
}
.success-box {
    background: #f0fff4;
    border-left: 4px solid #2f855a;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
}

/* 表格 */
.dataframe { font-size: 14px; }
.dataframe th { background-color: #3182ce !important; color: #fff !important; }

/* 侧边栏表格 */
.sidebar-table { font-size: 14px; color: #2d3748; }
.sidebar-table td { padding: 3px 8px; }
.sidebar-table .label { color: #718096; }
.sidebar-table .value { color: #2c5282; font-weight: 700; text-align: right; }

/* 报告 */
.report-content {
    background: #ffffff;
    color: #1a202c;
    padding: 32px 40px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    font-family: "Microsoft YaHei", "SimSun", serif;
    font-size: 16px;
    line-height: 1.9;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.report-content h1, .report-content h2, .report-content h3 {
    color: #1a3a6b !important;
    text-shadow: none;
}

/* selectbox */
div[data-baseweb="select"] > div { background-color: #ebf4ff; border-color: #90cdf4; color: #2d3748; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
#  初始化模型
# =====================================================================
_MODEL_CACHE_VERSION = "v2.0.1"

@st.cache_resource(max_entries=1)
def get_model():
    b = ModelBuilder()
    b.load_sam()
    b.build_solver()
    return b

builder = get_model()
sam = builder.sam
sam_summary = builder.get_sam_summary()

TIMING_OPTIONS = {"permanent": "永久实施", "temporary": "一次性脉冲", "anticipated": "预告后实施"}

# 行业选择列表 (用于TFP和定向投资)
SECTOR_OPTIONS = [f"S{i+1:02d} {SECTOR_NAMES_CN[i]}" for i in range(42)]

# =====================================================================
#  标题
# =====================================================================
st.markdown("""
<div style='text-align:center; padding:4px 0 12px 0; border-bottom: 2px solid #3182ce; margin-bottom:12px;'>
    <h1 style='font-size:24px; margin:0;'>中国42部门CGE政策模拟平台 v2.0</h1>
    <p style='color:#718096; font-size:13px; margin:4px 0 0 0;'>
        Computable General Equilibrium · Johansen对数线性化引擎 · 基准年2025 · 42部门 · 12季度(3年)动态路径
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
#  侧边栏
# =====================================================================
with st.sidebar:
    st.markdown("### 📊 基准经济概况 (2025)")
    labor_share = sam_summary.get('labor_compensation', 0) / max(sam_summary['gdp'], 1)
    cap_share = sam_summary.get('capital_income', 0) / max(sam_summary['gdp'], 1)
    st.markdown(f"""
    <table class="sidebar-table" style="width:100%;">
        <tr><td class="label">GDP</td><td class="value">{sam_summary['gdp']/1e4:,.0f} 万亿元</td></tr>
        <tr><td class="label">居民消费</td><td class="value">{sam_summary['household_consumption']/1e4:,.0f} 万亿</td></tr>
        <tr><td class="label">政府支出</td><td class="value">{sam_summary['government_consumption']/1e4:,.0f} 万亿</td></tr>
        <tr><td class="label">总投资</td><td class="value">{sam_summary['investment']/1e4:,.0f} 万亿</td></tr>
        <tr><td class="label">出口</td><td class="value">{sam_summary['exports']/1e4:,.0f} 万亿</td></tr>
        <tr><td class="label">进口</td><td class="value">{sam_summary['imports']/1e4:,.0f} 万亿</td></tr>
    </table>
    <hr style="border:0; border-top:1px solid #e2e8f0; margin:8px 0;">
    <table class="sidebar-table" style="width:100%;">
        <tr><td class="label">劳动报酬占比</td><td class="value" style="color:#2f855a;">{labor_share:.1%}</td></tr>
        <tr><td class="label">资本回报占比</td><td class="value" style="color:#b7791f;">{cap_share:.1%}</td></tr>
        <tr><td class="label">贸易差额</td><td class="value">{sam_summary.get('trade_balance',0)/1e4:,.0f} 万亿</td></tr>
        <tr><td class="label">财政收入</td><td class="value">{getattr(sam, 'fiscal_revenue', 216000.0)/1e4:,.1f} 万亿</td></tr>
        <tr><td class="label">财政支出</td><td class="value">{getattr(sam, 'fiscal_expenditure', 287300.0)/1e4:,.1f} 万亿</td></tr>
        <tr><td class="label">财政赤字</td><td class="value" style="color:#c53030;">{getattr(sam, 'fiscal_deficit', 71300.0)/1e4:,.1f} 万亿</td></tr>
        <tr><td class="label">1年期LPR</td><td class="value">{getattr(sam, 'lpr_1y', 3.0):.2f}%</td></tr>
        <tr><td class="label">部门数</td><td class="value">42</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 操作流程")
    st.markdown("""
    <div style="font-size:14px; color:#4a5568; line-height:2;">
    <b style="color:#3182ce;">1.</b> 「政策配置」设定政策参数与时序<br/>
    <b style="color:#3182ce;">2.</b> 点击「运行模拟」<br/>
    <b style="color:#3182ce;">3.</b> 「模拟结果」查看图表与部门明细<br/>
    <b style="color:#3182ce;">4.</b> 「宏观分析」读财政部专家报告<br/>
    <b style="color:#3182ce;">5.</b> 「产业分析」读产业政策专家报告
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### v2.0 新增功能")
    st.markdown("""
    <div style="font-size:13px; color:#4a5568; line-height:1.8;">
    ✅ 12季度(3年)动态路径<br/>
    ✅ 2025年基准数据<br/>
    ✅ 利率政策工具<br/>
    ✅ TFP全要素生产率冲击<br/>
    ✅ 定向产业投资支持<br/>
    ✅ 财政盈余/赤字动态<br/>
    ✅ 三部门信心指数<br/>
    ✅ 分行业产出动态钻取
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🔑 AI报告生成 API密钥")
    st.markdown("""
    <div style="font-size:12px; color:#718096; line-height:1.6;">
    输入你自己的GLM API Key后，AI报告生成功能将使用你的额度。<br/>
    留空则使用模板生成报告（无需API）。<br/>
    获取密钥：open.bigmodel.cn
    </div>
    """, unsafe_allow_html=True)
    user_api_key = st.text_input(
        "GLM API Key",
        value=st.session_state.get("user_api_key", ""),
        type="password",
        placeholder="粘贴你的GLM API Key...",
        help="密钥仅存储在本次会话中，不会保存到磁盘或发送给第三方。",
    )
    st.session_state["user_api_key"] = user_api_key
    if user_api_key:
        st.success("✅ 已配置API密钥，可生成AI报告", icon="✅")
    else:
        st.info("未配置API密钥，AI报告将降级为模板", icon="ℹ️")


# =====================================================================
#  Tabs
# =====================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 政策配置", "📈 模拟结果", "🏛️ 宏观分析", "🏭 产业分析"
])


# =====================================================================
#  TAB 1: 政策配置
# =====================================================================
with tab1:
    st.markdown("## 政策参数配置")
    st.markdown("""
    <div class="info-box">
    设定政策工具的调整幅度与实施时序。每项政策可独立选择「永久实施」「一次性脉冲」或「预告后实施」。
    正值表示收紧或扩大，负值表示放松或缩减。配置完成后点击「运行模拟」。<br/>
    <b>财政政策</b>调税收与支出；<b>货币政策</b>调利率；<b>结构性政策</b>调TFP与定向投资。
    </div>
    """, unsafe_allow_html=True)

    # ---- 财政政策工具 ----
    st.markdown("### 📑 财政政策工具")

    POLICIES = [
        ("consumption_tax", "消费税率", "百分点", -5.0, 5.0, 0.0, 0.1, "正值=加税，负值=减税"),
        ("gov_spending", "政府支出", "%", -50.0, 50.0, 0.0, 1.0, "正值=扩大支出，负值=缩减"),
        ("corporate_tax", "企业所得税率", "百分点", -10.0, 10.0, 0.0, 0.5, "正值=加税，负值=减税"),
        ("production_tax", "生产税率", "百分点", -5.0, 5.0, 0.0, 0.1, "正值=加税，负值=减税"),
    ]

    policy_values = {}
    policy_timings = {}
    policy_impl_lags = {}

    for key, label, unit, vmin, vmax, vdef, vstep, help_text in POLICIES:
        with st.container():
            col_val, col_time, col_lag = st.columns([2, 1.5, 1])
            with col_val:
                val = st.slider(
                    f"{label}（{unit}）",
                    min_value=vmin, max_value=vmax, value=vdef, step=vstep,
                    help=help_text, key=f"slider_{key}"
                )
                policy_values[key] = val
            with col_time:
                timing = st.selectbox(
                    "实施时序",
                    options=list(TIMING_OPTIONS.keys()),
                    format_func=lambda x: TIMING_OPTIONS[x],
                    key=f"timing_{key}"
                )
                policy_timings[key] = timing
            with col_lag:
                if timing == "anticipated":
                    lag = st.number_input("预告间隔（季度）", min_value=1, max_value=6, value=2, key=f"lag_{key}")
                    policy_impl_lags[key] = lag
                else:
                    st.write("")
                    policy_impl_lags[key] = 0

            if abs(val) > 0.01:
                d = "加税" if ("税" in label and val > 0) else ("减税" if "税" in label else ("扩大" if val > 0 else "缩减"))
                if key == "gov_spending":
                    gov_total = sam_summary['government_consumption'] / 1e4
                    st.markdown(f"<span style='color:#3182ce; font-size:13px;'>→ {label}{d}{abs(val)}{unit}（约{gov_total*abs(val)/100:,.0f}亿元，{TIMING_OPTIONS[timing]}）</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:#3182ce; font-size:13px;'>→ {label}{d}{abs(val)}{unit}（{TIMING_OPTIONS[timing]}）</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ---- 货币政策工具 ----
    st.markdown("### 💰 货币政策工具")

    col_ir1, col_ir2, col_ir3 = st.columns([2, 1.5, 1])
    with col_ir1:
        ir_val = st.slider(
            "利率调整（基点）", min_value=-200, max_value=200, value=0, step=10,
            help="正值=加息，负值=降息。100基点=1个百分点。当前1年期LPR=3.00%",
            key="slider_interest_rate"
        )
    with col_ir2:
        ir_timing = st.selectbox(
            "实施时序", options=list(TIMING_OPTIONS.keys()),
            format_func=lambda x: TIMING_OPTIONS[x], key="timing_interest_rate"
        )
        policy_timings["interest_rate"] = ir_timing
    with col_ir3:
        if ir_timing == "anticipated":
            ir_lag = st.number_input("预告间隔（季度）", min_value=1, max_value=6, value=2, key="lag_interest_rate")
            policy_impl_lags["interest_rate"] = ir_lag
        else:
            st.write("")
            policy_impl_lags["interest_rate"] = 0

    if abs(ir_val) > 5:
        d = "加息" if ir_val > 0 else "降息"
        st.markdown(f"<span style='color:#3182ce; font-size:13px;'>→ {d}{abs(ir_val)}基点（新LPR={getattr(sam,'lpr_1y',3.0)+ir_val/100:.2f}%，{TIMING_OPTIONS[ir_timing]}）</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ---- 结构性政策工具 ----
    st.markdown("### 🔧 结构性政策工具")

    # TFP冲击
    st.markdown("#### 全要素生产率(TFP)冲击")
    st.markdown("""
    <div class="info-box" style="font-size:13px;">
    模拟新质生产力政策对全要素生产率的影响。可选择全行业或特定行业。<br/>
    注：TFP本质上是内生变量，此处假设可被政策外生调整（如技术创新投资、制度改进）。
    </div>
    """, unsafe_allow_html=True)

    col_tfp1, col_tfp2, col_tfp3 = st.columns([1.5, 2, 1.5])
    with col_tfp1:
        tfp_val = st.slider(
            "TFP变化（%）", min_value=-5.0, max_value=5.0, value=0.0, step=0.1,
            help="正值=生产率提升，负值=生产率下降", key="slider_tfp"
        )
    with col_tfp2:
        tfp_scope = st.selectbox(
            "作用范围",
            options=["全行业"] + SECTOR_OPTIONS,
            key="tfp_scope"
        )
    with col_tfp3:
        tfp_timing = st.selectbox(
            "实施时序", options=list(TIMING_OPTIONS.keys()),
            format_func=lambda x: TIMING_OPTIONS[x], key="timing_tfp"
        )
        policy_timings["tfp"] = tfp_timing
        if tfp_timing == "anticipated":
            tfp_lag = st.number_input("预告间隔（季度）", min_value=1, max_value=6, value=2, key="lag_tfp")
            policy_impl_lags["tfp"] = tfp_lag
        else:
            policy_impl_lags["tfp"] = 0

    if abs(tfp_val) > 0.01:
        scope_str = tfp_scope if tfp_scope == "全行业" else tfp_scope
        d = "提升" if tfp_val > 0 else "下降"
        st.markdown(f"<span style='color:#3182ce; font-size:13px;'>→ TFP{d}{abs(tfp_val):.1f}%（{scope_str}，{TIMING_OPTIONS[tfp_timing]}）</span>", unsafe_allow_html=True)

    st.markdown("")

    # 定向产业投资支持
    st.markdown("#### 定向产业投资支持")
    st.markdown("""
    <div class="info-box" style="font-size:13px;">
    对一个或多个行业实施定向投资补贴/抑制。点击「+ 添加行业」可添加多行。政府承担部分投资成本（假设补贴率30%）。
    </div>
    """, unsafe_allow_html=True)

    # 定向投资时序（全局，所有行业共用）
    col_ti_time, _ = st.columns([1.5, 3])
    with col_ti_time:
        ti_timing = st.selectbox(
            "实施时序（适用于所有定向投资行）", options=list(TIMING_OPTIONS.keys()),
            format_func=lambda x: TIMING_OPTIONS[x], key="timing_targeted"
        )
        policy_timings["targeted"] = ti_timing
        if ti_timing == "anticipated":
            ti_lag = st.number_input("预告间隔（季度）", min_value=1, max_value=6, value=2, key="lag_targeted")
            policy_impl_lags["targeted"] = ti_lag
        else:
            policy_impl_lags["targeted"] = 0

    # ---- 动态多行业定向投资列表 ----
    if "ti_rows" not in st.session_state:
        st.session_state["ti_rows"] = [{"sector": SECTOR_OPTIONS[5], "val": 0.0}]

    # 渲染每一行
    rows_to_remove = []
    for row_idx, row_data in enumerate(st.session_state["ti_rows"]):
        col_sec, col_val, col_del, col_info = st.columns([2.5, 2, 0.6, 2.5])
        with col_sec:
            row_data["sector"] = st.selectbox(
                f"行业 {row_idx+1}", options=SECTOR_OPTIONS,
                index=SECTOR_OPTIONS.index(row_data["sector"]) if row_data["sector"] in SECTOR_OPTIONS else 0,
                key=f"ti_sector_{row_idx}"
            )
        with col_val:
            row_data["val"] = st.slider(
                f"投资增幅（%）", min_value=-50.0, max_value=50.0,
                value=float(row_data["val"]), step=1.0,
                help="正值=投资扩张，负值=投资收缩",
                key=f"ti_val_{row_idx}"
            )
        with col_del:
            if len(st.session_state["ti_rows"]) > 1:
                if st.button("✕", key=f"ti_del_{row_idx}", help="删除此行"):
                    rows_to_remove.append(row_idx)
        with col_info:
            if abs(row_data["val"]) > 0.01:
                sec_idx = int(row_data["sector"].split()[0][1:]) - 1
                base_inv = sam.investment[sec_idx]
                d = "扩张" if row_data["val"] > 0 else "收缩"
                st.markdown(f"<span style='color:#3182ce; font-size:12px;'>→ {d}{abs(row_data['val']):.0f}%（基准{base_inv:,.0f}亿）</span>", unsafe_allow_html=True)

    # 处理删除（从后往前删，避免索引错位）
    for idx in sorted(rows_to_remove, reverse=True):
        if len(st.session_state["ti_rows"]) > 1:
            st.session_state["ti_rows"].pop(idx)
            st.rerun()

    # 添加按钮
    col_add, _ = st.columns([1, 4])
    with col_add:
        if st.button("➕ 添加行业", key="ti_add", help="添加一行定向投资"):
            st.session_state["ti_rows"].append({"sector": SECTOR_OPTIONS[6], "val": 0.0})
            st.rerun()

    # 汇总有效行（投资增幅非零的行业）
    ti_active_rows = [
        r for r in st.session_state["ti_rows"] if abs(r["val"]) > 0.01
    ]

    st.markdown("---")

    # ---- 模型设置 ----
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        closure = st.selectbox(
            "劳动市场闭合",
            options=["keynesian", "neoclassical"],
            format_func=lambda x: "凯恩斯模式（固定工资，就业内生）" if x == "keynesian" else "新古典模式（充分就业，工资内生）",
            help="凯恩斯模式允许财政乘数效应；新古典模式下就业固定"
        )
    with col_m2:
        st.write("")

    # ---- 当前方案摘要 ----
    st.markdown("### 当前政策方案")
    active_policies = []
    for key, label, unit, *_ in POLICIES:
        v = policy_values[key]
        if abs(v) > 0.01:
            active_policies.append((key, label, v, unit, policy_timings[key]))

    # 新增工具
    if abs(ir_val) > 5:
        active_policies.append(("interest_rate", "利率调整", ir_val, "基点", ir_timing))
    if abs(tfp_val) > 0.01:
        active_policies.append(("tfp", "TFP冲击", tfp_val, "%", tfp_timing))
    for r in ti_active_rows:
        sec_short = r["sector"][:12]
        active_policies.append(("targeted", f"定向投资({sec_short})", r["val"], "%", ti_timing))

    if active_policies:
        rows = ""
        for key, label, v, unit, timing in active_policies:
            if key == "interest_rate":
                d = "加息" if v > 0 else "降息"
                rows += f"<tr><td style='padding:6px 12px; color:#4a5568;'>{label}</td><td style='padding:6px 12px; color:#2c5282; font-weight:700; text-align:right;'>{d}{abs(v)}{unit}</td><td style='padding:6px 12px; color:#718096;'>{TIMING_OPTIONS[timing]}</td></tr>"
            elif key in ("tfp", "targeted"):
                d = "提升" if v > 0 else "降低" if key == "tfp" else "扩张" if v > 0 else "收缩"
                rows += f"<tr><td style='padding:6px 12px; color:#4a5568;'>{label}</td><td style='padding:6px 12px; color:#2c5282; font-weight:700; text-align:right;'>{d}{abs(v)}{unit}</td><td style='padding:6px 12px; color:#718096;'>{TIMING_OPTIONS[timing]}</td></tr>"
            else:
                d = "提高" if v > 0 else "降低"
                if key == "gov_spending":
                    d = "增加" if v > 0 else "减少"
                rows += f"<tr><td style='padding:6px 12px; color:#4a5568;'>{label}</td><td style='padding:6px 12px; color:#2c5282; font-weight:700; text-align:right;'>{d}{abs(v)}{unit}</td><td style='padding:6px 12px; color:#718096;'>{TIMING_OPTIONS[timing]}</td></tr>"
        st.markdown(f"""
        <div class="info-box">
        <table style="width:100%; font-size:15px;">{rows}</table>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-box">⚠️ 当前未设定任何政策调整。请移动上方滑块设定至少一项政策。</div>
        """, unsafe_allow_html=True)

    # ---- 运行按钮 ----
    st.markdown("---")
    run_clicked = st.button("🚀 运行模拟", type="primary", use_container_width=True)

    if run_clicked:
        try:
            with st.spinner("正在求解..."):
                if builder.solver.closure != closure:
                    from cge_core.johansen import JohansenSolver, DynamicJohansenSolver
                    builder.solver = JohansenSolver(sam, closure=closure)
                    builder.dyn_solver = DynamicJohansenSolver(sam, horizon=12)
                    builder.dyn_solver.base_solver = builder.solver

                # 构建冲击参数
                shock_params = {
                    "consumption_tax_change": policy_values["consumption_tax"] / 100.0,
                    "gov_spending_change": policy_values["gov_spending"] / 100.0,
                    "corporate_tax_change": policy_values["corporate_tax"] / 100.0,
                    "production_tax_change": policy_values["production_tax"] / 100.0,
                }

                # 利率政策
                if abs(ir_val) > 5:
                    shock_params["interest_rate_change"] = ir_val / 10000.0  # bp → fractional pp

                # TFP冲击
                if abs(tfp_val) > 0.01:
                    if tfp_scope == "全行业":
                        shock_params["tfp_shock"] = tfp_val / 100.0
                    else:
                        idx = int(tfp_scope.split()[0][1:]) - 1
                        shock_params["tfp_shock"] = {idx: tfp_val / 100.0}

                # 定向投资（多行业）
                if ti_active_rows:
                    ti_dict = {}
                    for r in ti_active_rows:
                        idx = int(r["sector"].split()[0][1:]) - 1
                        ti_dict[idx] = r["val"] / 100.0
                    shock_params["targeted_investments"] = ti_dict

                t0 = time.time()
                static_result = builder.solve_shock(shock_params)
                solve_time = time.time() - t0

                # 动态求解：取最强冲击的时序作为主导时序
                all_policy_vals = {
                    "consumption_tax": abs(policy_values["consumption_tax"]),
                    "gov_spending": abs(policy_values["gov_spending"]),
                    "corporate_tax": abs(policy_values["corporate_tax"]),
                    "production_tax": abs(policy_values["production_tax"]),
                    "interest_rate": abs(ir_val) / 100.0,
                    "tfp": abs(tfp_val),
                    "targeted": max((abs(r["val"]) for r in ti_active_rows), default=0),
                }
                max_key = max(all_policy_vals, key=lambda k: all_policy_vals[k])
                dominant_timing = policy_timings.get(max_key, "permanent")
                dominant_lag = policy_impl_lags.get(max_key, 0)

                dyn_result = builder.solve_dynamic(
                    shock_params,
                    shock_timing=dominant_timing,
                    implementation_lag=dominant_lag,
                )

                st.session_state["static_result"] = static_result
                st.session_state["dyn_result"] = dyn_result
                st.session_state["shock_params"] = shock_params
                st.session_state["solve_time"] = solve_time
                st.session_state["has_result"] = True
                st.session_state["policy_timings"] = policy_timings
                st.session_state["closure"] = closure
                st.session_state["dominant_timing"] = dominant_timing
                st.session_state["active_policies_summary"] = active_policies

                # 清除旧报告
                st.session_state.pop("macro_report", None)
                st.session_state.pop("industry_report", None)

            st.markdown(f"""
            <div class="success-box" style="font-size:15px;">
            ✅ <b>模拟完成</b>，用时 {solve_time*1000:.0f} 毫秒。<br/>
            请点击「📈 模拟结果」查看图表，或点击「🏛️ 宏观分析」/「🏭 产业分析」生成AI报告。
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"模拟出错: {e}")
            import traceback
            with st.expander("错误详情"):
                st.code(traceback.format_exc())


# =====================================================================
#  TAB 2: 模拟结果
# =====================================================================
with tab2:
    if not st.session_state.get("has_result"):
        st.markdown("""
        <div class="alert-box" style="text-align:center; font-size:15px; padding:40px;">
        ⚠️ <b>尚未运行模拟</b><br/>请先在「政策配置」页签设定参数并运行模拟。
        </div>
        """, unsafe_allow_html=True)
    else:
        result = st.session_state["static_result"]
        dyn = st.session_state["dyn_result"]
        params = st.session_state["shock_params"]

        # ---- 核心指标 ----
        st.markdown("## 核心指标（静态均衡效应）")
        c1, c2, c3, c4, c5 = st.columns(5)

        def metric_card(col, label, value, good_pos=True, fmt=".3f", unit="%"):
            with col:
                is_pos = value >= 0
                if good_pos:
                    cls = "pos" if is_pos else "neg"
                else:
                    cls = "neg" if is_pos else "pos"
                arrow = "▲" if is_pos else "▼"
                color = "#2f855a" if cls == "pos" else "#c53030"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color};">{arrow} {abs(value):{fmt}}{unit}</div>
                </div>
                """, unsafe_allow_html=True)

        metric_card(c1, "GDP变化", result.gdp_change * 100)
        metric_card(c2, "就业变化", result.employment_change * 100)
        metric_card(c3, "CPI变化", result.cpi_change * 100, good_pos=False)
        metric_card(c4, "福利变化(EV)", result.welfare_change * 100)
        metric_card(c5, "财政平衡变化", getattr(result, 'fiscal_balance_pct', 0.0), fmt=".2f", unit="pp")

        st.markdown("")

        # ---- 信心指数 ----
        st.markdown("## 三部门信心指数变化")
        st.markdown("""
        <div class="info-box" style="font-size:13px;">
        基准值=50（中性）。正值=信心改善，负值=信心恶化。信心指数受GDP、CPI、就业、利率、财政等多因素驱动。
        </div>
        """, unsafe_allow_html=True)

        cc1, cc2, cc3 = st.columns(3)
        metric_card(cc1, "消费者信心", getattr(result, 'confidence_consumer', 0.0), fmt=".2f", unit="")
        metric_card(cc2, "企业信心", getattr(result, 'confidence_enterprise', 0.0), fmt=".2f", unit="")
        metric_card(cc3, "投资者信心", getattr(result, 'confidence_investor', 0.0), fmt=".2f", unit="")

        st.markdown("---")

        # ---- 12季度动态路径 ----
        st.markdown("## 12季度(3年)动态过渡路径")
        quarters = dyn["quarters"]
        dev = dyn["deviation"]
        fiscal_path = dyn.get("fiscal_path", np.zeros(12))
        conf_path = dyn.get("confidence_path", {})

        chart_layout = {
            "template": "plotly_white",
            "font": {"family": "Microsoft YaHei, sans-serif", "size": 12, "color": "#2d3748"},
            "xaxis": {
                "title": "政策实施后季度", "tickmode": "linear", "dtick": 1,
                "gridcolor": "#e2e8f0", "range": [0.5, 12.5],
            },
            "yaxis": {
                "title": "相对基准偏差（%）", "zeroline": True,
                "zerolinecolor": "#a0aec0", "zerolinewidth": 1.5,
                "gridcolor": "#e2e8f0",
            },
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            "margin": {"l": 60, "r": 30, "t": 50, "b": 40},
            "height": 320,
        }

        col_a, col_b = st.columns(2)

        def make_chart(title, data, color, fillcolor):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=quarters, y=[v * 100 for v in data],
                mode="lines+markers", name=title,
                line=dict(color=color, width=2.5),
                marker=dict(size=6), fill="tozeroy", fillcolor=fillcolor,
            ))
            fig.update_layout(title=title, **chart_layout)
            return fig

        with col_a:
            st.plotly_chart(make_chart("GDP路径", dev["gdp"], "#3182ce", "rgba(49,130,206,0.08)"), use_container_width=True)
            st.plotly_chart(make_chart("就业路径", dev["employment"], "#2f855a", "rgba(47,133,90,0.08)"), use_container_width=True)
            # 财政平衡路径
            fig_fiscal = go.Figure()
            fig_fiscal.add_trace(go.Scatter(
                x=quarters, y=[v for v in fiscal_path],
                mode="lines+markers", name="财政平衡/GDP变化",
                line=dict(color="#d69e2e", width=2.5),
                marker=dict(size=6), fill="tozeroy", fillcolor="rgba(214,158,46,0.08)",
            ))
            fig_fiscal.update_layout(
                title="财政平衡/GDP变化路径（百分点）",
                yaxis={"title": "财政平衡变化（pp）", "zeroline": True, "zerolinecolor": "#a0aec0", "zerolinewidth": 1.5, "gridcolor": "#e2e8f0"},
                **{k: v for k, v in chart_layout.items() if k != "yaxis"}
            )
            st.plotly_chart(fig_fiscal, use_container_width=True)

        with col_b:
            st.plotly_chart(make_chart("CPI路径", dev["cpi"], "#dd6b20", "rgba(221,107,32,0.08)"), use_container_width=True)
            st.plotly_chart(make_chart("投资路径", dev["investment"], "#805ad5", "rgba(128,90,213,0.08)"), use_container_width=True)
            # 信心指数路径（三条线）
            fig_conf = go.Figure()
            if conf_path:
                for label, key, color in [
                    ("消费者信心", "consumer", "#3182ce"),
                    ("企业信心", "enterprise", "#2f855a"),
                    ("投资者信心", "investor", "#dd6b20"),
                ]:
                    data = conf_path.get(key, np.zeros(12))
                    fig_conf.add_trace(go.Scatter(
                        x=quarters, y=[v for v in data],
                        mode="lines+markers", name=label,
                        line=dict(color=color, width=2),
                        marker=dict(size=5),
                    ))
            fig_conf.update_layout(
                title="三部门信心指数动态路径",
                yaxis={"title": "信心指数变化（点）", "zeroline": True, "zerolinecolor": "#a0aec0", "zerolinewidth": 1.5, "gridcolor": "#e2e8f0"},
                **{k: v for k, v in chart_layout.items() if k != "yaxis"}
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        st.markdown("---")

        # ---- 分行业产出动态钻取 ----
        st.markdown("## 分行业12季度产出动态")
        st.markdown("""
        <div class="info-box" style="font-size:13px;">
        选择一个行业，查看其12季度产出动态曲线。蓝线=该行业产出偏差(%), 虚线=同期GDP偏差(对比)。
        </div>
        """, unsafe_allow_html=True)

        sector_output_path = dyn.get("sector_output_path", np.zeros((42, 12)))

        col_sel, col_info = st.columns([2, 3])
        with col_sel:
            selected_sector = st.selectbox(
                "选择行业", options=SECTOR_OPTIONS, key="drilldown_sector"
            )
        with col_info:
            sec_idx = int(selected_sector.split()[0][1:]) - 1
            sec_static_chg = result.pct_changes["qo"][sec_idx] * 100
            sec_price_chg = result.pct_changes["po"][sec_idx] * 100
            sec_cons_chg = result.pct_changes["qh"][sec_idx] * 100
            st.markdown(f"""
            <div class="info-box">
            <b>{selected_sector}</b><br/>
            静态产出变化: <b style="color:{"#2f855a" if sec_static_chg >= 0 else "#c53030"};">{sec_static_chg:+.4f}%</b> |
            价格变化: <b style="color:{"#2f855a" if sec_price_chg >= 0 else "#c53030"};">{sec_price_chg:+.4f}%</b> |
            消费变化: <b style="color:{"#2f855a" if sec_cons_chg >= 0 else "#c53030"};">{sec_cons_chg:+.4f}%</b>
            </div>
            """, unsafe_allow_html=True)

        # 行业产出动态曲线
        fig_drill = go.Figure()
        sec_path = sector_output_path[sec_idx] * 100
        fig_drill.add_trace(go.Scatter(
            x=quarters, y=sec_path,
            mode="lines+markers", name=f"{SECTOR_NAMES_CN[sec_idx]} 产出偏差",
            line=dict(color="#3182ce", width=3),
            marker=dict(size=8),
            fill="tozeroy", fillcolor="rgba(49,130,206,0.1)",
        ))
        # GDP对比线
        gdp_path_pct = [v * 100 for v in dev["gdp"]]
        fig_drill.add_trace(go.Scatter(
            x=quarters, y=gdp_path_pct,
            mode="lines", name="GDP偏差（对比）",
            line=dict(color="#a0aec0", width=1.5, dash="dash"),
        ))
        fig_drill.update_layout(
            title=f"{SECTOR_NAMES_CN[sec_idx]} — 12季度产出动态路径",
            xaxis={"title": "政策实施后季度", "tickmode": "linear", "dtick": 1, "gridcolor": "#e2e8f0", "range": [0.5, 12.5]},
            yaxis={"title": "相对基准偏差（%）", "zeroline": True, "zerolinecolor": "#a0aec0", "zerolinewidth": 1.5, "gridcolor": "#e2e8f0"},
            template="plotly_white",
            font={"family": "Microsoft YaHei, sans-serif", "size": 12, "color": "#2d3748"},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            height=380,
            margin={"l": 60, "r": 30, "t": 50, "b": 40},
        )
        st.plotly_chart(fig_drill, use_container_width=True)

        st.markdown("---")

        # ---- 42部门热力图 ----
        st.markdown("## 42部门产出变化热力图（静态均衡）")
        sector_changes = result.pct_changes["qo"] * 100

        n_cols_grid = 7
        n_rows_grid = 6
        heatmap_data = np.zeros((n_rows_grid, n_cols_grid))
        heatmap_labels = np.empty((n_rows_grid, n_cols_grid), dtype=object)
        heatmap_short = np.empty((n_rows_grid, n_cols_grid), dtype=object)

        for i in range(42):
            r, c = divmod(i, n_cols_grid)
            heatmap_data[r, c] = sector_changes[i]
            short = SECTOR_NAMES_CN[i][:6] if len(SECTOR_NAMES_CN[i]) > 6 else SECTOR_NAMES_CN[i]
            heatmap_short[r, c] = short
            heatmap_labels[r, c] = f"S{i+1:02d} {SECTOR_NAMES_CN[i]}<br>{sector_changes[i]:+.3f}%"

        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_data, text=heatmap_short,
            texttemplate="<b>%{text}</b><br>%{z:+.2f}%",
            textfont={"size": 10, "color": "#000"},
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="产出变化%", tickfont=dict(size=11)),
        ))
        fig_heat.update_layout(
            template="plotly_white",
            font=dict(family="Microsoft YaHei", size=11, color="#2d3748"),
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ---- 42部门明细表 ----
        st.markdown("## 42部门详细数据")
        df_sectors = pd.DataFrame({
            "代码": [f"S{i+1:02d}" for i in range(42)],
            "部门名称": SECTOR_NAMES_CN,
            "英文名称": [s[2] for s in SECTORS_42],
            "产出变化(%)": np.round(sector_changes, 4),
            "价格变化(%)": np.round(result.pct_changes["po"] * 100, 4),
            "消费变化(%)": np.round(result.pct_changes["qh"] * 100, 4),
        })
        df_sectors = df_sectors.sort_values("产出变化(%)", ascending=False)
        st.dataframe(df_sectors, use_container_width=True, hide_index=True)

        csv = df_sectors.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥 导出CSV", data=csv, file_name=f"cge_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")


# =====================================================================
#  TAB 3: 宏观分析（财政部专家）
# =====================================================================
with tab3:
    if not st.session_state.get("has_result"):
        st.markdown('<div class="alert-box" style="text-align:center; font-size:15px; padding:40px;">⚠️ <b>尚未运行模拟</b><br/>请先在「政策配置」页签运行模拟。</div>', unsafe_allow_html=True)
    else:
        st.markdown("## 宏观经济分析报告")
        st.markdown("""
        <div class="info-box">
        <b>专家设定：</b>审慎、中立、客观的财政部宏观经济分析专家。<br/>
        分析维度：宏观经济态势、财政收支平衡、物价与就业、信心指数、动态过渡路径、政策风险与建议。
        </div>
        """, unsafe_allow_html=True)

        col_btn, _ = st.columns([1, 2])
        with col_btn:
            gen_macro = st.button("📝 生成宏观分析报告", type="primary", use_container_width=True)

        if gen_macro:
            with st.spinner("正在调用AI生成宏观分析报告..."):
                from dashboard.report_generator import generate_policy_report
                try:
                    report = generate_policy_report(
                        static_result=st.session_state["static_result"],
                        dyn_result=st.session_state["dyn_result"],
                        shock_params=st.session_state["shock_params"],
                        sam_summary=sam_summary,
                        closure=st.session_state.get("closure", "keynesian"),
                        shock_timing=st.session_state.get("dominant_timing", "permanent"),
                        policy_timings=st.session_state.get("policy_timings", {}),
                        use_ai=True,
                        persona="macro",
                        api_key=st.session_state.get("user_api_key", ""),
                    )
                    st.session_state["macro_report"] = report
                except Exception as e:
                    st.error(f"报告生成失败: {e}")
                    report = generate_policy_report(
                        static_result=st.session_state["static_result"],
                        dyn_result=st.session_state["dyn_result"],
                        shock_params=st.session_state["shock_params"],
                        sam_summary=sam_summary,
                        use_ai=False, persona="macro",
                    )
                    st.session_state["macro_report"] = report

        report = st.session_state.get("macro_report")
        if report:
            st.markdown(f"""
            <div style='text-align:right; color:#718096; font-size:13px; margin-bottom:8px;'>
            生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}
            </div>
            <div class="report-content">{report}</div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            st.download_button("📥 导出报告", data=report.encode("utf-8"), file_name=f"宏观分析报告_{datetime.now().strftime('%Y%m%d_%H%M')}.md", mime="text/markdown")


# =====================================================================
#  TAB 4: 产业分析（产业政策专家）
# =====================================================================
with tab4:
    if not st.session_state.get("has_result"):
        st.markdown('<div class="alert-box" style="text-align:center; font-size:15px; padding:40px;">⚠️ <b>尚未运行模拟</b><br/>请先在「政策配置」页签运行模拟。</div>', unsafe_allow_html=True)
    else:
        st.markdown("## 产业影响分析报告")
        st.markdown("""
        <div class="info-box">
        <b>专家设定：</b>资深产业政策专家，聚焦42部门差异化影响。<br/>
        分析维度：产业链传导机制、重点受益行业深度分析、受压行业风险提示、产业结构影响、产业政策建议。
        </div>
        """, unsafe_allow_html=True)

        col_btn, _ = st.columns([1, 2])
        with col_btn:
            gen_ind = st.button("📝 生成产业分析报告", type="primary", use_container_width=True)

        if gen_ind:
            with st.spinner("正在调用AI生成产业分析报告..."):
                from dashboard.report_generator import generate_policy_report
                try:
                    report = generate_policy_report(
                        static_result=st.session_state["static_result"],
                        dyn_result=st.session_state["dyn_result"],
                        shock_params=st.session_state["shock_params"],
                        sam_summary=sam_summary,
                        closure=st.session_state.get("closure", "keynesian"),
                        shock_timing=st.session_state.get("dominant_timing", "permanent"),
                        policy_timings=st.session_state.get("policy_timings", {}),
                        use_ai=True,
                        persona="industry",
                        api_key=st.session_state.get("user_api_key", ""),
                    )
                    st.session_state["industry_report"] = report
                except Exception as e:
                    st.error(f"报告生成失败: {e}")
                    report = generate_policy_report(
                        static_result=st.session_state["static_result"],
                        dyn_result=st.session_state["dyn_result"],
                        shock_params=st.session_state["shock_params"],
                        sam_summary=sam_summary,
                        use_ai=False, persona="industry",
                    )
                    st.session_state["industry_report"] = report

        report = st.session_state.get("industry_report")
        if report:
            st.markdown(f"""
            <div style='text-align:right; color:#718096; font-size:13px; margin-bottom:8px;'>
            生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}
            </div>
            <div class="report-content">{report}</div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            st.download_button("📥 导出报告", data=report.encode("utf-8"), file_name=f"产业分析报告_{datetime.now().strftime('%Y%m%d_%H%M')}.md", mime="text/markdown")
