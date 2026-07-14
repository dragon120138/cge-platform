# -*- coding: utf-8 -*-
"""
图表组件 — Plotly交互式时间序列图

所有图表遵循统一设计规范:
  - 基线(虚线蓝色) vs 反事实(实线橙色)
  - 零线突出显示
  - 月份x轴(1-12)
  - 百分比偏离y轴
"""

import plotly.graph_objects as go
import numpy as np
from typing import Optional, List


# 统一图表布局
CHART_LAYOUT = {
    "template": "plotly_white",
    "title": {"font": {"family": "Microsoft YaHei, Noto Sans SC", "size": 18}},
    "xaxis": {
        "title": "政策实施后月份",
        "tickmode": "linear",
        "dtick": 1,
        "range": [0.5, 12.5],
        "gridcolor": "#E2E8F0",
        "title_font": {"size": 14},
        "tickfont": {"size": 12},
    },
    "yaxis": {
        "title": "相对基线偏离 (%)",
        "zeroline": True,
        "zerolinecolor": "#64748B",
        "zerolinewidth": 2,
        "gridcolor": "#E2E8F0",
        "title_font": {"size": 14},
        "tickfont": {"size": 12},
    },
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
        "font": {"size": 12},
    },
    "margin": {"l": 60, "r": 30, "t": 60, "b": 40},
    "height": 350,
    "plot_bgcolor": "#FAFAFA",
    "paper_bgcolor": "white",
}

# 色板
COLORS = {
    "baseline": "#3B82F6",      # 蓝色
    "counterfactual": "#F97316", # 橙色
    "positive": "#10B981",       # 绿色
    "negative": "#EF4444",       # 红色
    "neutral": "#6B7280",        # 灰色
    "highlight": "#8B5CF6",      # 紫色
}


def plot_time_series(gdp_path, employment_path, cpi_path,
                      title_suffix: str = "") -> go.Figure:
    """绘制三合一时间序列图（GDP、就业、CPI）。

    Args:
        gdp_path: GDP偏离路径(np.array, %)
        employment_path: 就业偏离路径
        cpi_path: CPI偏离路径
        title_suffix: 图表标题后缀

    Returns:
        Plotly Figure
    """
    months = np.arange(1, len(gdp_path) + 1)

    fig = go.Figure()

    # GDP
    fig.add_trace(go.Scatter(
        x=months, y=gdp_path,
        name="GDP",
        mode="lines+markers",
        line=dict(color=COLORS["counterfactual"], width=2.5),
        marker=dict(size=6),
    ))

    # 就业
    fig.add_trace(go.Scatter(
        x=months, y=employment_path,
        name="就业",
        mode="lines+markers",
        line=dict(color=COLORS["baseline"], width=2.5, dash="dash"),
        marker=dict(size=6),
    ))

    # CPI
    fig.add_trace(go.Scatter(
        x=months, y=cpi_path,
        name="CPI",
        mode="lines+markers",
        line=dict(color=COLORS["highlight"], width=2.5, dash="dot"),
        marker=dict(size=6),
    ))

    layout = CHART_LAYOUT.copy()
    layout["title"]["text"] = f"宏观经济指标动态路径{title_suffix}"
    fig.update_layout(**layout)

    return fig


def plot_single_series(data: np.ndarray, name: str, title: str,
                        color: str = None, baseline_data: np.ndarray = None) -> go.Figure:
    """绘制单指标时间序列图（基线vs反事实）。"""
    months = np.arange(1, len(data) + 1)
    fig = go.Figure()

    if baseline_data is not None:
        fig.add_trace(go.Scatter(
            x=months, y=baseline_data,
            name=f"{name}（基线）",
            mode="lines",
            line=dict(color=COLORS["baseline"], width=2, dash="dash"),
            opacity=0.7,
        ))

    fig.add_trace(go.Scatter(
        x=months, y=data,
        name=f"{name}（政策冲击）",
        mode="lines+markers",
        line=dict(color=color or COLORS["counterfactual"], width=2.5),
        marker=dict(size=6),
        fill="tozeroy" if baseline_data is None else None,
        fillcolor=f"rgba{(249, 115, 22, 0.1)}" if baseline_data is None else None,
    ))

    layout = CHART_LAYOUT.copy()
    layout["title"]["text"] = title
    fig.update_layout(**layout)

    return fig


def plot_sectoral_heatmap(sector_dev: np.ndarray, sector_names: List[str],
                           title: str = "部门产出偏离热力图") -> go.Figure:
    """绘制42部门偏离热力图。"""
    # 重排为6×7网格
    n = len(sector_dev)
    ncols = 7
    nrows = (n + ncols - 1) // ncols

    z = np.zeros((nrows, ncols))
    labels = [[""] * ncols for _ in range(nrows)]
    for i in range(n):
        r, c = divmod(i, ncols)
        z[r, c] = sector_dev[i]
        labels[r][c] = f"{sector_names[i]}<br>{sector_dev[i]:+.2f}%"

    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=labels,
        texttemplate="%{text}",
        textfont={"size": 9},
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="偏离(%)"),
    ))

    fig.update_layout(
        title=title,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange="reversed"),
    )

    return fig


def plot_comparison_bar(comparison_data: dict, title: str = "政策效果对比") -> go.Figure:
    """绘制指标对比柱状图。"""
    indicators = list(comparison_data.keys())
    values = list(comparison_data.values())

    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in values]

    fig = go.Figure(data=go.Bar(
        x=indicators,
        y=values,
        marker_color=colors,
        text=[f"{v:+.3f}%" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        yaxis=dict(title="相对基线偏离 (%)", zeroline=True, zerolinecolor="#64748B"),
        template="plotly_white",
        height=350,
        margin=dict(l=60, r=30, t=50, b=60),
        showlegend=False,
    )

    return fig
