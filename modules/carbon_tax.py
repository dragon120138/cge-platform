# -*- coding: utf-8 -*-
"""
碳税模块 [扩展] — 碳定价与收入回收

功能:
  - 按部门碳排放强度征收碳税
  - 碳税收入可选择一次性返还居民或削减其他税种
  - 内嵌中国42部门碳排放系数
"""

import numpy as np
from typing import Dict, Any

from cge_core.base_module import CGEModule, SAMData


class CarbonTaxModule(CGEModule):
    """碳税扩展模块。"""

    name = "carbon_tax"
    description = "碳定价机制：按CO₂排放征收碳税，收入可回收"
    version = "1.0.0"
    dependencies = ["production", "government"]

    # 中国42部门碳排放强度 (吨CO₂/万元产出，2020年估算)
    EMISSION_INTENSITY = np.array([
        0.08, 4.50, 3.20, 0.85, 0.65,   # S01-S05
        0.15, 0.35, 0.12, 0.10, 0.22,   # S06-S10
        3.80, 1.85, 2.10, 2.45, 0.28,   # S11-S15
        0.20, 0.18, 0.15, 0.16, 0.08,   # S16-S20
        0.10, 0.25, 0.12, 3.50, 0.35,   # S21-S25
        0.22, 0.05, 0.12, 0.08, 0.05,   # S26-S30
        0.04, 0.03, 0.04, 0.06, 0.04,   # S31-S35
        0.08, 0.06, 0.04, 0.05, 0.04,   # S36-S40
        0.03, 0.02,                       # S41-S42
    ])

    def declare_parameters(self, model) -> None:
        model.emission_intensity = list(self.EMISSION_INTENSITY)
        model.carbon_price = self.config.get("carbon_price", 100.0)  # 元/吨CO₂
        model.revenue_recycling = self.config.get("revenue_recycling", "lump_sum")
        # lump_sum: 一次性返还居民
        # tax_cut: 削减企业所得税
        # gov_revenue: 留作政府收入

    def declare_variables(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        # 部门碳排放
        model.CO2 = pyo.Var(model.SECTORS, initialize=1.0, bounds=(0, None))

        # 碳税收入
        model.CARBON_REV = pyo.Var(initialize=0.0, bounds=(0, None))

        # 总排放
        model.TOT_CO2 = pyo.Var(initialize=1.0, bounds=(0, None))

    def declare_equations(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        carbon_price = model.carbon_price / 1e4  # 元/吨 → 亿元/万吨

        def emission_rule(m, j):
            """部门CO₂排放 = 排放强度 × 产出"""
            return m.CO2[j] == m.emission_intensity[j] * m.XS[j]

        model.eq_co2 = pyo.Constraint(model.SECTORS, rule=emission_rule)

        def total_emission_rule(m):
            """总排放"""
            return m.TOT_CO2 == sum(m.CO2[j] for j in range(n))

        model.eq_tot_co2 = pyo.Constraint(rule=total_emission_rule)

        def carbon_revenue_rule(m):
            """碳税收入 = 碳价 × 总排放"""
            return m.CARBON_REV == m.carbon_price * m.TOT_CO2 / 1e4  # 转为亿元

        model.eq_carbon_rev = pyo.Constraint(rule=carbon_revenue_rule)

    def calibrate(self, model, sam: SAMData) -> None:
        pass

    def scenario_params(self) -> Dict[str, Dict[str, Any]]:
        return {
            "carbon_price": {
                "default": 100.0, "min": 0.0, "max": 500.0, "step": 10.0,
                "label": "碳价（元/吨CO₂）", "type": "slider",
            },
            "revenue_recycling": {
                "default": "lump_sum",
                "label": "碳税收入回收方式",
                "type": "select",
            },
        }
