# -*- coding: utf-8 -*-
"""
劳动市场分割模块 [扩展] — 技能/非技能劳动力异质性

将单一劳动L分解为:
  - L_S: 技能劳动力（大学及以上）
  - L_U: 非技能劳动力（高中及以下）

两市场不完全替代，各有独立工资和供给弹性。
"""

import numpy as np
from typing import Dict, Any

from cge_core.base_module import CGEModule, SAMData


class LaborSegmentationModule(CGEModule):
    """劳动市场分割模块。"""

    name = "labor_segmentation"
    description = "技能/非技能劳动力异质性，两市场分别出清"
    version = "1.0.0"
    dependencies = ["production"]

    # 部门技能劳动力占比（2020年估算）
    SKILLED_SHARE = np.array([
        0.15, 0.30, 0.35, 0.28, 0.25,   # S01-S05
        0.20, 0.18, 0.15, 0.22, 0.25,   # S06-S10
        0.35, 0.30, 0.25, 0.28, 0.30,   # S11-S15
        0.35, 0.38, 0.32, 0.35, 0.45,   # S16-S20
        0.42, 0.28, 0.30, 0.38, 0.35,   # S21-S25
        0.32, 0.25, 0.30, 0.22, 0.28,   # S26-S30
        0.55, 0.65, 0.50, 0.48, 0.62,   # S31-S35
        0.42, 0.35, 0.72, 0.68, 0.55,   # S36-S40
        0.75, 0.70,                       # S41-S42
    ])

    def declare_parameters(self, model) -> None:
        n = model.num_sectors
        model.skilled_share = list(self.SKILLED_SHARE)
        model.sigma_labor = 1.5  # 技能/非技能替代弹性
        model.labor_supply_elasticity_skilled = 0.15
        model.labor_supply_elasticity_unskilled = 0.10

    def declare_variables(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        # 技能劳动力需求
        model.LDS = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 非技能劳动力需求
        model.LDU = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 技能/非技能工资
        model.WLS = pyo.Var(initialize=1.2, bounds=(1e-6, None))  # 技能溢价
        model.WLU = pyo.Var(initialize=0.8, bounds=(1e-6, None))

        # 劳动总供给
        model.LSS = pyo.Var(initialize=0.3, bounds=(1e-6, None))
        model.LSU = pyo.Var(initialize=0.7, bounds=(1e-6, None))

    def declare_equations(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        def skilled_demand_rule(m, j):
            """技能劳动力需求 = 份额 × 总劳动需求"""
            return m.LDS[j] == m.skilled_share[j] * m.LD[j]

        model.eq_lds = pyo.Constraint(model.SECTORS, rule=skilled_demand_rule)

        def unskilled_demand_rule(m, j):
            """非技能劳动力需求"""
            return m.LDU[j] == (1 - m.skilled_share[j]) * m.LD[j]

        model.eq_ldu = pyo.Constraint(model.SECTORS, rule=unskilled_demand_rule)

        def avg_wage_rule(m):
            """加权平均工资 = 总劳动需求约束"""
            total_skilled = sum(m.LDS[j] for j in range(n))
            total_unskilled = sum(m.LDU[j] for j in range(n))
            return m.WL * m.LS == m.WLS * total_skilled + m.WLU * total_unskilled

        model.eq_avg_wage = pyo.Constraint(rule=avg_wage_rule)

        def skilled_market_rule(m):
            """技能劳动市场出清"""
            return m.LSS == sum(m.LDS[j] for j in range(n))

        model.eq_skilled_market = pyo.Constraint(rule=skilled_market_rule)

        def unskilled_market_rule(m):
            """非技能劳动市场出清"""
            return m.LSU == sum(m.LDU[j] for j in range(n))

        model.eq_unskilled_market = pyo.Constraint(rule=unskilled_market_rule)

    def calibrate(self, model, sam: SAMData) -> None:
        pass

    def scenario_params(self) -> Dict[str, Dict[str, Any]]:
        return {
            "sigma_labor": {
                "default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1,
                "label": "技能/非技能替代弹性", "type": "slider",
            },
        }
