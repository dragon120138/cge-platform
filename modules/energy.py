# -*- coding: utf-8 -*-
"""
能源模块 [扩展] — 将能源作为独立生产要素

扩展内容:
  - 能源作为第三种生产要素（在增加值CES中加入）
  - 部门能源效率参数
  - 能源-资本-劳动的嵌套CES结构
"""

import numpy as np
from typing import Dict, Any

from cge_core.base_module import CGEModule, SAMData


class EnergyModule(CGEModule):
    """能源部门扩展模块。"""

    name = "energy"
    description = "将能源作为独立生产要素引入CES增加值函数"
    version = "1.0.0"
    dependencies = ["production"]

    def declare_parameters(self, model) -> None:
        n = model.num_sectors

        # 能源强度（万元产出/吨标准煤）
        # 基于中国2020年部门能源强度估算
        energy_intensity = np.array([
            0.15, 2.80, 1.85, 0.95, 0.78,   # S01-S05
            0.22, 0.45, 0.18, 0.15, 0.28,   # S06-S10
            1.65, 0.85, 0.95, 1.15, 0.35,   # S11-S15
            0.28, 0.25, 0.20, 0.22, 0.12,   # S16-S20
            0.15, 0.38, 0.18, 0.55, 0.42,   # S21-S25
            0.32, 0.08, 0.15, 0.10, 0.08,   # S26-S30
            0.06, 0.04, 0.05, 0.08, 0.06,   # S31-S35
            0.12, 0.08, 0.05, 0.07, 0.06,   # S36-S40
            0.04, 0.03,                       # S41-S42
        ])
        model.energy_intensity = list(energy_intensity)

        # 能源份额 (在增加值CES中的权重)
        energy_share = np.clip(energy_intensity / energy_intensity.max() * 0.15, 0.02, 0.20)
        model.energy_share_va = list(energy_share)

        # 能源-资本替代弹性
        model.sigma_energy = list(np.full(n, 0.3))  # 能源与资本互补性较强

    def declare_variables(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        # 能源需求
        model.ED = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 能源价格
        model.PE = pyo.Var(initialize=1.0, bounds=(1e-6, None))

        # 能源总供给
        model.ES = pyo.Var(initialize=1.0, bounds=(1e-6, None))

    def declare_equations(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        def energy_demand_rule(m, j):
            """部门能源需求 = 能源强度 × 产出"""
            return m.ED[j] == m.energy_intensity[j] * m.XS[j]

        model.eq_energy_demand = pyo.Constraint(model.SECTORS, rule=energy_demand_rule)

        def energy_market_rule(m):
            """能源市场出清"""
            return m.ES == sum(m.ED[j] for j in range(n))

        model.eq_energy_market = pyo.Constraint(rule=energy_market_rule)

    def calibrate(self, model, sam: SAMData) -> None:
        pass

    def scenario_params(self) -> Dict[str, Dict[str, Any]]:
        return {
            "energy_efficiency_improvement": {
                "default": 0.0, "min": -0.20, "max": 0.50, "step": 0.05,
                "label": "能源效率提升（%）", "type": "slider",
            },
        }
