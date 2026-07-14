# -*- coding: utf-8 -*-
"""
金融加速器模块 [扩展] — 信贷约束与投资敏感性

Bernanke-Gertler-Gilchrist (BGG) 金融加速器简化版:
  - 外部融资溢价 = f(杠杆率)
  - 投资对净资产变化敏感
  - 金融冲击放大传导
"""

import numpy as np
from typing import Dict, Any

from cge_core.base_module import CGEModule, SAMData


class FinancialAcceleratorModule(CGEModule):
    """金融加速器模块。"""

    name = "financial_accelerator"
    description = "BGG金融加速器：信贷约束、外部融资溢价、投资敏感性"
    version = "1.0.0"
    dependencies = ["production", "household"]

    def declare_parameters(self, model) -> None:
        n = model.num_sectors

        # 外部融资依赖度（各部门外部融资/总投资）
        external_finance_dependency = np.array([
            0.20, 0.45, 0.50, 0.48, 0.42,   # S01-S05
            0.30, 0.35, 0.32, 0.38, 0.35,   # S06-S10
            0.50, 0.42, 0.45, 0.48, 0.40,   # S11-S15
            0.45, 0.43, 0.46, 0.44, 0.50,   # S16-S20
            0.42, 0.38, 0.35, 0.55, 0.52,   # S21-S25
            0.40, 0.25, 0.30, 0.28, 0.32,   # S26-S30
            0.35, 0.45, 0.55, 0.40, 0.35,   # S31-S35
            0.28, 0.30, 0.20, 0.25, 0.32,   # S36-S40
            0.15, 0.10,                       # S41-S42
        ])
        model.external_finance_dep = list(external_finance_dependency)

        # 加速器参数
        model.accelerator_elasticity = 0.05  # 外部融资溢价对杠杆率的弹性
        model.leverage_ratio = list(np.full(n, 2.0))  # 部门杠杆率（资产/净值）

    def declare_variables(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        # 外部融资溢价（超过无风险利率的部分）
        model.PREMIUM = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1.0, None))

        # 部门净资产
        model.NETWORTH = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 信贷可得性乘数
        model.CREDIT_MULT = pyo.Var(model.SECTORS, initialize=1.0, bounds=(0.1, 5.0))

    def declare_equations(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        def premium_rule(m, j):
            """外部融资溢价 = 1 + 弹性 × 杠杆率
            杠杆率越高，外部融资成本越高
            """
            eps = 1e-10
            leverage = m.KD[j] / (m.NETWORTH[j] + eps)
            return m.PREMIUM[j] == 1.0 + m.accelerator_elasticity * leverage

        model.eq_premium = pyo.Constraint(model.SECTORS, rule=premium_rule)

        def credit_multiplier_rule(m, j):
            """信贷乘数 = f(净资产)
            净资产越高，信贷约束越松
            """
            eps = 1e-10
            return m.CREDIT_MULT[j] == 1.0 + 0.1 * (m.NETWORTH[j] - 1.0)

        model.eq_credit_mult = pyo.Constraint(model.SECTORS, rule=credit_multiplier_rule)

        def networth_rule(m, j):
            """净资产 = 资本存量 - 负债（简化）"""
            eps = 1e-10
            return m.NETWORTH[j] == m.KD[j] * m.WK / (m.leverage_ratio[j] + eps)

        model.eq_networth = pyo.Constraint(model.SECTORS, rule=networth_rule)

    def calibrate(self, model, sam: SAMData) -> None:
        pass

    def scenario_params(self) -> Dict[str, Dict[str, Any]]:
        return {
            "accelerator_elasticity": {
                "default": 0.05, "min": 0.0, "max": 0.20, "step": 0.01,
                "label": "金融加速器弹性", "type": "slider",
            },
        }
