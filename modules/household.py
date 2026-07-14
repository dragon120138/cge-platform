# -*- coding: utf-8 -*-
"""
住户模块 — LES需求系统，代表性居民户

线性支出系统(LES):
  C_i = theta_i + (mu_i / P_i) * (Y - sum_j theta_j * P_j)
  其中:
    theta_i = 生存性支出
    mu_i = 边际预算份额
    Y = 可支配收入
    Frisch参数将mu_i和theta_i联系起来
"""

import numpy as np
from typing import Dict, Any

from cge_core.base_module import CGEModule, SAMData


class HouseholdModule(CGEModule):
    """代表性居民户LES需求模块。"""

    name = "household"
    description = "代表性居民户LES需求系统（线性支出系统）"
    version = "1.0.0"
    is_core = True
    dependencies = ["production"]

    def declare_parameters(self, model) -> None:
        cal = model.calibrator.params
        model.beta_cons = list(cal["beta_cons"])
        model.theta_les = list(cal.get("theta_les", np.ones(model.num_sectors) * 0.3))
        model.frisch = cal.get("frisch", -1.5)
        model.savings_rate = cal.get("savings_rate", 0.35)
        model.base_hh_cons_total = float(cal.get("total_hh_cons", 1.0))

    def declare_variables(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        # 居民消费
        model.CH = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 居民总收入
        model.YH = pyo.Var(initialize=1.0, bounds=(1e-6, None))

        # 居民可支配收入
        model.YD = pyo.Var(initialize=1.0, bounds=(1e-6, None))

        # 总储蓄
        model.HSAV = pyo.Var(initialize=1.0, bounds=(0, None))

        # 总消费支出
        model.TOTCH = pyo.Var(initialize=1.0, bounds=(1e-6, None))

    def declare_equations(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        def income_rule(m):
            """居民总收入 = 劳动收入 + 资本收入 + 政府转移支付"""
            labor_income = sum(m.WL * m.LD[j] for j in range(n))
            capital_income = sum(m.WK * m.KD[j] for j in range(n))
            return m.YH == labor_income + capital_income

        model.eq_hh_income = pyo.Constraint(rule=income_rule)

        def disposable_income_rule(m):
            """可支配收入 = 总收入 - 所得税"""
            tax_rate = m.config.get("households", {}).get("income_tax_rate", 0.10)
            scenario_override = m.scenario_params.get("household", {}).get("income_tax_rate", None)
            if scenario_override is not None:
                tax_rate = scenario_override
            return m.YD == m.YH * (1 - tax_rate)

        model.eq_disposable = pyo.Constraint(rule=disposable_income_rule)

        def savings_rule(m):
            """储蓄 = 可支配收入 × 储蓄率"""
            return m.HSAV == m.YD * m.savings_rate

        model.eq_savings = pyo.Constraint(rule=savings_rule)

        def total_consumption_rule(m):
            """总消费支出 = 可支配收入 - 储蓄"""
            return m.TOTCH == m.YD - m.HSAV

        model.eq_total_cons = pyo.Constraint(rule=total_consumption_rule)

        def les_demand_rule(m, i):
            """LES需求:
            C_i = theta_i + beta_i * (TOTCH/P_i - sum_j theta_j)
            简化形式（价格归一化时）:
            C_i = theta_i * C0_i + beta_i * (1 - sum theta) * TOTCH / P_i
            """
            # 使用LES的弹性形式
            # 当价格=1时简化为: C_i ≈ beta_i * TOTCH
            # 加入Frisch参数的调整
            eps = 1e-10
            theta = m.theta_les[i]
            beta = m.beta_cons[i]

            # LES: C_i = theta_i + beta_i/(P_i) * (TOTCH - sum_j theta_j * P_j)
            # 在基期价格=1时: C_i = theta_i + beta_i * (TOTCH - sum theta)
            subsistence_expenditure = sum(m.theta_les[j] for j in range(n))
            residual = m.TOTCH - subsistence_expenditure * m.PX[i]

            return m.CH[i] == theta * m.base_hh_cons_total / m.num_sectors + beta * (residual + eps) / (m.PX[i] + eps)

        model.eq_les_demand = pyo.Constraint(model.SECTORS, rule=les_demand_rule)

    def calibrate(self, model, sam: SAMData) -> None:
        pass

    def scenario_params(self) -> Dict[str, Dict[str, Any]]:
        return {
            "income_tax_rate": {
                "default": 0.10, "min": 0.0, "max": 0.45, "step": 0.01,
                "label": "个人所得税率", "type": "slider",
            },
            "savings_rate": {
                "default": 0.35, "min": 0.10, "max": 0.60, "step": 0.01,
                "label": "居民储蓄率", "type": "slider",
            },
        }
