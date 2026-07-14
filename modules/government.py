# -*- coding: utf-8 -*-
"""
政府模块 — 多税收工具 + 财政闭合

税收工具:
  - 消费税（增值税近似）
  - 企业所得税
  - 生产税
  - 进口关税

财政闭合选项:
  (a) 平衡预算（支出=收入）
  (b) 固定赤字
  (c) 内生税率
"""

import numpy as np
from typing import Dict, Any

from cge_core.base_module import CGEModule, SAMData


class GovernmentModule(CGEModule):
    """政府税收与支出模块。"""

    name = "government"
    description = "多税收工具（消费税/所得税/生产税/关税）+ 财政闭合"
    version = "1.0.0"
    is_core = True
    dependencies = ["production", "household"]

    def declare_parameters(self, model) -> None:
        cal = model.calibrator.params
        n = model.num_sectors

        model.gamma_cons = list(cal["gamma_cons"])
        model.production_tax_rate = list(cal.get("tax_rates", {}).get(
            "production_tax_rate", np.zeros(n)))
        model.consumption_tax_base = 0.13
        model.corporate_tax_base = 0.25
        model.import_tariff_base = 0.05
        model.gov_purchase_base = float(cal.get("total_gov_cons", 1.0))

    def declare_variables(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        # 政府消费
        model.CG = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 政府总收入
        model.GREV = pyo.Var(initialize=1.0, bounds=(1e-6, None))

        # 政府总支出
        model.GEXP = pyo.Var(initialize=1.0, bounds=(1e-6, None))

        # 政府储蓄（赤字）
        model.GSAV = pyo.Var(initialize=0.0, bounds=(-1e10, None))

        # 有效税率（场景可调）
        model.tau_c = pyo.Var(initialize=0.13, bounds=(0, 1.0))     # 消费税率
        model.tau_ci = pyo.Var(initialize=0.25, bounds=(0, 1.0))    # 企业所得税率

    def declare_equations(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        # 场景参数覆盖
        scenario_gov = model.scenario_params.get("government", {})

        def consumption_tax_rate_rule(m):
            """设定有效消费税率（含场景冲击）"""
            base = 0.13
            change = scenario_gov.get("consumption_tax_rate_change", 0.0)
            return m.tau_c == max(base + change, 0.0)

        model.eq_tau_c = pyo.Constraint(rule=consumption_tax_rate_rule)

        def corporate_tax_rate_rule(m):
            """设定有效企业所得税率"""
            base = 0.25
            change = scenario_gov.get("corporate_income_tax_rate_change", 0.0)
            return m.tau_ci == max(base + change, 0.0)

        model.eq_tau_ci = pyo.Constraint(rule=corporate_tax_rate_rule)

        def gov_consumption_rule(m, i):
            """政府消费 = 份额 × 总政府支出"""
            # 场景冲击：政府购买增加
            multiplier = scenario_gov.get("gov_purchase_multiplier", 1.0)
            increase = scenario_gov.get("gov_purchase_increase_yi", 0.0)
            total_gov = m.gov_purchase_base * multiplier + increase
            return m.CG[i] == m.gamma_cons[i] * total_gov / (m.PX[i] + 1e-10)

        model.eq_gov_cons = pyo.Constraint(model.SECTORS, rule=gov_consumption_rule)

        def gov_revenue_rule(m):
            """政府收入 = 消费税 + 所得税 + 生产税 + 关税"""
            # 消费税
            cons_tax = sum(m.tau_c * m.PX[i] * m.CH[i] for i in range(n))
            # 企业所得税
            corp_income = sum(
                m.tau_ci * m.WK * m.KD[i] for i in range(n)
            )
            # 生产税（按部门）
            prod_tax = sum(
                m.production_tax_rate[i] * m.PX[i] * m.XS[i] for i in range(n)
            )

            return m.GREV == cons_tax + corp_income + prod_tax

        model.eq_grev = pyo.Constraint(rule=gov_revenue_rule)

        def gov_expenditure_rule(m):
            """政府支出 = 政府消费 + 转移支付"""
            total_cons = sum(m.PX[i] * m.CG[i] for i in range(n))
            return m.GEXP == total_cons

        model.eq_gexp = pyo.Constraint(rule=gov_expenditure_rule)

        def gov_savings_rule(m):
            """政府储蓄 = 收入 - 支出"""
            return m.GSAV == m.GREV - m.GEXP

        model.eq_gsav = pyo.Constraint(rule=gov_savings_rule)

    def calibrate(self, model, sam: SAMData) -> None:
        pass

    def scenario_params(self) -> Dict[str, Dict[str, Any]]:
        return {
            "consumption_tax_rate_change": {
                "default": 0.0, "min": -0.05, "max": 0.05, "step": 0.005,
                "label": "消费税率变化（百分点）", "type": "slider",
            },
            "corporate_income_tax_rate_change": {
                "default": 0.0, "min": -0.10, "max": 0.10, "step": 0.01,
                "label": "企业所得税率变化（百分点）", "type": "slider",
            },
            "gov_purchase_multiplier": {
                "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                "label": "政府购买乘数", "type": "slider",
            },
            "gov_purchase_increase_yi": {
                "default": 0.0, "min": 0.0, "max": 50000.0, "step": 1000.0,
                "label": "政府购买增加额（亿元）", "type": "slider",
            },
        }
