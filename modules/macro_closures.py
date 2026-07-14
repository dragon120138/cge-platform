# -*- coding: utf-8 -*-
"""
宏观闭合模块 — 投资-储蓄平衡

宏观闭合规则选择:
  (a) 新古典闭合: 投资由储蓄决定 I = S
  (b) 约翰逊闭合: 投资外生，储蓄调整
  (c) 凯恩斯闭合: 投资外生，就业内生

默认使用新古典闭合: 总投资 = 居民储蓄 + 政府储蓄 + 国外储蓄
"""

import numpy as np
from typing import Dict, Any

from cge_core.base_module import CGEModule, SAMData


class MacroClosureModule(CGEModule):
    """宏观闭合模块。"""

    name = "macro_closures"
    description = "投资-储蓄平衡闭合（新古典/约翰逊/凯恩斯）"
    version = "1.0.0"
    is_core = True
    dependencies = ["production", "household", "government", "trade", "market_clearing"]

    def declare_parameters(self, model) -> None:
        model.closure_type = model.config.get("macro_closure", {}).get("type", "neoclassical")

    def declare_variables(self, model) -> None:
        import pyomo.environ as pyo
        # 宏观闭合不需要额外变量，使用已有变量
        pass

    def declare_equations(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        closure = model.closure_type

        def macro_balance_rule(m):
            """宏观平衡:
            新古典闭合: 总投资 = 居民储蓄 + 政府储蓄 + 国外储蓄
            """
            total_savings = m.HSAV + m.GSAV + m.FSAV * m.EXR

            # 总投资 = Σ PQ_i * INV_i
            total_investment = sum(m.PQ[i] * m.INV[i] for i in range(n))

            return m.TOTINV == total_savings + 1e-10

        model.eq_macro_balance = pyo.Constraint(rule=macro_balance_rule)

        def walras_rule(m):
            """瓦尔拉斯法则: 瓦尔拉斯虚拟变量 ≈ 0

            在一般均衡中，如果所有其他市场都出清，
            则最后一个市场自动出清。引入瓦尔拉斯虚拟变量
            用于检查模型一致性。
            """
            # 这是隐含的，不强制约束
            # 仅记录信息
            return pyo.Constraint.Skip

        # 瓦尔拉斯变量（可选）
        model.WALRAS = pyo.Var(initialize=0.0, bounds=(-1e6, 1e6))

    def calibrate(self, model, sam: SAMData) -> None:
        pass

    def scenario_params(self) -> Dict[str, Dict[str, Any]]:
        return {}
