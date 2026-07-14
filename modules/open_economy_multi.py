# -*- coding: utf-8 -*-
"""
多区域开放经济模块 [扩展] — 中国 + 世界其他地区(ROW)

将小国开放经济扩展为两区域模型:
  - 中国(CHN)
  - 世界其他地区(ROW)
  - 双向贸易流内生于双方需求
  - 内生贸易条件(terms of trade)
"""

import numpy as np
from typing import Dict, Any

from cge_core.base_module import CGEModule, SAMData


class OpenEconomyMultiModule(CGEModule):
    """多区域开放经济模块。"""

    name = "open_economy_multi"
    description = "两区域（中国+ROW）贸易模型，内生贸易条件"
    version = "1.0.0"
    dependencies = ["production", "trade"]

    def declare_parameters(self, model) -> None:
        n = model.num_sectors

        # ROW的Armington弹性（通常小于小国假设）
        model.sigma_armington_row = list(np.full(n, 2.0))

        # ROW对中国出口品的需求弹性
        model.row_demand_elasticity = list(np.full(n, -1.5))

        # 中国占全球产出份额（影响贸易条件）
        model.chn_global_share = list(np.full(n, 0.25))

        # 贸易成本（冰山型）
        model.iceberg_cost = list(np.full(n, 1.1))  # 10%贸易成本

    def declare_variables(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        # 内生世界价格（不再是外生）
        model.PWM = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 内生出口品世界价格
        model.PWE = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # ROW对中国产品的需求
        model.ROW_DEMAND = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 贸易条件
        model.TOT = pyo.Var(initialize=1.0, bounds=(1e-6, None))

    def declare_equations(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        def world_price_rule(m, i):
            """世界价格 = 中国价格 × 中国份额 + ROW基准价格 × (1-中国份额)
            中国份额越大，对世界价格影响越大
            """
            row_base = 1.0  # ROW基准价格归一化
            chn_share = m.chn_global_share[i]
            return m.PWM[i] == chn_share * m.PX[i] + (1 - chn_share) * row_base

        model.eq_world_price = pyo.Constraint(model.SECTORS, rule=world_price_rule)

        def row_demand_rule(m, i):
            """ROW对中国出口品的需求（恒弹性需求函数）"""
            eps = 1e-10
            elasticity = m.row_demand_elasticity[i]
            # 基期出口=1, 当价格上升时需求下降
            price_ratio = m.PWE[i] * m.EXR / (m.PWM[i] + eps)
            return m.ROW_DEMAND[i] == (price_ratio + eps) ** elasticity

        model.eq_row_demand = pyo.Constraint(model.SECTORS, rule=row_demand_rule)

        def export_eq_row_demand_rule(m, i):
            """出口 = ROW需求（含贸易成本调整）"""
            return m.E[i] == m.ROW_DEMAND[i] * m.iceberg_cost[i]

        model.eq_export_row = pyo.Constraint(model.SECTORS, rule=export_eq_row_demand_rule)

        def terms_of_trade_rule(m):
            """贸易条件 = 出口价格指数 / 进口价格指数"""
            eps = 1e-10
            export_price_idx = sum(m.PWE[i] * m.E[i] for i in range(n)) / \
                               (sum(m.E[i] for i in range(n)) + eps)
            import_price_idx = sum(m.PWM[i] * m.M[i] for i in range(n)) / \
                               (sum(m.M[i] for i in range(n)) + eps)
            return m.TOT == export_price_idx / (import_price_idx + eps)

        model.eq_tot = pyo.Constraint(rule=terms_of_trade_rule)

    def calibrate(self, model, sam: SAMData) -> None:
        pass

    def scenario_params(self) -> Dict[str, Dict[str, Any]]:
        return {
            "iceberg_cost": {
                "default": 1.1, "min": 1.0, "max": 1.5, "step": 0.05,
                "label": "冰山贸易成本", "type": "slider",
            },
        }
