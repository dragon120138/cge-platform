# -*- coding: utf-8 -*-
"""
市场出清模块 — 商品、要素、外汇市场

42商品市场: 供给(QA) = 需求(中间投入+居民消费+政府消费+投资)
劳动市场: 供给(LS) = 需求(Σ LD_j)
资本市场: 供给(KS) = 需求(Σ KD_j)
"""

import numpy as np
from typing import Dict, Any

from cge_core.base_module import CGEModule, SAMData


class MarketClearingModule(CGEModule):
    """市场出清条件模块。"""

    name = "market_clearing"
    description = "42商品市场 + 劳动/资本市场出清条件"
    version = "1.0.0"
    is_core = True
    dependencies = ["production", "household", "government", "trade"]

    def declare_parameters(self, model) -> None:
        cal = model.calibrator.params
        n = model.num_sectors

        model.iota_cons = list(cal["iota_cons"])
        model.total_labor_supply = float(cal.get("total_labor", 1.0))
        model.total_capital_supply = float(cal.get("total_capital", 1.0))

    def declare_variables(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        # 投资需求
        model.INV = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 劳动供给
        model.LS = pyo.Var(initialize=1.0, bounds=(1e-6, None))

        # 资本供给
        model.KS = pyo.Var(initialize=1.0, bounds=(1e-6, None))

        # 总投资
        model.TOTINV = pyo.Var(initialize=1.0, bounds=(1e-6, None))

    def declare_equations(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        def goods_market_rule(m, i):
            """商品市场出清:
            QA_i = Σ_j INTJ_ij + CH_i + CG_i + INV_i
            """
            intermediate_demand = sum(m.INTJ[i, j] for j in range(n))
            return m.QA[i] == intermediate_demand + m.CH[i] + m.CG[i] + m.INV[i]

        model.eq_goods_market = pyo.Constraint(model.SECTORS, rule=goods_market_rule)

        def investment_rule(m, i):
            """投资需求 = 份额 × 总投资"""
            return m.INV[i] == m.iota_cons[i] * m.TOTINV / (m.PQ[i] + 1e-10)

        model.eq_investment = pyo.Constraint(model.SECTORS, rule=investment_rule)

        def labor_market_rule(m):
            """劳动市场出清: LS = Σ LD_j"""
            return m.LS == sum(m.LD[j] for j in range(n))

        model.eq_labor_market = pyo.Constraint(rule=labor_market_rule)

        def capital_market_rule(m):
            """资本市场出清: KS = Σ KD_j"""
            return m.KS == sum(m.KD[j] for j in range(n))

        model.eq_capital_market = pyo.Constraint(rule=capital_market_rule)

    def calibrate(self, model, sam: SAMData) -> None:
        pass

    def scenario_params(self) -> Dict[str, Dict[str, Any]]:
        return {}
