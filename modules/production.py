# -*- coding: utf-8 -*-
"""
生产模块 — 嵌套CES生产函数，42部门

结构:
  顶层: Leontief(中间投入束, 增加值束)
  增加值: CES(劳动L, 资本K)
  中间投入: Leontief(42种中间产品)

零利润条件: PX_i * XS_i = PVA_i * VA_i + PINT_i * INT_i
"""

import numpy as np
from typing import Dict, Any

from cge_core.base_module import CGEModule, SAMData


class ProductionModule(CGEModule):
    """嵌套CES生产模块。"""

    name = "production"
    description = "42部门嵌套CES生产函数（Leontief顶层 + CES增加值）"
    version = "1.0.0"
    is_core = True

    def declare_parameters(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors
        cal = model.calibrator.params

        # 中间投入系数 a_ij (存储在model上供其他模块访问)
        model.a_inter = list(cal["a_inter"])  # n×n矩阵
        model.int_coef = list(cal["int_coef"])
        model.va_coef = list(cal["va_coef"])
        model.alpha_L = list(cal["alpha_L"])
        model.alpha_K = list(cal["alpha_K"])
        model.A_va = list(cal["A_va"])
        model.sigma_va = list(cal["sigma_va"])
        model.rho_va = list(cal["rho_va"])
        model.base_output = list(cal["total_output"])

    def declare_variables(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors
        cal = model.calibrator.params

        # 部门总产出
        model.XS = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 增加值
        model.VA = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 中间投入总量
        model.INT = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 劳动需求
        model.LD = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 资本需求
        model.KD = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 中间投入需求
        model.INTJ = pyo.Var(model.SECTORS, model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 增加值价格
        model.PVA = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 中间投入价格指数
        model.PINT = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 部门产出价格
        model.PX = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 要素价格
        model.WL = pyo.Var(initialize=1.0, bounds=(1e-6, None))  # 工资
        model.WK = pyo.Var(initialize=1.0, bounds=(1e-6, None))  # 资本租金

        # 初始化变量为校准值
        for i in range(n):
            model.XS[i] = 1.0
            model.VA[i] = 1.0
            model.INT[i] = 1.0
            model.LD[i] = 1.0
            model.KD[i] = 1.0
            model.PVA[i] = 1.0
            model.PINT[i] = 1.0
            model.PX[i] = 1.0

    def declare_equations(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        def leontief_output_rule(m, j):
            """顶层Leontief: 总产出 = 增加值/增加值系数 = 中间投入/中间系数"""
            return m.XS[j] * m.va_coef[j] == m.VA[j]

        model.eq_leontief_va = pyo.Constraint(model.SECTORS, rule=leontief_output_rule)

        def leontief_int_rule(m, j):
            """中间投入 = 总产出 × 中间投入系数"""
            return m.INT[j] == sum(m.a_inter[i][j] * m.XS[j] for i in range(n))

        model.eq_leontief_int = pyo.Constraint(model.SECTORS, rule=leontief_int_rule)

        def intermediate_demand_rule(m, i, j):
            """部门j对部门i产品的中间需求"""
            return m.INTJ[i, j] == m.a_inter[i][j] * m.XS[j]

        model.eq_int_demand = pyo.Constraint(model.SECTORS, model.SECTORS,
                                              rule=intermediate_demand_rule)

        def ces_va_production_rule(m, j):
            """CES增加值生产函数:
            VA_j = A_j * [alpha_L * L^rho + alpha_K * K^rho]^(1/rho)
            """
            rho = m.rho_va[j]
            alpha_L = m.alpha_L[j]
            alpha_K = m.alpha_K[j]
            A = m.A_va[j]

            inner = alpha_L * (m.LD[j] ** rho) + alpha_K * (m.KD[j] ** rho)
            return m.VA[j] == A * (inner + 1e-10) ** (1.0 / rho)

        model.eq_ces_va = pyo.Constraint(model.SECTORS, rule=ces_va_production_rule)

        def ces_labor_demand_rule(m, j):
            """劳动需求的一阶条件:
            LD = VA/A * alpha_L * (PVA / (WL * A))^(sigma-1)
            或直接用成本最小化条件
            """
            sigma = m.sigma_va[j]
            rho = m.rho_va[j]
            alpha_L = m.alpha_L[j]

            # 成本最小化: LD/LD0 = (WL0/WL * PVA/PVA0)^sigma * alpha_L^...
            # 简化为: WL * LD = alpha_L * PVA * VA (Cobb-Douglas极限)
            # 通用CES条件:
            # alpha_L * (LD^(rho-1)) / [alpha_L*LD^rho + alpha_K*KD^rho] * VA = WL/PVA * VA

            eps = 1e-10
            # FOC: WL/PVA = alpha_L * LD^(rho-1) / inner
            inner = alpha_L * (m.LD[j] ** rho) + m.alpha_K[j] * (m.KD[j] ** rho)
            lhs = m.WL
            rhs = m.PVA[j] * m.A_va[j] * alpha_L * (m.LD[j] + eps) ** (rho - 1) * \
                  (inner + eps) ** (1.0/rho - 1)
            return lhs == rhs + eps

        model.eq_labor_demand = pyo.Constraint(model.SECTORS, rule=ces_labor_demand_rule)

        def ces_capital_demand_rule(m, j):
            """资本需求的一阶条件"""
            rho = m.rho_va[j]
            alpha_L = m.alpha_L[j]
            alpha_K = m.alpha_K[j]

            eps = 1e-10
            inner = alpha_L * (m.LD[j] ** rho) + alpha_K * (m.KD[j] ** rho)
            lhs = m.WK
            rhs = m.PVA[j] * m.A_va[j] * alpha_K * (m.KD[j] + eps) ** (rho - 1) * \
                  (inner + eps) ** (1.0/rho - 1)
            return lhs == rhs + eps

        model.eq_capital_demand = pyo.Constraint(model.SECTORS, rule=ces_capital_demand_rule)

        def pva_price_rule(m, j):
            """增加值价格 = 单位成本函数"""
            # PVA = (1/A) * [alpha_L^sigma * WL^(1-sigma) + alpha_K^sigma * WK^(1-sigma)]^(1/(1-sigma))
            sigma = m.sigma_va[j]
            alpha_L = m.alpha_L[j]
            alpha_K = m.alpha_K[j]
            A = m.A_va[j]

            eps = 1e-10
            term_L = (alpha_L + eps) ** sigma * (m.WL + eps) ** (1 - sigma)
            term_K = (alpha_K + eps) ** sigma * (m.WK + eps) ** (1 - sigma)
            return m.PVA[j] == (1.0/A) * (term_L + term_K + eps) ** (1.0/(1-sigma))

        model.eq_pva = pyo.Constraint(model.SECTORS, rule=pva_price_rule)

        def zero_profit_rule(m, j):
            """零利润条件: PX * XS = PVA * VA + PINT * INT"""
            return m.PX[j] * m.XS[j] == m.PVA[j] * m.VA[j] + m.PINT[j] * m.INT[j]

        model.eq_zero_profit = pyo.Constraint(model.SECTORS, rule=zero_profit_rule)

    def calibrate(self, model, sam: SAMData) -> None:
        """校准已在Calibrator中完成，此处不需额外操作。"""
        pass

    def scenario_params(self) -> Dict[str, Dict[str, Any]]:
        return {
            "sigma_va_override": {
                "default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1,
                "label": "增加值CES替代弹性", "type": "slider",
            },
        }
