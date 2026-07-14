# -*- coding: utf-8 -*-
"""
贸易模块 — Armington + CET，小国开放经济

Armington假设: 国内产品和进口品是不完全替代的
  Q_i = A_arm * [delta_M * M^(-rho_A) + delta_D * D^(-rho_A)]^(-1/rho_A)

CET转换: 国内产出在出口和内销之间分配
  XS_i = B_cet * [phi_E * E^(rho_C) + phi_D * D^(rho_C)]^(1/rho_C)
"""

import numpy as np
from typing import Dict, Any

from cge_core.base_module import CGEModule, SAMData


class TradeModule(CGEModule):
    """小国开放经济贸易模块（Armington + CET）。"""

    name = "trade"
    description = "Armington进口替代 + CET出口转换，小国开放经济"
    version = "1.0.0"
    is_core = True
    dependencies = ["production"]

    def declare_parameters(self, model) -> None:
        cal = model.calibrator.params
        n = model.num_sectors

        model.sigma_armington = list(cal["sigma_armington"])
        model.sigma_cet = list(cal["sigma_cet"])
        model.import_share = list(cal["import_share"])
        model.domestic_share = list(cal["domestic_share"])
        model.export_share = list(cal["export_share"])

        # 世界价格（外生，小国假设）
        model.pwm = list(np.ones(n))  # 进口品世界价格
        model.pwe = list(np.ones(n))  # 出口品世界价格

        # 基期值（归一化为1）
        model.base_imports = list(model.sam.imports / model.sam.imports.sum())
        model.base_exports = list(model.sam.exports / model.sam.exports.sum())
        model.base_domestic = list(
            (model.sam.total_output - model.sam.exports) /
            (model.sam.total_output.sum() + 1e-10)
        )

    def declare_variables(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        # Armington复合品供给
        model.QA = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 进口
        model.M = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 国内供给
        model.D = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 出口
        model.E = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # Armington复合品价格
        model.PQ = pyo.Var(model.SECTORS, initialize=1.0, bounds=(1e-6, None))

        # 汇率
        model.EXR = pyo.Var(initialize=1.0, bounds=(1e-6, None))

        # 国外储蓄
        model.FSAV = pyo.Var(initialize=0.0, bounds=(-1e10, 1e10))

    def declare_equations(self, model) -> None:
        import pyomo.environ as pyo
        n = model.num_sectors

        def armington_composite_rule(m, i):
            """Armington复合:
            QA_i = [delta_M^(sigma) * M^((sigma-1)/sigma) +
                     delta_D^(sigma) * D^((sigma-1)/sigma)]^(sigma/(sigma-1))
            """
            sigma = m.sigma_armington[i]
            delta_M = m.import_share[i]
            delta_D = m.domestic_share[i]
            eps = 1e-10

            rho = (sigma - 1) / sigma
            inner = (delta_M + eps) ** sigma * m.M[i] ** rho + \
                    (delta_D + eps) ** sigma * m.D[i] ** rho
            return m.QA[i] == (inner + eps) ** (sigma / (sigma - 1))

        model.eq_armington = pyo.Constraint(model.SECTORS, rule=armington_composite_rule)

        def import_demand_rule(m, i):
            """进口需求（成本最小化FOC）:
            M/D = (delta_M/delta_D * PD/PM)^sigma
            """
            sigma = m.sigma_armington[i]
            delta_M = m.import_share[i]
            delta_D = m.domestic_share[i]
            eps = 1e-10

            tariff = m.config.get("trade", {}).get("import_tariff", 0.05)
            PM = m.pwm[i] * (1 + tariff) * m.EXR
            PD = m.PX[i]

            ratio = m.M[i] / (m.D[i] + eps)
            return ratio == ((delta_M + eps) / (delta_D + eps) * PD / (PM + eps)) ** sigma

        model.eq_import_demand = pyo.Constraint(model.SECTORS, rule=import_demand_rule)

        def cet_transformation_rule(m, i):
            """CET转换:
            XS_i = [phi_E^(sigma_cet+1) * E^(sigma_cet) +
                     phi_D^(sigma_cet+1) * D^(sigma_cet)]^(1/sigma_cet)
            """
            sigma = m.sigma_cet[i]
            phi_E = m.export_share[i]
            phi_D = 1.0 - m.export_share[i]
            eps = 1e-10

            rho = sigma / (sigma - 1)
            inner = (phi_E + eps) ** (1 - rho) * m.E[i] ** rho + \
                    (phi_D + eps) ** (1 - rho) * m.D[i] ** rho
            return m.XS[i] == (inner + eps) ** (1.0 / rho)

        model.eq_cet = pyo.Constraint(model.SECTORS, rule=cet_transformation_rule)

        def export_supply_rule(m, i):
            """出口供给（利润最大化FOC）:
            E/D = (phi_E/phi_D * PE/PD)^sigma_cet
            """
            sigma = m.sigma_cet[i]
            phi_E = m.export_share[i]
            phi_D = 1.0 - m.export_share[i]
            eps = 1e-10

            PE = m.pwe[i] * m.EXR
            PD = m.PX[i]

            ratio = m.E[i] / (m.D[i] + eps)
            return ratio == ((phi_E + eps) / (phi_D + eps) * PE / (PD + eps)) ** sigma

        model.eq_export_supply = pyo.Constraint(model.SECTORS, rule=export_supply_rule)

        def composite_price_rule(m, i):
            """Armington复合品价格"""
            tariff = m.config.get("trade", {}).get("import_tariff", 0.05)
            PM = m.pwm[i] * (1 + tariff) * m.EXR
            PD = m.PX[i]

            return m.PQ[i] * m.QA[i] == PD * m.D[i] + PM * m.M[i]

        model.eq_pq = pyo.Constraint(model.SECTORS, rule=composite_price_rule)

        def current_account_rule(m):
            """国际收支平衡:
            sum_i pwe_i * E_i + FSAV = sum_i pwm_i * M_i
            """
            eps = 1e-10
            total_exports = sum(m.pwe[i] * m.E[i] for i in range(n))
            total_imports = sum(m.pwm[i] * m.M[i] for i in range(n))
            return total_exports + m.FSAV == total_imports

        model.eq_current_account = pyo.Constraint(rule=current_account_rule)

    def calibrate(self, model, sam: SAMData) -> None:
        pass

    def scenario_params(self) -> Dict[str, Dict[str, Any]]:
        return {
            "import_tariff": {
                "default": 0.05, "min": 0.0, "max": 0.30, "step": 0.01,
                "label": "进口关税率", "type": "slider",
            },
        }
