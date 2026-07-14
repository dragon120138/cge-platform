# -*- coding: utf-8 -*-
"""
校准器 — 从SAM提取CGE模型参数
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .base_module import SAMData
from .sectors import NUM_SECTORS, SECTOR_CODES


class Calibrator:
    """从平衡的SAM提取所有CGE模型校准参数。

    参数命名遵循CGE标准记号：
      a_ij   : 中间投入系数
      alpha  : 增加值中劳动份额
      beta   : 居民消费份额
      gamma  : 政府消费份额
      iota   : 投资份额
      io     : CES份额参数
    """

    def __init__(self, sam: SAMData):
        self.sam = sam
        self.params: Dict[str, Any] = {}
        self._calibrate()

    def _calibrate(self):
        sam = self.sam
        n = sam.num_sectors

        # ---- 总产出 ----
        total_output = sam.total_output.copy()

        # ---- 中间投入系数 a_ij ----
        # a_ij = X_ij / X_j  (部门j对部门i产品的单位消耗)
        a_inter = sam.intermediate / (total_output[np.newaxis, :] + 1e-10)
        self.params["a_inter"] = a_inter

        # ---- 增加值分解 ----
        # 增加值 = 劳动报酬 + 资本回报 + 生产税净额
        va = sam.labor_comp + sam.capital_income + sam.production_tax
        va = np.maximum(va, 1e-6)

        # 劳动份额和资本份额 (CES)
        alpha_L = sam.labor_comp / (sam.labor_comp + sam.capital_income + 1e-10)
        alpha_K = 1.0 - alpha_L
        self.params["alpha_L"] = alpha_L
        self.params["alpha_K"] = alpha_K

        # 增加值生产税份额
        va_tax_share = sam.production_tax / (va + 1e-10)
        self.params["va_tax_share"] = va_tax_share

        # ---- CES份额和规模参数 ----
        # VA_i = A_i * [alpha_L * L^rho + alpha_K * K^rho]^(1/rho)
        # 校准：A_i = VA_i / [alpha_L * L^rho + alpha_K * K^rho]^(1/rho)
        sigma_va = sam.ces_va_sigma
        rho_va = (sigma_va - 1.0) / sigma_va

        ces_term = (alpha_L * (sam.labor_comp ** rho_va) +
                    alpha_K * (sam.capital_income ** rho_va))
        A_va = va / (np.power(np.maximum(ces_term, 1e-10), 1.0 / rho_va) + 1e-10)
        self.params["A_va"] = A_va
        self.params["sigma_va"] = sigma_va
        self.params["rho_va"] = rho_va

        # ---- 顶层Leontief系数 ----
        # 总产出 = min(int/coef, VA/va_coef)
        int_coef = sam.intermediate.sum(axis=0) / (total_output + 1e-10)
        va_coef = va / (total_output + 1e-10)
        self.params["int_coef"] = int_coef
        self.params["va_coef"] = va_coef

        # ---- 居民消费份额 (Cobb-Douglas / LES) ----
        total_hh_cons = sam.household_cons.sum()
        beta_cons = sam.household_cons / (total_hh_cons + 1e-10)
        self.params["beta_cons"] = beta_cons
        self.params["total_hh_cons"] = total_hh_cons

        # LES subsistence parameters (theta = subsistence expenditure / total expenditure)
        # 假设生存性支出约为消费的30-50%
        theta_les = np.clip(0.15 + 0.3 * np.random.default_rng(0).random(n), 0.1, 0.5)
        self.params["theta_les"] = theta_les
        self.params["frisch"] = sam.frisch

        # ---- 政府消费份额 ----
        total_gov_cons = sam.government_cons.sum()
        gamma_cons = sam.government_cons / (total_gov_cons + 1e-10)
        self.params["gamma_cons"] = gamma_cons
        self.params["total_gov_cons"] = total_gov_cons

        # ---- 投资份额 ----
        total_inv = sam.investment.sum()
        iota_cons = sam.investment / (total_inv + 1e-10)
        self.params["iota_cons"] = iota_cons
        self.params["total_inv"] = total_inv

        # ---- 贸易参数 ----
        # Armington: 国内产品与进口品的组合
        domestic_supply = total_output - sam.exports
        domestic_supply = np.maximum(domestic_supply, 1e-6)

        # 进口份额
        total_supply = domestic_supply + sam.imports
        import_share = sam.imports / (total_supply + 1e-10)
        domestic_share = domestic_supply / (total_supply + 1e-10)
        self.params["import_share"] = import_share
        self.params["domestic_share"] = domestic_share

        # CET: 国内与出口的分配
        export_share = sam.exports / (total_output + 1e-10)
        self.params["export_share"] = export_share

        # Armington/CET弹性
        self.params["sigma_armington"] = sam.armington_sigma
        self.params["sigma_cet"] = sam.cet_sigma

        # ---- 税率 ----
        self.params["tax_rates"] = sam.tax_rates.copy() if isinstance(sam.tax_rates, dict) else {}

        # ---- 宏观总量 ----
        self.params["gdp"] = sam.gdp
        self.params["total_output"] = total_output

        # ---- 劳动和资本总量 ----
        self.params["total_labor"] = sam.labor_comp.sum()
        self.params["total_capital"] = sam.capital_income.sum()

        # ---- 折旧率 ----
        # delta_i = depreciation_i / capital_stock_i
        # 资本存量 = 资本回报 / 收益率 (假设收益率 = 0.10)
        assumed_return = 0.10
        capital_stock = sam.capital_income / assumed_return
        delta_depreciation = sam.depreciation / (capital_stock + 1e-10)
        self.params["depreciation_rate"] = np.clip(delta_depreciation, 0.02, 0.15)
        self.params["capital_stock"] = capital_stock

        # ---- 储蓄率 ----
        total_income = sam.labor_comp.sum() + sam.capital_income.sum()
        total_savings = total_inv  # 简化：储蓄=投资
        savings_rate = total_savings / (total_income + 1e-10)
        self.params["savings_rate"] = float(np.clip(savings_rate, 0.15, 0.55))

    def get(self, key: str, default=None):
        return self.params.get(key, default)

    def to_json(self, path: str) -> None:
        """保存校准参数为JSON。"""
        out = {}
        for k, v in self.params.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, dict):
                out[k] = {
                    kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                    for kk, vv in v.items()
                }
            else:
                out[k] = v
        Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[校准器] 参数已保存至 {path}")

    def summary(self) -> str:
        lines = [
            "校准参数摘要:",
            f"  GDP: {self.params['gdp']:,.0f} 亿元",
            f"  总产出: {self.params['total_output'].sum():,.0f} 亿元",
            f"  居民消费: {self.params['total_hh_cons']:,.0f} 亿元",
            f"  政府消费: {self.params['total_gov_cons']:,.0f} 亿元",
            f"  总投资: {self.params['total_inv']:,.0f} 亿元",
            f"  储蓄率: {self.params['savings_rate']:.1%}",
            f"  Frisch参数: {self.params['frisch']}",
            f"  劳动报酬/GDP: {self.params['total_labor']/self.params['gdp']:.1%}",
            f"  资本回报/GDP: {self.params['total_capital']/self.params['gdp']:.1%}",
        ]
        return "\n".join(lines)
