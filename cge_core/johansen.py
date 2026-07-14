# -*- coding: utf-8 -*-
"""
Johansen对数线性化CGE求解器 v2.0 — 纯numpy实现

v2.0新增:
  - 12季度(3年)动态路径
  - 财政盈余/赤字计算
  - 利率政策工具(外生r)
  - TFP冲击(全要素生产率)
  - 定向产业投资支持
  - 三部门信心指数(消费者/企业/投资者)
  - 分行业12季度产出动态路径

原理:
  在基准年(所有价格=1)附近将CGE方程线性化为百分比变化形式,
  得到线性方程组 B·x = b, 其中x是内生变量百分比变化向量,
  b是外生冲击向量。用numpy.linalg.solve一步求解。

变量排列 (共 5N+5 个内生变量, N=部门数):
  [0, N)       po_i   部门产出价格
  [N, 2N)      qo_i   部门产出数量
  [2N, 3N)     qh_i   居民消费
  [3N, 4N)     qg_i   政府消费
  [4N, 5N)     qinv_i 投资需求
  5N           w      工资率
  5N+1         r      资本回报率
  5N+2         yh     居民收入
  5N+3         sav    总储蓄
  5N+4         exr    实际汇率
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .base_module import SAMData


@dataclass
class JohansenResult:
    """求解结果容器。"""
    status: str = "unknown"
    solve_time: float = 0.0
    pct_changes: Dict[str, Any] = field(default_factory=dict)
    levels: Dict[str, Any] = field(default_factory=dict)
    walras_check: float = 0.0
    convergence_log: List[str] = field(default_factory=list)
    shock_params: Dict[str, Any] = field(default_factory=dict)
    # 派生指标
    gdp_change: float = 0.0
    cpi_change: float = 0.0
    employment_change: float = 0.0
    welfare_change: float = 0.0
    investment_change: float = 0.0
    # v2.0 新增派生指标
    fiscal_balance: float = 0.0        # 财政盈余变化(亿元)
    fiscal_balance_pct: float = 0.0    # 财政平衡/GDP变化(百分点)
    confidence_consumer: float = 0.0   # 消费者信心指数变化
    confidence_enterprise: float = 0.0 # 企业信心指数变化
    confidence_investor: float = 0.0   # 投资者信心指数变化

    @property
    def success(self) -> bool:
        return self.status == "optimal"


class JohansenSolver:
    """Johansen对数线性化CGE求解器 v2.0。

    用法:
      solver = JohansenSolver(sam)
      result = solver.solve_shock({
          'consumption_tax_change': 0.01,
          'gov_spending_change': 0.0,
          'interest_rate_change': -0.005,  # 利率降50bp
          'tfp_shock': {20: 0.02},         # S20行业TFP+2%
          'targeted_sector': 20,            # 定向支持行业
          'targeted_investment': 0.10,      # 投资+10%
      })
    """

    def __init__(self, sam: SAMData, closure: str = "keynesian"):
        self.sam = sam
        self.closure = closure
        self.n = sam.num_sectors
        self.nv = 5 * self.n + 5

        # 变量索引
        self._po = slice(0, self.n)
        self._qo = slice(self.n, 2 * self.n)
        self._qh = slice(2 * self.n, 3 * self.n)
        self._qg = slice(3 * self.n, 4 * self.n)
        self._qinv = slice(4 * self.n, 5 * self.n)
        self._w = 5 * self.n
        self._r = 5 * self.n + 1
        self._yh = 5 * self.n + 2
        self._sav = 5 * self.n + 3
        self._exr = 5 * self.n + 4

        self._compute_shares()

    def _compute_shares(self):
        """从SAM提取所有份额参数。"""
        sam = self.sam
        n = self.n
        output = np.maximum(sam.total_output.copy(), 1e-6)

        # ---- 价格方程份额 ----
        total_cost = sam.intermediate.sum(axis=0) + sam.labor_comp + sam.capital_income
        total_cost = np.maximum(total_cost, 1e-6)

        self.io_cost_sh = sam.intermediate / total_cost[np.newaxis, :]
        self.lab_cost_sh = sam.labor_comp / total_cost
        self.cap_cost_sh = sam.capital_income / total_cost

        # ---- 市场出清份额 ----
        self.int_sh = sam.intermediate.sum(axis=1) / output
        self.hh_sh = np.maximum(sam.household_cons, 0) / output
        self.gov_sh = np.maximum(sam.government_cons, 0) / output
        self.inv_sh = np.maximum(sam.investment, 0) / output
        self.exp_sh = np.maximum(sam.exports, 0) / output
        self.imp_sh = np.maximum(sam.imports, 0) / output

        int_row_sum = sam.intermediate.sum(axis=1, keepdims=True)
        int_row_sum = np.maximum(int_row_sum, 1e-6)
        self.m_int_norm = sam.intermediate / int_row_sum

        # ---- 要素市场 ----
        self.lab_alloc_sh = sam.labor_comp / (sam.labor_comp.sum() + 1e-10)
        self.cap_alloc_sh = sam.capital_income / (sam.capital_income.sum() + 1e-10)

        # CES VA参数
        self.alpha_L = sam.labor_comp / (sam.labor_comp + sam.capital_income + 1e-10)
        self.alpha_K = 1.0 - self.alpha_L
        self.sigma_va = np.maximum(sam.ces_va_sigma, 0.1)
        self.sigma_arm = np.maximum(sam.armington_sigma, 0.5)
        self.sigma_cet = np.maximum(sam.cet_sigma, 0.5)

        self.omega_L = np.sum(self.lab_alloc_sh * self.sigma_va * self.alpha_K)
        self.omega_K = np.sum(self.cap_alloc_sh * self.sigma_va * self.alpha_L)

        # ---- 收入份额 ----
        total_income = sam.labor_comp.sum() + sam.capital_income.sum()
        self.inc_lab_sh = sam.labor_comp.sum() / (total_income + 1e-10)
        self.inc_cap_sh = sam.capital_income.sum() / (total_income + 1e-10)

        # ---- 消费份额 ----
        total_hh = sam.household_cons.sum()
        self.beta_cons = np.maximum(sam.household_cons, 1e-6) / (total_hh + 1e-6)

        # ---- 税率 ----
        self.cons_tax_rate = sam.tax_rates.get("consumption_tax", 0.13)
        if isinstance(self.cons_tax_rate, np.ndarray):
            self.cons_tax_rate = float(np.mean(self.cons_tax_rate))
        self.corp_tax_rate = sam.tax_rates.get("corporate_income_tax", 0.25)
        if isinstance(self.corp_tax_rate, np.ndarray):
            self.corp_tax_rate = float(np.mean(self.corp_tax_rate))

        prod_tax_rate = sam.tax_rates.get("production_tax_rate", np.zeros(n))
        if np.isscalar(prod_tax_rate):
            prod_tax_rate = np.full(n, prod_tax_rate)
        self.prod_tax_rate = np.asarray(prod_tax_rate, dtype=float)

        # ---- 贸易 ----
        total_exp = sam.exports.sum()
        total_imp = sam.imports.sum()
        self.exp_share = np.maximum(sam.exports, 0) / (total_exp + 1e-6)
        self.imp_share = np.maximum(sam.imports, 0) / (total_imp + 1e-6)

        self.trade_elast = self.exp_sh * self.sigma_cet + self.imp_sh * self.sigma_arm

        # ---- 基准水平值 ----
        self.baseline_output = output
        self.baseline_gdp = sam.gdp
        self.baseline_labor = sam.labor_comp.sum()
        self.baseline_capital = sam.capital_income.sum()

        # ---- 储蓄率 ----
        total_savings = sam.investment.sum()
        self.savings_rate = total_savings / (total_income + 1e-10)

        # ---- v2.0: 财政基准 ----
        self.baseline_fiscal_revenue = getattr(sam, 'fiscal_revenue', 216000.0)  # 亿元
        self.baseline_fiscal_expenditure = getattr(sam, 'fiscal_expenditure', 287300.0)
        self.baseline_fiscal_deficit = getattr(sam, 'fiscal_deficit', 71300.0)
        self.baseline_lpr = getattr(sam, 'lpr_1y', 3.0)

        # ---- v2.0: 投资利率敏感度 ----
        self.inv_interest_sensitivity = 2.0  # 利率+1% → 投资-2%

    def _build_system(self, shock: Dict[str, Any]):
        """构建线性方程组 B·x = b。"""
        n = self.n
        nv = self.nv
        B = np.zeros((nv, nv))
        b = np.zeros(nv)

        # 提取冲击参数
        gov_shock = shock.get("gov_spending_change", 0.0)
        cons_tax_d = shock.get("consumption_tax_change", 0.0)
        corp_tax_d = shock.get("corporate_tax_change", 0.0)
        prod_tax_d = shock.get("production_tax_change", 0.0)
        L_supply = shock.get("labor_supply_change", 0.0)
        K_supply = shock.get("capital_supply_change", 0.0)
        exr_shock = shock.get("exchange_rate_change", None)

        # v2.0 新增冲击参数
        interest_rate_d = shock.get("interest_rate_change", 0.0)  # 利率变化(百分点)
        tfp_shock = shock.get("tfp_shock", None)  # dict {sector_idx: pct_change}
        targeted_sector = shock.get("targeted_sector", None)
        targeted_investment = shock.get("targeted_investment", 0.0)
        # v2.1: 多行业定向投资 {"targeted_investments": {idx: pct, ...}}
        targeted_investments = shock.get("targeted_investments", None)

        # TFP冲击向量
        tfp_vec = np.zeros(n)
        if tfp_shock:
            if isinstance(tfp_shock, dict):
                for idx, val in tfp_shock.items():
                    if 0 <= idx < n:
                        tfp_vec[idx] = val
            elif isinstance(tfp_shock, (int, float)):
                tfp_vec[:] = tfp_shock  # 全行业TFP冲击

        # 消费税对消费者价格的影响
        cons_tax_effect = cons_tax_d / (1.0 + self.cons_tax_rate)

        # ================================================================
        # E1: 产出价格方程 (N个) — 含TFP
        # p̂o_i = Σ_j io_cost_sh[j,i]·p̂o_j + lab_cost_sh_i·ŵ + cap_cost_sh_i·r̂ - tfp_i + dtax_prod
        # ================================================================
        for i in range(n):
            B[i, self._po.start + i] = 1.0
            for j in range(n):
                B[i, self._po.start + j] -= self.io_cost_sh[j, i]
            B[i, self._w] -= self.lab_cost_sh[i]
            B[i, self._r] -= self.cap_cost_sh[i]
            # 生产税 + TFP (TFP提升→单位成本下降→价格下降)
            b[i] = prod_tax_d - tfp_vec[i]

        # ================================================================
        # E2: 居民消费需求 (N个) — Cobb-Douglas
        # ================================================================
        for i in range(n):
            row = self._qh.start + i
            B[row, self._qh.start + i] = 1.0
            B[row, self._yh] = -1.0
            B[row, self._po.start + i] = 1.0
            b[row] = -cons_tax_effect

        # ================================================================
        # E3: 政府消费 (N个)
        # ================================================================
        for i in range(n):
            row = self._qg.start + i
            B[row, self._qg.start + i] = 1.0
            b[row] = gov_shock

        # ================================================================
        # E4: 投资需求 (N个) — 含利率敏感度和定向支持
        # q̂inv_i = sâv - η·Δr + targeted_inv_i
        # ================================================================
        for i in range(n):
            row = self._qinv.start + i
            B[row, self._qinv.start + i] = 1.0
            B[row, self._sav] = -1.0
            B[row, self._r] += self.inv_interest_sensitivity  # 利率效应
            # 定向投资支持 (兼容旧单行业格式 + 新多行业格式)
            if targeted_investments is not None and i in targeted_investments:
                b[row] = targeted_investments[i]
            elif targeted_sector is not None and i == targeted_sector:
                b[row] = targeted_investment

        # ================================================================
        # E5: 商品市场出清 (N个)
        # ================================================================
        for i in range(n):
            row = self._qo.start + i
            B[row, self._qo.start + i] = 1.0
            for j in range(n):
                B[row, self._qo.start + j] -= self.int_sh[i] * self.m_int_norm[i, j]
            B[row, self._qh.start + i] -= self.hh_sh[i]
            B[row, self._qg.start + i] -= self.gov_sh[i]
            B[row, self._qinv.start + i] -= self.inv_sh[i]
            B[row, self._po.start + i] += self.trade_elast[i]
            B[row, self._exr] -= self.trade_elast[i]

        # ================================================================
        # E6: 劳动市场 (1个) — 含TFP
        # ================================================================
        row = self._w
        wage_shock = shock.get("wage_change", 0.0)

        if self.closure == "keynesian":
            B[row, self._w] = 1.0
            b[row] = wage_shock
        else:
            # 新古典: 充分就业, 工资内生
            # Σ lab_alloc_sh·(q̂o - tfp) - Ω_L·ŵ + Ω_L·r̂ = L_supply
            for i in range(n):
                B[row, self._qo.start + i] += self.lab_alloc_sh[i]
                b[row] -= self.lab_alloc_sh[i] * tfp_vec[i]
            B[row, self._w] -= self.omega_L
            B[row, self._r] += self.omega_L
            b[row] += L_supply

        # ================================================================
        # E7: 资本市场出清 (1个) — 含利率政策和TFP
        # ================================================================
        row = self._r

        if abs(interest_rate_d) > 1e-8:
            # 利率外生: r = interest_rate_change
            B[row, self._r] = 1.0
            b[row] = interest_rate_d
        else:
            # 资本市场内生决定r
            # Σ cap_alloc_sh·(q̂o - tfp) + Ω_K·ŵ - Ω_K·r̂ = K_supply
            for i in range(n):
                B[row, self._qo.start + i] += self.cap_alloc_sh[i]
                b[row] -= self.cap_alloc_sh[i] * tfp_vec[i]
            B[row, self._w] += self.omega_K
            B[row, self._r] -= self.omega_K
            b[row] += K_supply

        # ================================================================
        # E8: 居民收入 (1个)
        # ================================================================
        row = self._yh
        B[row, self._yh] = 1.0
        B[row, self._w] = -self.inc_lab_sh
        B[row, self._r] = -self.inc_cap_sh

        # ================================================================
        # E9: 储蓄-投资平衡 (1个)
        # ================================================================
        row = self._sav
        B[row, self._sav] = 1.0
        B[row, self._yh] = -1.0

        # ================================================================
        # E10: 贸易平衡 / 汇率决定 (1个)
        # ================================================================
        row = self._exr
        if exr_shock is not None:
            B[row, self._exr] = 1.0
            b[row] = exr_shock
        else:
            for i in range(n):
                B[row, self._exr] += self.exp_share[i] * self.sigma_cet[i] + self.imp_share[i] * self.sigma_arm[i]
                B[row, self._po.start + i] -= (self.exp_share[i] * self.sigma_cet[i] + self.imp_share[i] * self.sigma_arm[i])

        return B, b

    def solve_shock(self, shock_params: Dict[str, Any]) -> JohansenResult:
        """求解政策冲击的百分比变化效应。

        Args:
            shock_params: 政策冲击参数字典, 可包含:
              - consumption_tax_change: 消费税率变化 (百分点)
              - gov_spending_change: 政府支出%变化
              - corporate_tax_change: 企业所得税率变化
              - production_tax_change: 生产税率变化
              - labor_supply_change: 劳动供给%变化
              - capital_supply_change: 资本供给%变化
              - exchange_rate_change: 汇率变化(None=内生)
              - interest_rate_change: 利率变化(百分点, 如-0.005=降50bp)
              - tfp_shock: dict{sector_idx: pct_change} 或 float(全行业)
              - targeted_sector: int, 定向支持行业索引
              - targeted_investment: float, 定向投资增幅(%)

        Returns:
            JohansenResult
        """
        result = JohansenResult(shock_params=shock_params.copy())
        t0 = time.time()

        B, b = self._build_system(shock_params)

        try:
            x = np.linalg.solve(B, b)
            result.status = "optimal"
            result.walras_check = float(np.max(np.abs(B @ x - b)))
        except np.linalg.LinAlgError as e:
            result.convergence_log.append(f"矩阵奇异, 使用最小二乘: {e}")
            x, residuals, rank, sv = np.linalg.lstsq(B, b, rcond=None)
            result.status = "suboptimal"
            result.walras_check = float(np.max(np.abs(B @ x - b)))

        result.solve_time = time.time() - t0

        # 提取百分比变化
        n = self.n
        pct = {}
        pct["po"] = x[self._po]
        pct["qo"] = x[self._qo]
        pct["qh"] = x[self._qh]
        pct["qg"] = x[self._qg]
        pct["qinv"] = x[self._qinv]
        pct["w"] = x[self._w]
        pct["r"] = x[self._r]
        pct["yh"] = x[self._yh]
        pct["sav"] = x[self._sav]
        pct["exr"] = x[self._exr]

        # 派生: 进出口变化
        pct["qe"] = self.sigma_cet * (x[self._exr] - x[self._po])
        pct["qm"] = self.sigma_arm * (x[self._po] - x[self._exr])

        # 派生: CPI
        cons_tax_d = shock_params.get("consumption_tax_change", 0.0)
        cons_tax_effect = cons_tax_d / (1.0 + self.cons_tax_rate)
        cpi_change = np.sum(self.beta_cons * (x[self._po] + cons_tax_effect))
        pct["cpi"] = cpi_change

        # GDP变化
        va_weights = (self.sam.labor_comp + self.sam.capital_income + self.sam.production_tax) / (self.sam.gdp + 1e-10)
        gdp_change = np.sum(va_weights * x[self._qo])
        result.gdp_change = float(gdp_change)
        result.cpi_change = float(cpi_change)

        # 就业变化
        tfp_shock = shock_params.get("tfp_shock", None)
        tfp_vec = np.zeros(n)
        if tfp_shock:
            if isinstance(tfp_shock, dict):
                for idx, val in tfp_shock.items():
                    if 0 <= idx < n:
                        tfp_vec[idx] = val
            elif isinstance(tfp_shock, (int, float)):
                tfp_vec[:] = tfp_shock

        emp_change = np.sum(self.lab_alloc_sh * (
            x[self._qo] - tfp_vec - self.sigma_va * self.alpha_K * (x[self._w] - x[self._r])
        ))
        result.employment_change = float(emp_change)

        # 福利变化
        real_cons_change = x[self._yh] - cpi_change
        result.welfare_change = float(real_cons_change)

        # 投资变化
        total_inv_change = np.sum(self.inv_sh * x[self._qinv])
        result.investment_change = float(total_inv_change)

        # ---- v2.0: 财政平衡计算 ----
        result.fiscal_balance, result.fiscal_balance_pct = self._compute_fiscal_balance(
            x, shock_params, gdp_change
        )

        # ---- v2.0: 信心指数计算 ----
        cc, ec, ic = self._compute_confidence(
            x, gdp_change, cpi_change, emp_change,
            total_inv_change, shock_params
        )
        result.confidence_consumer = cc
        result.confidence_enterprise = ec
        result.confidence_investor = ic

        # 变化后水平值
        levels = {}
        levels["output"] = self.baseline_output * (1.0 + x[self._qo])
        levels["household_cons"] = self.sam.household_cons * (1.0 + x[self._qh])
        levels["government_cons"] = self.sam.government_cons * (1.0 + x[self._qg])
        levels["investment"] = self.sam.investment * (1.0 + x[self._qinv])
        levels["exports"] = self.sam.exports * (1.0 + pct["qe"])
        levels["imports"] = self.sam.imports * (1.0 + pct["qm"])
        levels["gdp"] = self.baseline_gdp * (1.0 + gdp_change)
        levels["wage"] = 1.0 + x[self._w]
        levels["rental_rate"] = 1.0 + x[self._r]
        levels["exchange_rate"] = 1.0 + x[self._exr]

        result.pct_changes = pct
        result.levels = levels

        result.convergence_log.append(
            f"Johansen v2.0求解完成: {self.n}部门, "
            f"耗时{result.solve_time*1000:.1f}ms, "
            f"瓦尔拉斯残差={result.walras_check:.2e}"
        )

        return result

    def _compute_fiscal_balance(self, x, shock, gdp_change):
        """计算财政盈余/赤字变化。

        财政收入 = 消费税 + 企业所得税 + 生产税 + 个人所得税
        财政支出 = 政府消费 + 定向补贴 + 其他转移
        """
        n = self.n

        # 基准财政收入 (亿元)
        baseline_rev = self.baseline_fiscal_revenue

        # 收入变化
        # 1. 消费税收入变化 = cons_tax_rate * Δ(居民消费)
        cons_tax_rate = self.cons_tax_rate
        hh_cons_change = np.sum(self.sam.household_cons * x[self._qh])
        cons_tax_rev_change = cons_tax_rate * hh_cons_change

        # 2. 企业所得税收入变化 = corp_tax_rate * Δ(资本回报)
        corp_tax_rate = self.corp_tax_rate
        cap_income_change = np.sum(self.sam.capital_income * x[self._r])
        corp_tax_rev_change = corp_tax_rate * cap_income_change

        # 3. 生产税收入变化
        prod_tax_rev_change = np.sum(self.sam.production_tax * x[self._qo])

        # 4. 政策税率调整的直接影响
        cons_tax_d = shock.get("consumption_tax_change", 0.0)
        corp_tax_d = shock.get("corporate_tax_change", 0.0)
        prod_tax_d = shock.get("production_tax_change", 0.0)

        # 税率变化的收入效应 = Δrate * base
        cons_tax_policy_effect = cons_tax_d * self.sam.household_cons.sum()
        corp_tax_policy_effect = corp_tax_d * self.sam.capital_income.sum()
        prod_tax_policy_effect = prod_tax_d * self.baseline_output.sum()

        total_rev_change = (cons_tax_rev_change + corp_tax_rev_change +
                           prod_tax_rev_change +
                           cons_tax_policy_effect + corp_tax_policy_effect +
                           prod_tax_policy_effect)

        # 支出变化
        gov_spending_d = shock.get("gov_spending_change", 0.0)
        gov_exp_change = gov_spending_d * self.sam.government_cons.sum()

        # 定向投资补贴成本 (支持多行业)
        targeted_sector = shock.get("targeted_sector", None)
        targeted_investment = shock.get("targeted_investment", 0.0)
        targeted_investments = shock.get("targeted_investments", None)
        subsidy_cost = 0.0
        subsidy_rate = 0.30  # 政府承担30%的投资成本
        if targeted_investments is not None:
            for idx, pct in targeted_investments.items():
                if abs(pct) > 1e-8 and 0 <= idx < len(self.sam.investment):
                    subsidy_cost += pct * self.sam.investment[idx] * subsidy_rate
        elif targeted_sector is not None and abs(targeted_investment) > 1e-8:
            base_investment = self.sam.investment[targeted_sector]
            subsidy_cost = targeted_investment * base_investment * subsidy_rate

        total_exp_change = gov_exp_change + subsidy_cost

        # 财政平衡变化 = 收入变化 - 支出变化
        fiscal_balance_change = total_rev_change - total_exp_change

        # 财政平衡/GDP变化(百分点)
        new_gdp = self.baseline_gdp * (1.0 + gdp_change)
        fiscal_balance_pct = fiscal_balance_change / (new_gdp + 1e-10) * 100

        return float(fiscal_balance_change), float(fiscal_balance_pct)

    def _compute_confidence(self, x, gdp_change, cpi_change, emp_change,
                            inv_change, shock):
        """计算三部门信心指数变化。

        基准值 = 50 (中性)
        正值 = 信心改善, 负值 = 信心恶化

        v2.0修正:
          - 投资项clip到±10%防极端值(Johansen线性化对TFP冲击的投资反应过大)
          - 投资权重从2.0降至1.0(投资减少不一定利空,如TFP提升时资本替代是正常的)
          - 新增TFP直接项: 生产率提升直接利好企业和投资者信心
        """
        interest_rate_d = shock.get("interest_rate_change", 0.0)
        exr_change = float(x[self._exr])

        # 财政改善 (来自_compute_fiscal_balance)
        fiscal_bal, fiscal_pct = self._compute_fiscal_balance(x, shock, gdp_change)
        # 归一化: 每1000亿元改善 → +1点信心
        fiscal_signal = fiscal_bal / 1000.0

        # TFP信号: 生产率提升直接利好企业和投资者信心
        tfp_shock = shock.get("tfp_shock", 0.0)
        if isinstance(tfp_shock, dict):
            # 定向TFP: 按部门产出加权为全经济TFP信号
            total_out = self.baseline_output.sum() + 1e-10
            tfp_signal = sum(v * self.baseline_output[k] for k, v in tfp_shock.items()) / total_out
        else:
            tfp_signal = float(tfp_shock)

        # 投资变化clip到±10%(防Johansen线性化极端值)
        inv_clipped = max(-0.10, min(0.10, inv_change))

        # 消费者信心: GDP↑, CPI↓, 就业↑, 财政改善, TFP↑(降价利好) → 信心↑
        consumer = (
            3.0 * gdp_change * 100        # GDP每+1% → +3点
            - 2.0 * cpi_change * 100      # CPI每+1% → -2点
            + 1.5 * emp_change * 100      # 就业每+1% → +1.5点
            + 0.5 * fiscal_signal          # 财政改善 → +0.5点/千亿
            + 3.0 * tfp_signal * 100       # TFP每+1% → +3点(生产率提升→商品降价→实际收入提升)
        )

        # 企业信心: GDP↑, 利率↓, 投资↑, 财政改善, TFP↑ → 信心↑
        # 投资权重1.0(降低,因投资减少在TFP提升时是正常的资本替代)
        # TFP直接计入: 生产率提升=利润空间扩大+竞争力增强
        enterprise = (
            2.5 * gdp_change * 100
            - 3.0 * interest_rate_d * 100  # 利率每-1% → +3点 (降息利好)
            + 1.0 * inv_clipped * 100       # 投资每+1% → +1点(clip后)
            + 0.8 * fiscal_signal
            + 10.0 * tfp_signal * 100       # TFP每+1% → +10点(直接生产率利好,权重较大)
        )

        # 投资者信心: GDP↑, 利率↓, 汇率(贬值利好出口), 财政改善, TFP↑
        investor = (
            2.0 * gdp_change * 100
            - 2.5 * interest_rate_d * 100
            + 1.0 * exr_change * 100       # 汇率贬值(+)→出口利好→投资者信心↑
            + 1.0 * fiscal_signal
            + 5.0 * tfp_signal * 100        # TFP每+1% → +5点(资本回报率提升)
        )

        return float(consumer), float(enterprise), float(investor)

    def solve_baseline(self) -> JohansenResult:
        """求解基准情景(无冲击)。"""
        return self.solve_shock({})

    def solve_multi_step(self, shock_params: Dict[str, Any],
                          steps: int = 5) -> JohansenResult:
        """多步Johansen求解(提高大冲击的精度)。"""
        if steps <= 1:
            return self.solve_shock(shock_params)

        sub_shock = {}
        for k, v in shock_params.items():
            if isinstance(v, (int, float)):
                sub_shock[k] = v / steps
            else:
                sub_shock[k] = v

        accumulated_pct = None
        for step in range(steps):
            result = self.solve_shock(sub_shock)
            if accumulated_pct is None:
                accumulated_pct = {k: v.copy() if hasattr(v, 'copy') else v
                                   for k, v in result.pct_changes.items()}
            else:
                for k in accumulated_pct:
                    old = accumulated_pct[k]
                    new = result.pct_changes[k]
                    if hasattr(old, '__add__') and hasattr(new, '__add__'):
                        accumulated_pct[k] = old + new + old * new
                    else:
                        accumulated_pct[k] = old + new

        result.pct_changes = accumulated_pct
        return result


class DynamicJohansenSolver:
    """递归动态CGE求解器 v2.0 — 12季度(3年)路径。

    每期用Johansen求解器求解静态均衡, 期与期之间通过资本积累和
    预期调整链接。支持:
      - 永久/暂时/预告 三种时序模式
      - 财政平衡动态路径
      - 信心指数动态路径(AR(1)持续性)
      - 分行业产出动态路径(用于行业钻取)
    """

    def __init__(self, sam: SAMData, horizon: int = 12,
                 depreciation_rate: float = 0.05):
        self.base_solver = JohansenSolver(sam)
        self.n = sam.num_sectors
        self.horizon = horizon
        self.delta = depreciation_rate  # 年度折旧率
        self.sam = sam

    def solve_dynamic(self, shock_params: Dict[str, Any],
                      shock_timing: str = "permanent",
                      implementation_lag: int = 0) -> Dict[str, Any]:
        """求解12季度动态路径。

        Args:
            shock_params: 政策冲击参数
            shock_timing: "permanent"(永久), "temporary"(一次性脉冲),
                          "anticipated"(预告后实施)
            implementation_lag: 政策实施滞后期(季度), 仅anticipated模式

        Returns:
            dict with:
              'baseline_path': 基线路径
              'counterfactual_path': 政策路径
              'deviation': 偏差百分比
              'quarters': [Q1, Q2, ..., Q12]
              'sector_output_path': 42部门×12季度产出路径
              'fiscal_path': 财政平衡路径
              'confidence_path': 信心指数路径
        """
        horizon = self.horizon

        # 基线路径
        baseline_path = {
            "gdp": np.zeros(horizon),
            "employment": np.zeros(horizon),
            "cpi": np.zeros(horizon),
            "investment": np.zeros(horizon),
            "welfare": np.zeros(horizon),
        }

        # 政策路径
        cf_path = {
            "gdp": np.zeros(horizon),
            "employment": np.zeros(horizon),
            "cpi": np.zeros(horizon),
            "investment": np.zeros(horizon),
            "welfare": np.zeros(horizon),
        }

        # 静态冲击效应
        static_result = self.base_solver.solve_shock(shock_params)
        static_gdp = static_result.gdp_change
        static_emp = static_result.employment_change
        static_cpi = static_result.cpi_change
        static_inv = static_result.investment_change
        static_welfare = static_result.welfare_change

        # v2.0: 静态财政和信心
        static_fiscal = static_result.fiscal_balance_pct
        static_cc = static_result.confidence_consumer
        static_ec = static_result.confidence_enterprise
        static_ic = static_result.confidence_investor

        # v2.0: 分行业静态产出变化
        static_sector_output = static_result.pct_changes["qo"].copy()

        # 动态路径
        fiscal_path = np.zeros(horizon)
        cc_path = np.zeros(horizon)
        ec_path = np.zeros(horizon)
        ic_path = np.zeros(horizon)
        sector_output_path = np.zeros((self.n, horizon))

        # 季度调整参数
        adjustment_speed = 0.15  # 季度调整速度
        short_run_mult = 1.4     # 短期乘数
        long_run_mult = 1.0      # 长期乘数

        for t in range(horizon):
            quarter = t + 1

            if shock_timing == "permanent":
                if implementation_lag > 0 and quarter <= implementation_lag:
                    # 预告期: 部分预期效应
                    anticipation_factor = 0.3 * quarter / implementation_lag
                    effect_factor = anticipation_factor * 0.2
                else:
                    # 实施后: 逐渐调整
                    t_since = max(quarter - implementation_lag, 1)
                    transition = 1.0 - np.exp(-adjustment_speed * t_since)
                    effect_factor = short_run_mult - (short_run_mult - long_run_mult) * transition

                cf_path["gdp"][t] = static_gdp * effect_factor
                cf_path["employment"][t] = static_emp * effect_factor
                cf_path["cpi"][t] = static_cpi * (1.0 - np.exp(-0.25 * quarter)) / (1.0 - np.exp(-0.25 * horizon))
                cf_path["investment"][t] = static_inv * effect_factor
                cf_path["welfare"][t] = static_welfare * effect_factor

            elif shock_timing == "temporary":
                # 一次性脉冲: 实施季度满效应, 之后指数衰减
                if implementation_lag > 0 and quarter <= implementation_lag:
                    effect_factor = 0.2 * quarter / implementation_lag
                elif quarter == implementation_lag + 1:
                    effect_factor = 1.0
                else:
                    t_since = quarter - implementation_lag - 1
                    decay = np.exp(-0.35 * t_since)
                    effect_factor = decay

                cf_path["gdp"][t] = static_gdp * effect_factor
                cf_path["employment"][t] = static_emp * effect_factor
                cf_path["cpi"][t] = static_cpi * effect_factor
                cf_path["investment"][t] = static_inv * effect_factor
                cf_path["welfare"][t] = static_welfare * effect_factor

            elif shock_timing == "anticipated":
                # 预告后实施: t=0预告, t=lag实施
                if quarter < implementation_lag:
                    # 预告期: 消费平滑, 投资提前反应
                    pre_effect = 0.15 * (quarter / implementation_lag)
                    cf_path["gdp"][t] = static_gdp * pre_effect
                    cf_path["employment"][t] = static_emp * pre_effect * 0.5
                    cf_path["cpi"][t] = static_cpi * pre_effect * 0.3
                    cf_path["investment"][t] = static_inv * pre_effect * 1.5
                    cf_path["welfare"][t] = static_welfare * pre_effect
                else:
                    t_since = quarter - implementation_lag + 1
                    transition = 1.0 - np.exp(-adjustment_speed * t_since)
                    effect_factor = short_run_mult - (short_run_mult - long_run_mult) * transition
                    cf_path["gdp"][t] = static_gdp * effect_factor
                    cf_path["employment"][t] = static_emp * effect_factor
                    cf_path["cpi"][t] = static_cpi * effect_factor
                    cf_path["investment"][t] = static_inv * effect_factor
                    cf_path["welfare"][t] = static_welfare * effect_factor

            # v2.0: 财政平衡路径
            # 财政效应有滞后性: 政策实施后逐渐显现
            fiscal_transition = 1.0 - np.exp(-0.2 * max(quarter - implementation_lag, 0.5))
            fiscal_path[t] = static_fiscal * fiscal_transition * effect_factor / max(effect_factor, 0.01) if effect_factor != 0 else 0

            # v2.0: 信心指数路径 (AR(1)持续性)
            # 信心指数有惯性: ρ=0.85 (季度持续性)
            rho = 0.85
            if t == 0:
                cc_path[t] = static_cc * effect_factor
                ec_path[t] = static_ec * effect_factor
                ic_path[t] = static_ic * effect_factor
            else:
                # 信心 = ρ * 上期信心 + (1-ρ) * 本期冲击效应
                current_shock_cc = static_cc * effect_factor
                current_shock_ec = static_ec * effect_factor
                current_shock_ic = static_ic * effect_factor
                cc_path[t] = rho * cc_path[t-1] + (1 - rho) * current_shock_cc
                ec_path[t] = rho * ec_path[t-1] + (1 - rho) * current_shock_ec
                ic_path[t] = rho * ic_path[t-1] + (1 - rho) * current_shock_ic

            # v2.0: 分行业产出路径
            # 各行业按同一effect_factor过渡, 但加入行业特异性调整速度
            for i in range(self.n):
                # 行业调整速度差异: 制造业调整快, 服务业慢
                if i < 27:  # 制造业部门
                    sector_speed = adjustment_speed * 1.2
                else:  # 服务业部门
                    sector_speed = adjustment_speed * 0.8
                sector_transition = 1.0 - np.exp(-sector_speed * max(quarter - implementation_lag, 0.5))
                sector_mult = short_run_mult - (short_run_mult - long_run_mult) * sector_transition
                sector_output_path[i, t] = static_sector_output[i] * sector_mult

        # 偏差
        deviation = {}
        for key in baseline_path:
            deviation[key] = cf_path[key] - baseline_path[key]

        return {
            "baseline_path": baseline_path,
            "counterfactual_path": cf_path,
            "deviation": deviation,
            "quarters": list(range(1, horizon + 1)),
            "static_result": static_result,
            "horizon": horizon,
            "shock_timing": shock_timing,
            "implementation_lag": implementation_lag,
            # v2.0 新增
            "fiscal_path": fiscal_path,
            "confidence_path": {
                "consumer": cc_path,
                "enterprise": ec_path,
                "investor": ic_path,
            },
            "sector_output_path": sector_output_path,  # (42, 12) 分行业产出路径
        }
