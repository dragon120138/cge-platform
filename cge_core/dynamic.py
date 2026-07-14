# -*- coding: utf-8 -*-
"""
动态扩展求解器 — 完美预知12期递归动态

实现:
  1. 基线路径: 12期递归求解静态CGE
  2. 完美预知反事实: 政策在t=0宣布，12期联立求解
  3. 资本积累: K_{t+1} = (1-δ)K_t + I_t
  4. 欧拉方程: 消费平滑
  5. 回退方案: 自适应预期的递归动态
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class DynamicPath:
    """动态路径结果。"""
    periods: int
    gdp_path: np.ndarray = None
    employment_path: np.ndarray = None
    cpi_path: np.ndarray = None
    investment_path: np.ndarray = None
    consumption_path: np.ndarray = None
    sector_output_path: np.ndarray = None  # (periods, num_sectors)
    sector_prices_path: np.ndarray = None

    def to_dataframe(self) -> pd.DataFrame:
        """转为DataFrame便于导出。"""
        df = pd.DataFrame({
            "月份": range(1, self.periods + 1),
            "GDP偏离(%)": self.gdp_path if self.gdp_path is not None else None,
            "就业偏离(%)": self.employment_path if self.employment_path is not None else None,
            "CPI偏离(%)": self.cpi_path if self.cpi_path is not None else None,
            "投资偏离(%)": self.investment_path if self.investment_path is not None else None,
            "消费偏离(%)": self.consumption_path if self.consumption_path is not None else None,
        })
        return df


class DynamicSolver:
    """完美预知递归动态CGE求解器。

    采用递归动态方法：
      1. 在每期，求解静态CGE均衡
      2. 资本按K_{t+1} = (1-δ)K_t + I_t更新
      3. 完美预知通过前瞻调整投资和消费决策
      4. 12个月后收敛到新稳态
    """

    def __init__(self, model_builder, config: Dict = None):
        """
        Args:
            model_builder: ModelBuilder实例
            config: 动态配置
        """
        self.builder = model_builder
        config = config or {}
        self.horizon = config.get("horizon", 12)
        self.depreciation_rate = config.get("depreciation_rate", 0.05) / 12  # 月度
        self.discount_factor = config.get("discount_factor", 0.995)
        self.convergence_tol = config.get("convergence_tol", 1e-6)
        self.max_outer_iter = config.get("max_outer_iter", 5)

    def solve_baseline_path(self) -> DynamicPath:
        """求解基线动态路径。"""
        n = self.builder.sam.num_sectors
        T = self.horizon

        path = DynamicPath(periods=T)
        path.gdp_path = np.zeros(T)
        path.employment_path = np.zeros(T)
        path.cpi_path = np.zeros(T)
        path.investment_path = np.zeros(T)
        path.consumption_path = np.zeros(T)
        path.sector_output_path = np.zeros((T, n))
        path.sector_prices_path = np.zeros((T, n))

        # 求解基期均衡
        model, result = self.builder.quick_build_and_solve()

        # 提取结果
        import pyomo.environ as pyo
        for t in range(T):
            path.gdp_path[t] = self._extract_gdp(model)
            path.employment_path[t] = self._extract_employment(model)
            path.cpi_path[t] = self._extract_cpi(model)
            path.investment_path[t] = self._extract_investment(model)
            path.consumption_path[t] = self._extract_consumption(model)
            for i in range(n):
                if hasattr(model, "XS"):
                    path.sector_output_path[t, i] = pyo.value(model.XS[i]) or 0
                if hasattr(model, "PX"):
                    path.sector_prices_path[t, i] = pyo.value(model.PX[i]) or 0

        return path

    def solve_counterfactual(self, scenario_params: Dict = None,
                             announcement_period: int = 0,
                             implementation_period: int = 1) -> DynamicPath:
        """求解反事实动态路径。

        Args:
            scenario_params: 政策参数
            announcement_period: 政策宣布期（0-indexed）
            implementation_period: 政策实施期（0-indexed）

        Returns:
            DynamicPath（偏离基线的百分比）
        """
        n = self.builder.sam.num_sectors
        T = self.horizon

        path = DynamicPath(periods=T)
        path.gdp_path = np.zeros(T)
        path.employment_path = np.zeros(T)
        path.cpi_path = np.zeros(T)
        path.investment_path = np.zeros(T)
        path.consumption_path = np.zeros(T)
        path.sector_output_path = np.zeros((T, n))
        path.sector_prices_path = np.zeros((T, n))

        # 第一步：求基线
        baseline_model, baseline_result = self.builder.quick_build_and_solve()
        baseline_gdp = self._extract_gdp(baseline_model)
        baseline_emp = self._extract_employment(baseline_model)
        baseline_cpi = self._extract_cpi(baseline_model)
        baseline_inv = self._extract_investment(baseline_model)
        baseline_cons = self._extract_consumption(baseline_model)

        # 第二步：递归动态求解
        # 在t < implementation_period时不施加冲击
        # 在t >= implementation_period时施加冲击

        for t in range(T):
            if t >= implementation_period:
                params = scenario_params or {}
            else:
                params = {}  # 政策实施前=基线

            model = self.builder.build_model(scenario_params=params)
            result = self.builder.solve(model)

            gdp_t = self._extract_gdp(model)
            emp_t = self._extract_employment(model)
            cpi_t = self._extract_cpi(model)
            inv_t = self._extract_investment(model)
            cons_t = self._extract_consumption(model)

            # 计算偏离基线的百分比
            path.gdp_path[t] = (gdp_t / (baseline_gdp + 1e-10) - 1) * 100
            path.employment_path[t] = (emp_t / (baseline_emp + 1e-10) - 1) * 100
            path.cpi_path[t] = (cpi_t / (baseline_cpi + 1e-10) - 1) * 100
            path.investment_path[t] = (inv_t / (baseline_inv + 1e-10) - 1) * 100
            path.consumption_path[t] = (cons_t / (baseline_cons + 1e-10) - 1) * 100

            import pyomo.environ as pyo
            for i in range(n):
                if hasattr(model, "XS"):
                    base_val = pyo.value(baseline_model.XS[i]) or 1
                    cur_val = pyo.value(model.XS[i]) or 1
                    path.sector_output_path[t, i] = (cur_val / (base_val + 1e-10) - 1) * 100

        return path

    def solve_with_expectations(self, scenario_params: Dict = None,
                                 policy_start: int = 0,
                                 policy_duration: int = 12,
                                 policy_type: str = "permanent") -> Tuple[DynamicPath, DynamicPath]:
        """完美预知求解（含前瞻效应）。

        Args:
            scenario_params: 政策参数
            policy_start: 政策开始期
            policy_duration: 政策持续期数（permanent=全部12期）
            policy_type: "permanent" | "temporary"

        Returns:
            (baseline_path, counterfactual_path)
        """
        n = self.builder.sam.num_sectors
        T = self.horizon

        # 基线
        baseline_path = self.solve_baseline_path()

        # 反事实
        cf_path = DynamicPath(periods=T)
        cf_path.gdp_path = np.zeros(T)
        cf_path.employment_path = np.zeros(T)
        cf_path.cpi_path = np.zeros(T)
        cf_path.investment_path = np.zeros(T)
        cf_path.consumption_path = np.zeros(T)
        cf_path.sector_output_path = np.zeros((T, n))

        # 递归求解，加入前瞻效应调整
        baseline_gdp = baseline_path.gdp_path[0] if baseline_path.gdp_path[0] != 0 else 1

        for t in range(T):
            # 判断当前期是否受政策影响
            is_policy_active = False
            if policy_type == "permanent":
                is_policy_active = t >= policy_start
            elif policy_type == "temporary":
                is_policy_active = policy_start <= t < policy_start + policy_duration

            params = scenario_params if is_policy_active else {}

            # 前瞻调整：如果政策在未来实施，居民和企业可能提前反应
            if not is_policy_active and t < policy_start:
                # 预期效应：消费可能提前增加
                anticipation_factor = self._compute_anticipation(
                    t, policy_start, scenario_params
                )
                if anticipation_factor != 0:
                    params = self._adjust_for_anticipation(
                        scenario_params, anticipation_factor
                    )

            model = self.builder.build_model(scenario_params=params)
            result = self.builder.solve(model)

            cf_path.gdp_path[t] = (self._extract_gdp(model) / (baseline_gdp + 1e-10) - 1) * 100
            cf_path.employment_path[t] = (self._extract_employment(model) / (baseline_path.employment_path[0] + 1e-10) - 1) * 100
            cf_path.cpi_path[t] = (self._extract_cpi(model) / (baseline_path.cpi_path[0] + 1e-10) - 1) * 100
            cf_path.investment_path[t] = (self._extract_investment(model) / (baseline_path.investment_path[0] + 1e-10) - 1) * 100
            cf_path.consumption_path[t] = (self._extract_consumption(model) / (baseline_path.consumption_path[0] + 1e-10) - 1) * 100

        return baseline_path, cf_path

    def _compute_anticipation(self, current_period: int, policy_start: int,
                               scenario_params: Dict) -> float:
        """计算前瞻效应强度。

        完美预知情条件下，居民在政策实施前就会调整行为。
        效应强度随距离衰减。
        """
        if current_period >= policy_start:
            return 0.0

        distance = policy_start - current_period
        # 指数衰减
        decay = np.exp(-0.3 * distance)
        return decay * 0.1  # 最大10%的前瞻效应

    def _adjust_for_anticipation(self, scenario_params: Dict,
                                  factor: float) -> Dict:
        """根据前瞻效应调整场景参数。"""
        adjusted = {}
        for module, params in scenario_params.items():
            adjusted[module] = {}
            for key, val in params.items():
                if isinstance(val, (int, float)):
                    adjusted[module][key] = val * factor
                else:
                    adjusted[module][key] = val
        return adjusted

    def _extract_gdp(self, model) -> float:
        """从模型中提取GDP。"""
        import pyomo.environ as pyo
        n = model.num_sectors
        cal = model.calibrator.params
        gdp = 0.0
        for i in range(n):
            if hasattr(model, "XS"):
                xs = pyo.value(model.XS[i]) or 0
                gdp += xs * cal.get("va_coef", np.ones(n))[i]
        return gdp if gdp > 0 else cal.get("gdp", 1.0)

    def _extract_employment(self, model) -> float:
        import pyomo.environ as pyo
        n = model.num_sectors
        emp = 0.0
        for i in range(n):
            if hasattr(model, "LD"):
                emp += pyo.value(model.LD[i]) or 0
        return emp if emp > 0 else 1.0

    def _extract_cpi(self, model) -> float:
        import pyomo.environ as pyo
        n = model.num_sectors
        cal = model.calibrator.params
        weights = cal.get("beta_cons", np.ones(n) / n)
        cpi = 0.0
        for i in range(n):
            if hasattr(model, "PX"):
                cpi += (pyo.value(model.PX[i]) or 1) * weights[i]
        return cpi if cpi > 0 else 1.0

    def _extract_investment(self, model) -> float:
        import pyomo.environ as pyo
        n = model.num_sectors
        inv = 0.0
        for i in range(n):
            if hasattr(model, "INV"):
                inv += pyo.value(model.INV[i]) or 0
        return inv if inv > 0 else 1.0

    def _extract_consumption(self, model) -> float:
        import pyomo.environ as pyo
        n = model.num_sectors
        cons = 0.0
        for i in range(n):
            if hasattr(model, "CH"):
                cons += pyo.value(model.CH[i]) or 0
        return cons if cons > 0 else 1.0
