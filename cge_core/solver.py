# -*- coding: utf-8 -*-
"""
求解器封装 — 统一接口, 底层委托给Johansen求解器

不再依赖Pyomo/IPOPT。所有求解通过Johansen对数线性化完成(纯numpy)。
"""

import numpy as np
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .johansen import JohansenSolver, JohansenResult, DynamicJohansenSolver


@dataclass
class SolverResult:
    """求解结果容器 — 兼容旧接口。"""
    status: str = "unknown"
    objective: float = 0.0
    solve_time: float = 0.0
    iterations: int = 0
    variables: Dict[str, Any] = field(default_factory=dict)
    dual_variables: Dict[str, Any] = field(default_factory=dict)
    walras_check: float = 0.0
    convergence_log: List[str] = field(default_factory=list)
    solver_name: str = "johansen"
    message: str = ""

    # Johansen特有字段
    pct_changes: Dict[str, Any] = field(default_factory=dict)
    levels: Dict[str, Any] = field(default_factory=dict)
    gdp_change: float = 0.0
    cpi_change: float = 0.0
    employment_change: float = 0.0
    welfare_change: float = 0.0
    investment_change: float = 0.0
    shock_params: Dict[str, Any] = field(default_factory=dict)
    # v2.0 新增
    fiscal_balance: float = 0.0
    fiscal_balance_pct: float = 0.0
    confidence_consumer: float = 0.0
    confidence_enterprise: float = 0.0
    confidence_investor: float = 0.0

    @property
    def success(self) -> bool:
        return self.status in ("optimal", "suboptimal")

    @classmethod
    def from_johansen(cls, jr: JohansenResult) -> "SolverResult":
        """从JohansenResult转换。"""
        r = cls(
            status=jr.status,
            solve_time=jr.solve_time,
            walras_check=jr.walras_check,
            convergence_log=jr.convergence_log.copy(),
            solver_name="johansen",
            pct_changes=jr.pct_changes,
            levels=jr.levels,
            gdp_change=jr.gdp_change,
            cpi_change=jr.cpi_change,
            employment_change=jr.employment_change,
            welfare_change=jr.welfare_change,
            investment_change=jr.investment_change,
            shock_params=jr.shock_params,
            # v2.0
            fiscal_balance=jr.fiscal_balance,
            fiscal_balance_pct=jr.fiscal_balance_pct,
            confidence_consumer=jr.confidence_consumer,
            confidence_enterprise=jr.confidence_enterprise,
            confidence_investor=jr.confidence_investor,
            message="; ".join(jr.convergence_log),
        )
        # 兼容: variables字段
        r.variables = {k: v for k, v in jr.levels.items()}
        r.variables["gdp_pct"] = jr.gdp_change
        r.variables["cpi_pct"] = jr.cpi_change
        r.variables["employment_pct"] = jr.employment_change
        r.variables["welfare_pct"] = jr.welfare_change
        r.variables["investment_pct"] = jr.investment_change
        # v2.0
        r.variables["fiscal_balance"] = jr.fiscal_balance
        r.variables["fiscal_balance_pct"] = jr.fiscal_balance_pct
        r.variables["confidence_consumer"] = jr.confidence_consumer
        r.variables["confidence_enterprise"] = jr.confidence_enterprise
        r.variables["confidence_investor"] = jr.confidence_investor
        return r


class SolverManager:
    """求解器管理器 — 统一接口。"""

    def __init__(self):
        self.available_solvers = ["johansen"]
        self.preferred_solver = "johansen"

    def solve(self, solver: JohansenSolver,
              shock_params: Dict[str, Any]) -> SolverResult:
        """使用Johansen求解器求解政策冲击。

        Args:
            solver: JohansenSolver实例
            shock_params: 政策冲击参数

        Returns:
            SolverResult
        """
        jr = solver.solve_shock(shock_params)
        return SolverResult.from_johansen(jr)

    def solve_dynamic(self, dyn_solver: DynamicJohansenSolver,
                      shock_params: Dict[str, Any],
                      shock_timing: str = "permanent",
                      implementation_lag: int = 0) -> Dict[str, Any]:
        """求解动态路径。"""
        return dyn_solver.solve_dynamic(
            shock_params, shock_timing=shock_timing,
            implementation_lag=implementation_lag
        )
