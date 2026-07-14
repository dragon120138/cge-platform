# -*- coding: utf-8 -*-
"""
场景引擎 — 基线→冲击→求解→比较

管理政策模拟的完整工作流：
  1. 定义基线场景（无政策变化）
  2. 定义冲击场景（如减税1%、万亿政府购买等）
  3. 分别求解
  4. 计算偏离度（% deviation from baseline）
  5. 提取宏观指标（GDP、就业、通胀、福利）
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy


@dataclass
class Scenario:
    """场景定义。"""
    name: str
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    # params示例:
    # {
    #     "government": {"consumption_tax_rate_change": -0.01},  # 消费税减1%
    #     "government": {"gov_purchase_multiplier": 1.1},        # 政府购买增加10%
    # }

    def __repr__(self):
        return f"<Scenario: {self.name}>"


@dataclass
class MacroResults:
    """宏观结果汇总。"""
    gdp: float = 0.0
    total_output: float = 0.0
    employment: float = 0.0       # 就业总量（劳动投入指数）
    household_welfare: float = 0.0  # 居民福利（等效变换EV）
    cpi_index: float = 1.0        # CPI指数
    gov_revenue: float = 0.0      # 政府收入
    gov_expenditure: float = 0.0  # 政府支出
    trade_balance: float = 0.0    # 贸易差额
    investment: float = 0.0       # 总投资
    sector_output: np.ndarray = None  # 部门产出向量
    sector_prices: np.ndarray = None  # 部门价格向量
    sector_employment: np.ndarray = None  # 部门就业

    def to_dict(self) -> Dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        return d


@dataclass
class ScenarioComparison:
    """场景比较结果。"""
    baseline: MacroResults
    counterfactual: MacroResults
    scenario: Scenario

    @property
    def gdp_deviation_pct(self) -> float:
        return (self.counterfactual.gdp / self.baseline.gdp - 1) * 100

    @property
    def employment_deviation_pct(self) -> float:
        return (self.counterfactual.employment / max(self.baseline.employment, 1e-10) - 1) * 100

    @property
    def cpi_deviation_pct(self) -> float:
        return (self.counterfactual.cpi_index / self.baseline.cpi_index - 1) * 100

    @property
    def welfare_change_pct(self) -> float:
        """等效变换（EV）占基线收入百分比。"""
        return ((self.counterfactual.household_welfare - self.baseline.household_welfare)
                / max(self.baseline.household_welfare, 1e-10)) * 100

    @property
    def sector_winners(self) -> List[Tuple[str, float]]:
        """产出增加最大的部门。"""
        if self.baseline.sector_output is None:
            return []
        dev = (self.counterfactual.sector_output / (self.baseline.sector_output + 1e-10) - 1) * 100
        sorted_idx = np.argsort(dev)[::-1]
        return [(f"S{i+1:02d}", float(dev[i])) for i in sorted_idx[:5]]

    @property
    def sector_losers(self) -> List[Tuple[str, float]]:
        """产出减少最大的部门。"""
        if self.baseline.sector_output is None:
            return []
        dev = (self.counterfactual.sector_output / (self.baseline.sector_output + 1e-10) - 1) * 100
        sorted_idx = np.argsort(dev)
        return [(f"S{i+1:02d}", float(dev[i])) for i in sorted_idx[:5]]

    def summary_table(self) -> pd.DataFrame:
        """生成摘要对比表。"""
        data = {
            "指标": ["GDP（亿元）", "就业指数", "CPI指数", "居民福利",
                      "政府收入（亿元）", "政府支出（亿元）", "贸易差额（亿元）", "总投资（亿元）"],
            "基线": [self.baseline.gdp, self.baseline.employment, self.baseline.cpi_index,
                     self.baseline.household_welfare, self.baseline.gov_revenue,
                     self.baseline.gov_expenditure, self.baseline.trade_balance,
                     self.baseline.investment],
            "反事实": [self.counterfactual.gdp, self.counterfactual.employment, self.counterfactual.cpi_index,
                       self.counterfactual.household_welfare, self.counterfactual.gov_revenue,
                       self.counterfactual.gov_expenditure, self.counterfactual.trade_balance,
                       self.counterfactual.investment],
        }
        df = pd.DataFrame(data)
        df["变化(%)"] = ((df["反事实"] / df["基线"].replace(0, 1e-10) - 1) * 100).round(3)
        return df


class ScenarioEngine:
    """场景管理引擎。"""

    def __init__(self, model_builder):
        """
        Args:
            model_builder: ModelBuilder实例
        """
        self.builder = model_builder
        self.last_baseline: Optional[MacroResults] = None

    def extract_results(self, model, result) -> MacroResults:
        """从求解后的模型中提取宏观结果。"""
        import pyomo.environ as pyo

        mr = MacroResults()

        # 提取部门产出
        n = model.num_sectors
        sector_output = np.ones(n)
        sector_prices = np.ones(n)
        sector_employment = np.ones(n)

        if hasattr(model, "XS"):
            for i in range(n):
                val = pyo.value(model.XS[i])
                if val is not None:
                    sector_output[i] = val

        if hasattr(model, "PX"):
            for i in range(n):
                val = pyo.value(model.PX[i])
                if val is not None:
                    sector_prices[i] = val

        if hasattr(model, "LD"):
            for i in range(n):
                val = pyo.value(model.LD[i])
                if val is not None:
                    sector_employment[i] = val

        mr.sector_output = sector_output
        mr.sector_prices = sector_prices
        mr.sector_employment = sector_employment

        # 宏观总量
        sam = model.sam
        cal = model.calibrator

        # GDP = 增加值之和
        va_base = sam.labor_comp + sam.capital_income + sam.production_tax
        mr.gdp = float(np.sum(sector_output * cal.params.get("va_coef", np.ones(n))))

        # 就业
        mr.employment = float(np.sum(sector_employment))

        # CPI = 加权平均价格
        hh_weights = cal.params.get("beta_cons", np.ones(n) / n)
        mr.cpi_index = float(np.sum(sector_prices * hh_weights))

        # 福利 ≈ 居民消费总额 / CPI
        if hasattr(model, "CH"):
            total_hh_cons = sum(pyo.value(model.CH[i]) or 0 for i in range(n))
            mr.household_welfare = float(total_hh_cons / max(mr.cpi_index, 1e-10))
        else:
            mr.household_welfare = mr.gdp * 0.4

        # 政府收支
        mr.gov_revenue = float(np.sum(sector_output * cal.params.get("va_tax_share", np.zeros(n))))
        mr.gov_expenditure = float(np.sum(sam.government_cons))

        # 贸易差额
        mr.trade_balance = float(np.sum(sam.exports - sam.imports))

        # 投资
        mr.investment = float(np.sum(sam.investment))

        # 总产出
        mr.total_output = float(np.sum(sector_output))

        return mr

    def run_baseline(self) -> Tuple[Any, MacroResults]:
        """运行基线场景。"""
        print("[场景引擎] 求解基线场景...")
        model = self.builder.build_model(scenario_params={})
        result = self.builder.solve(model)
        mr = self.extract_results(model, result)
        self.last_baseline = mr
        print(f"[场景引擎] 基线GDP = {mr.gdp:,.0f} 亿元, 就业指数 = {mr.employment:.4f}")
        return model, mr

    def run_scenario(self, scenario: Scenario,
                     baseline: MacroResults = None) -> ScenarioComparison:
        """运行单个政策冲击场景并与基线比较。

        Args:
            scenario: 场景定义
            baseline: 基线结果（None=自动求解基线）

        Returns:
            ScenarioComparison
        """
        # 基线
        if baseline is None:
            if self.last_baseline is None:
                _, baseline = self.run_baseline()
            else:
                baseline = self.last_baseline

        # 反事实
        print(f"[场景引擎] 求解场景: {scenario.name}...")
        model = self.builder.build_model(scenario_params=scenario.params)
        result = self.builder.solve(model)
        counterfactual = self.extract_results(model, result)

        comparison = ScenarioComparison(
            baseline=baseline,
            counterfactual=counterfactual,
            scenario=scenario,
        )

        print(f"[场景引擎] {scenario.name}: GDP偏离 = {comparison.gdp_deviation_pct:+.3f}%, "
              f"就业偏离 = {comparison.employment_deviation_pct:+.3f}%")

        return comparison

    def run_multiple(self, scenarios: List[Scenario]) -> List[ScenarioComparison]:
        """批量运行多个场景。"""
        comparisons = []
        _, baseline = self.run_baseline()
        for sc in scenarios:
            comp = self.run_scenario(sc, baseline=baseline)
            comparisons.append(comp)
        return comparisons


# ============================================================
# 预定义场景
# ============================================================

def scenario_consumption_tax_cut(rate_change: float = -0.01) -> Scenario:
    """消费税减免场景。"""
    return Scenario(
        name=f"消费税{'减' if rate_change < 0 else '增'}税{abs(rate_change)*100:.1f}%",
        description=f"消费税率变化 {rate_change*100:+.1f} 个百分点",
        params={"government": {"consumption_tax_rate_change": rate_change}},
    )


def scenario_gov_purchase_shock(amount_yi: float = 10000) -> Scenario:
    """政府购买冲击（万亿级）。"""
    return Scenario(
        name=f"政府购买+{amount_yi/1e4:.1f}万亿",
        description=f"政府购买增加 {amount_yi:,.0f} 亿元",
        params={"government": {"gov_purchase_increase_yi": amount_yi}},
    )


def scenario_corporate_tax_cut(rate_change: float = -0.05) -> Scenario:
    """企业所得税减免。"""
    return Scenario(
        name=f"企业所得税{'减' if rate_change < 0 else '增'}税{abs(rate_change)*100:.1f}%",
        description=f"企业所得税率变化 {rate_change*100:+.1f} 个百分点",
        params={"government": {"corporate_income_tax_rate_change": rate_change}},
    )


def scenario_carbon_tax(price_per_ton: float = 100) -> Scenario:
    """碳税场景。"""
    return Scenario(
        name=f"碳税{price_per_ton:.0f}元/吨CO₂",
        description=f"征收碳税 {price_per_ton:.0f} 元/吨CO₂",
        params={"carbon_tax": {"carbon_price": price_per_ton}},
    )
