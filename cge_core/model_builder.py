# -*- coding: utf-8 -*-
"""
模型构建器 — YAML配置 → Johansen求解器组装

不再依赖Pyomo。读取model_config.yaml, 加载SAM数据,
初始化Johansen对数线性化求解器, 执行政策冲击模拟。
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base_module import SAMData, ModuleRegistry
from .calibrator import Calibrator
from .solver import SolverManager, SolverResult
from .johansen import JohansenSolver, DynamicJohansenSolver
from .sectors import NUM_SECTORS, SECTOR_CODES, SECTOR_NAMES_CN


class ModelBuilder:
    """从配置构建CGE模型并求解。

    工作流程:
      1. 加载YAML配置
      2. 加载/生成SAM数据
      3. 校准参数
      4. 初始化Johansen求解器
      5. 执行政策冲击 → 返回结果
    """

    def __init__(self, config_path: str = None, config_dict: Dict = None):
        if config_dict:
            self.config = config_dict
        elif config_path:
            self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        else:
            self.config = self._default_config()

        self.sam: Optional[SAMData] = None
        self.calibrator: Optional[Calibrator] = None
        self.solver: Optional[JohansenSolver] = None
        self.dyn_solver: Optional[DynamicJohansenSolver] = None
        self.solver_manager = SolverManager()
        self.registry = ModuleRegistry()
        self._register_builtin_modules()

    def _default_config(self) -> Dict:
        return {
            "model": {
                "name": "中国CGE - 42部门 v2.0",
                "base_year": 2025,
                "num_sectors": 42,
                "num_factors": 2,
            },
            "production": {"type": "nested_ces"},
            "households": {
                "type": "representative",
                "demand_system": "LES",
                "frisch_parameter": -1.5,
            },
            "government": {
                "instruments": ["consumption_tax", "corporate_income_tax", "production_tax",
                               "interest_rate", "tfp", "targeted_investment"]
            },
            "trade": {"type": "small_open_economy"},
            "expectations": {"type": "perfect_foresight", "horizon": 12},  # 12 quarters
            "macro_closure": "keynesian",
            "extensions": [],
        }

    def _register_builtin_modules(self):
        """注册所有内置模块类。"""
        for mod_name in [
            "production", "household", "government", "trade",
            "market_clearing", "macro_closures",
            "energy", "carbon_tax", "labor_segmentation",
            "financial_accelerator", "open_economy_multi",
        ]:
            try:
                mod = __import__(f"modules.{mod_name}", fromlist=[mod_name])
                classes = [c for c in dir(mod) if c.endswith("Module") or c.endswith("Extension")]
                for cls_name in classes:
                    cls = getattr(mod, cls_name)
                    if isinstance(cls, type):
                        self.registry.register_class(mod_name, cls)
                        break
            except (ImportError, IndexError, AttributeError):
                pass

    def load_sam(self, sam: SAMData = None, data_dir: str = None,
                 seed: int = 42) -> SAMData:
        """加载或生成SAM数据。"""
        if sam:
            self.sam = sam
        elif data_dir:
            sam_csv = Path(data_dir) / "sam_final_demand.csv"
            if sam_csv.exists():
                self.sam = self._load_sam_from_csv(data_dir)
            else:
                from data.data_pipeline import build_synthetic_sam
                self.sam = build_synthetic_sam(output_dir=data_dir, seed=seed)
        else:
            from data.data_pipeline import build_synthetic_sam
            data_dir = str(Path(__file__).parent.parent / "data" / "processed")
            self.sam = build_synthetic_sam(output_dir=data_dir, seed=seed)
        return self.sam

    def _load_sam_from_csv(self, data_dir: str) -> SAMData:
        """从CSV加载SAM。"""
        import pandas as pd
        d = Path(data_dir)

        df_final = pd.read_csv(d / "sam_final_demand.csv", encoding="utf-8-sig")
        df_inter = pd.read_csv(d / "sam_intermediate.csv", encoding="utf-8-sig", index_col=0)
        df_elast = pd.read_csv(d / "sam_elasticities.csv", encoding="utf-8-sig")

        sam = SAMData()
        sam.num_sectors = len(df_final)
        sam.sector_codes = df_final["部门代码"].tolist()
        sam.sector_names = df_final["部门名称"].tolist()
        sam.intermediate = df_inter.values
        sam.household_cons = df_final["居民消费"].values
        sam.government_cons = df_final["政府消费"].values
        sam.investment = df_final["固定资本形成"].values
        sam.exports = df_final["出口"].values
        sam.imports = df_final["进口"].values
        sam.labor_comp = df_final["劳动报酬"].values
        sam.capital_income = df_final["资本回报"].values
        sam.depreciation = df_final["折旧"].values
        sam.production_tax = df_final["生产税"].values
        sam.armington_sigma = df_elast["Armington弹性"].values
        sam.ces_va_sigma = df_elast["CES_VA弹性"].values
        sam.cet_sigma = df_elast["CET弹性"].values
        sam.frisch = -1.5
        sam.gdp = (df_final["劳动报酬"].sum() +
                   df_final["资本回报"].sum() +
                   df_final["生产税"].sum())
        sam.tax_rates = {
            "consumption_tax": 0.13,
            "corporate_income_tax": 0.25,
            "production_tax_rate": (df_final["生产税"] / df_final["总产出"]).fillna(0).values,
        }
        # v2.0: 财政与金融数据
        sam.fiscal_revenue = 216000.0   # 2025年一般公共预算收入(亿元)
        sam.fiscal_expenditure = 287300.0
        sam.fiscal_deficit = 71300.0
        sam.lpr_1y = 3.0
        sam.cpi_baseline = 0.0
        sam.exchange_rate = 7.1429
        return sam

    def calibrate(self) -> Calibrator:
        """校准参数。"""
        if self.sam is None:
            raise RuntimeError("必须先调用load_sam()")
        self.calibrator = Calibrator(self.sam)
        return self.calibrator

    def build_solver(self) -> JohansenSolver:
        """构建Johansen求解器。"""
        if self.calibrator is None:
            self.calibrate()

        closure = self.config.get("macro_closure", "keynesian")
        self.solver = JohansenSolver(self.sam, closure=closure)

        # 动态求解器
        horizon = self.config.get("expectations", {}).get("horizon", 12)
        self.dyn_solver = DynamicJohansenSolver(self.sam, horizon=horizon)

        return self.solver

    def solve_shock(self, shock_params: Dict[str, Any]) -> SolverResult:
        """求解政策冲击。

        Args:
            shock_params: 政策参数, 支持:
              - consumption_tax_change: 消费税率变化 (百分点)
              - gov_spending_change: 政府支出%变化
              - corporate_tax_change: 企业所得税率变化
              - production_tax_change: 生产税率变化
              - labor_supply_change: 劳动供给%变化
              - capital_supply_change: 资本供给%变化

        Returns:
            SolverResult
        """
        if self.solver is None:
            self.build_solver()
        return self.solver_manager.solve(self.solver, shock_params)

    def solve_dynamic(self, shock_params: Dict[str, Any],
                      shock_timing: str = "permanent",
                      implementation_lag: int = 0) -> Dict[str, Any]:
        """求解12期动态路径。

        Args:
            shock_timing: "permanent" / "temporary" / "anticipated"
            implementation_lag: 政策实施滞后期(月)

        Returns:
            动态路径结果字典
        """
        if self.dyn_solver is None:
            self.build_solver()
        return self.solver_manager.solve_dynamic(
            self.dyn_solver, shock_params,
            shock_timing=shock_timing,
            implementation_lag=implementation_lag,
        )

    def quick_build_and_solve(self, scenario_params: Dict = None) -> tuple:
        """快速构建并求解(便捷方法)。

        Returns:
            (self, SolverResult)
        """
        if self.sam is None:
            self.load_sam()
        if self.solver is None:
            self.build_solver()
        result = self.solve_shock(scenario_params or {})
        return self, result

    def get_sam_summary(self) -> Dict[str, Any]:
        """返回SAM摘要统计。"""
        if self.sam is None:
            self.load_sam()
        sam = self.sam
        return {
            "gdp": sam.gdp,
            "num_sectors": sam.num_sectors,
            "household_consumption": float(sam.household_cons.sum()),
            "government_consumption": float(sam.government_cons.sum()),
            "investment": float(sam.investment.sum()),
            "exports": float(sam.exports.sum()),
            "imports": float(sam.imports.sum()),
            "labor_compensation": float(sam.labor_comp.sum()),
            "capital_income": float(sam.capital_income.sum()),
            "production_tax": float(sam.production_tax.sum()),
            "trade_balance": float(sam.exports.sum() - sam.imports.sum()),
        }

    def list_available_modules(self) -> Dict[str, Dict]:
        """列出所有可用模块。"""
        return self.registry.list_available()
