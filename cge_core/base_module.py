# -*- coding: utf-8 -*-
"""
CGE模块抽象基类
所有经济特性（税收工具、生产函数、市场出清条件、预期机制）必须封装为
符合此接口的可插拔Python类。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, List

if TYPE_CHECKING:
    import pyomo.environ as pyo


class CGEModule(ABC):
    """所有CGE扩展模块的基类。

    子类必须实现四个核心方法：
      - declare_variables: 声明Pyomo变量
      - declare_parameters: 声明校准参数
      - declare_equations: 添加约束和均衡条件
      - calibrate: 从SAM数据校准参数

    可选实现：
      - scenario_params: 返回仪表盘暴露的参数及其边界
      - register_callbacks: 注册模型求解前/后的回调
    """

    # 模块元信息
    name: str = "unnamed_module"
    description: str = ""
    version: str = "1.0.0"
    # 依赖的其他模块名称列表
    dependencies: List[str] = []
    # 是否为核心模块（核心模块不可移除）
    is_core: bool = False

    def __init__(self, config: Dict[str, Any] = None):
        """初始化模块。

        Args:
            config: 从YAML配置文件读取的模块特定配置字典
        """
        self.config = config or {}
        self._active = True

    @abstractmethod
    def declare_variables(self, model: "pyo.ConcreteModel") -> None:
        """声明此模块需要的Pyomo变量。

        Args:
            model: Pyomo ConcreteModel实例
        """
        ...

    @abstractmethod
    def declare_parameters(self, model: "pyo.ConcreteModel") -> None:
        """声明校准参数（从SAM/配置读取）。

        Args:
            model: Pyomo ConcreteModel实例
        """
        ...

    @abstractmethod
    def declare_equations(self, model: "pyo.ConcreteModel") -> None:
        """添加约束条件和均衡条件到模型。

        Args:
            model: Pyomo ConcreteModel实例
        """
        ...

    @abstractmethod
    def calibrate(self, model: "pyo.ConcreteModel", sam: "SAMData") -> None:
        """从SAM数据校准参数。

        Args:
            model: Pyomo ConcreteModel实例
            sam: 社会核算矩阵数据对象
        """
        ...

    def scenario_params(self) -> Dict[str, Dict[str, Any]]:
        """返回仪表盘暴露的参数及其边界。

        Returns:
            字典: {
                "param_name": {
                    "default": 默认值,
                    "min": 最小值,
                    "max": 最大值,
                    "step": 步长,
                    "label": "中文标签",
                    "type": "slider" | "number" | "select"
                }
            }
        """
        return {}

    def pre_solve_hook(self, model: "pyo.ConcreteModel") -> None:
        """求解前的回调钩子（可选）。"""
        pass

    def post_solve_hook(self, model: "pyo.ConcreteModel", results: Any) -> None:
        """求解后的回调钩子（可选）。"""
        pass

    @property
    def active(self) -> bool:
        return self._active

    def activate(self):
        self._active = True

    def deactivate(self):
        if self.is_core:
            raise ValueError(f"核心模块 '{self.name}' 不可移除")
        self._active = False

    def __repr__(self):
        return f"<CGEModule: {self.name} v{self.version}>"


class SAMData:
    """社会核算矩阵数据容器。

    封装42部门SAM的所有数据，供模块校准使用。
    """

    def __init__(self):
        # 基础维度
        self.num_sectors: int = 42
        self.sector_names: List[str] = []
        self.sector_codes: List[str] = []

        # 核心矩阵 (亿元)
        self.intermediate: Any = None        # 中间投入矩阵 (42×42)
        self.household_cons: Any = None       # 居民消费向量 (42,)
        self.government_cons: Any = None      # 政府消费向量 (42,)
        self.investment: Any = None           # 固定资本形成向量 (42,)
        self.exports: Any = None              # 出口向量 (42,)
        self.imports: Any = None              # 进口向量 (42,)

        # 增加值
        self.labor_comp: Any = None           # 劳动报酬向量 (42,)
        self.capital_income: Any = None       # 资本回报向量 (42,)
        self.depreciation: Any = None         # 折旧向量 (42,)
        self.production_tax: Any = None       # 生产税向量 (42,)

        # 弹性参数
        self.armington_sigma: Any = None      # Armington弹性
        self.ces_va_sigma: Any = None         # 增加值CES替代弹性
        self.cet_sigma: Any = None            # CET转换弹性
        self.frisch: float = -1.5             # Frisch参数

        # 宏观总量
        self.gdp: float = 0.0
        self.total_household_income: float = 0.0
        self.total_gov_revenue: float = 0.0
        self.total_investment: float = 0.0

        # 税率
        self.tax_rates: Dict[str, Any] = {}

    @property
    def total_output(self):
        """各部门总产出。"""
        import numpy as np
        if self.intermediate is not None:
            intermediate_sum = self.intermediate.sum(axis=0)
            final_demand = (
                (self.household_cons if self.household_cons is not None else 0)
                + (self.government_cons if self.government_cons is not None else 0)
                + (self.investment if self.investment is not None else 0)
                + (self.exports if self.exports is not None else 0)
                - (self.imports if self.imports is not None else 0)
            )
            return intermediate_sum + final_demand
        return None

    def summary(self) -> str:
        """返回SAM摘要信息。"""
        lines = [
            f"SAM数据摘要 ({self.num_sectors}部门)",
            f"  GDP: {self.gdp:,.0f} 亿元",
            f"  Frisch参数: {self.frisch}",
        ]
        return "\n".join(lines)


class ModuleRegistry:
    """模块注册表 — 管理所有已注册的CGE模块。"""

    def __init__(self):
        self._modules: Dict[str, CGEModule] = {}
        self._module_classes: Dict[str, type] = {}

    def register_class(self, name: str, cls: type) -> None:
        """注册一个模块类（不实例化）。"""
        self._module_classes[name] = cls

    def create_module(self, name: str, config: Dict = None) -> CGEModule:
        """创建并注册一个模块实例。"""
        if name not in self._module_classes:
            raise KeyError(f"未注册的模块类型: '{name}'。已注册: {list(self._module_classes.keys())}")
        cls = self._module_classes[name]
        instance = cls(config=config)
        self._modules[name] = instance
        return instance

    def get_module(self, name: str) -> CGEModule:
        return self._modules[name]

    def get_active_modules(self) -> List[CGEModule]:
        """按依赖顺序返回所有活跃模块。"""
        active = [m for m in self._modules.values() if m.active]
        # 拓扑排序：确保依赖在前
        sorted_modules = []
        visited = set()

        def visit(mod):
            if mod.name in visited:
                return
            visited.add(mod.name)
            for dep_name in mod.dependencies:
                if dep_name in self._modules:
                    visit(self._modules[dep_name])
            sorted_modules.append(mod)

        for mod in active:
            visit(mod)

        return sorted_modules

    def list_available(self) -> Dict[str, Dict]:
        """列出所有可用模块类（含元信息）。"""
        result = {}
        for name, cls in self._module_classes.items():
            result[name] = {
                "name": name,
                "description": getattr(cls, "description", ""),
                "version": getattr(cls, "version", ""),
                "is_core": getattr(cls, "is_core", False),
                "dependencies": getattr(cls, "dependencies", []),
                "active": name in self._modules and self._modules[name].active,
            }
        return result

    def has_module(self, name: str) -> bool:
        return name in self._modules and self._modules[name].active
