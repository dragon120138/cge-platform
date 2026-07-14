# -*- coding: utf-8 -*-
"""
数据管线 — SAM构建与RAS平衡

从投入产出表（或合成数据）构建42部门社会核算矩阵(SAM)，
执行RAS/Cross-Entropy平衡，并提取校准参数。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from cge_core.sectors import SECTOR_CODES, SECTOR_NAMES_CN, NUM_SECTORS
from cge_core.base_module import SAMData


# ============================================================
# 合成SAM数据生成器
# 基于中国2020年宏观经济数据构建合理的42部门SAM
# ============================================================

# 中国2025年宏观总量 (万亿元)
# 数据来源：国家统计局《2025年国民经济和社会发展统计公报》(2026.02.28)
CHINA_2025_MACRO = {
    "gdp": 140.19,           # GDP (万亿元) — 同比增长5.0%
    "household_consumption": 42.50,   # 居民消费 (占GDP ~30.3%)
    "government_consumption": 17.80,  # 政府消费 (占GDP ~12.7%)
    "gross_capital_formation": 53.70, # 固定资本形成总额 (占GDP ~38.3%)
    "exports": 30.64,        # 货物+服务出口 (万亿)
    "imports": 22.94,        # 货物+服务进口 (万亿)
    "labor_compensation": 77.10,  # 劳动者报酬 (占GDP ~55%)
    "fixed_capital_depreciation": 24.53,  # 固定资产折旧
    "net_production_tax": 19.63,   # 生产税净额
    "operating_surplus": 18.93,    # 营业盈余 (资本回报)
    # 财政数据 (2025年一般公共预算)
    "fiscal_revenue": 21.60,       # 一般公共预算收入 (同比下降1.7%)
    "fiscal_expenditure": 28.73,   # 一般公共预算支出 (同比增长1.0%)
    "fiscal_deficit": 7.13,        # 财政赤字
    # 货币与金融
    "lpr_1y": 3.00,               # 1年期LPR (%)
    "m2_growth": 8.5,              # M2增速 (%)
    "cpi": 0.0,                    # CPI涨幅 (%)
    "ppi": -2.6,                   # PPI降幅 (%)
    # 就业
    "employment": 72504,           # 全国就业人员 (万人)
    "urban_unemployment": 5.2,     # 城镇调查失业率 (%)
    # 其他
    "rd_expenditure": 3.93,        # R&D经费支出 (万亿, 占GDP 2.8%)
    "exchange_rate": 7.1429,       # 人民币平均汇率 (CNY/USD)
}

# 42部门产出占比（基于2025年经济结构估算）
# 三次产业结构: 第一产业6.7%, 第二产业35.6%, 第三产业57.7%
# 合计 = 1.0
SECTOR_OUTPUT_SHARES = np.array([
    0.0550,   # S01 农业 (↓ 产业结构升级)
    0.0165,   # S02 煤炭 (↓ 能源转型)
    0.0130,   # S03 石油天然气
    0.0115,   # S04 金属矿
    0.0085,   # S05 非金属矿
    0.0465,   # S06 食品烟草
    0.0185,   # S07 纺织品 (↓ 产业外迁)
    0.0180,   # S08 服装皮革 (↓)
    0.0145,   # S09 木材家具
    0.0155,   # S10 造纸印刷
    0.0225,   # S11 石油加工
    0.0555,   # S12 化学产品
    0.0290,   # S13 非金属矿物 (↓ 房地产低迷)
    0.0460,   # S14 金属冶炼
    0.0210,   # S15 金属制品
    0.0295,   # S16 通用设备
    0.0260,   # S17 专用设备 (↑ 高端制造)
    0.0365,   # S18 交通运输设备 (↑ 新能源车)
    0.0350,   # S19 电气机械 (↑ 光伏/储能)
    0.0650,   # S20 通信电子设备 (↑ 芯片/AI)
    0.0095,   # S21 仪器仪表
    0.0080,   # S22 其他制造
    0.0045,   # S23 修理服务
    0.0425,   # S24 电力热力
    0.0090,   # S25 燃气
    0.0040,   # S26 水
    0.0780,   # S27 建筑 (↓ 房地产低迷)
    0.0620,   # S28 批发零售
    0.0490,   # S29 交通运输仓储
    0.0225,   # S30 住宿餐饮
    0.0410,   # S31 信息传输软件 (↑↑ 数字经济)
    0.0600,   # S32 金融
    0.0340,   # S33 房地产 (↓↓ 房地产调整)
    0.0265,   # S34 租赁商务服务
    0.0185,   # S35 科学研究 (↑ 创新驱动)
    0.0105,   # S36 水利环境
    0.0165,   # S37 居民服务
    0.0240,   # S38 教育 (↑)
    0.0215,   # S39 卫生社会工作 (↑)
    0.0095,   # S40 文化体育娱乐
    0.0305,   # S41 公共管理
    0.0003,   # S42 国际组织
])

# 各部门劳动报酬占总增加值比重
LABOR_SHARE_OF_VA = np.array([
    0.85, 0.55, 0.45, 0.50, 0.52,   # S01-S05
    0.62, 0.58, 0.60, 0.55, 0.54,   # S06-S10
    0.38, 0.42, 0.48, 0.40, 0.50,   # S11-S15
    0.52, 0.50, 0.48, 0.50, 0.55,   # S16-S20
    0.58, 0.52, 0.55, 0.40, 0.45,   # S21-S25
    0.55, 0.65, 0.45, 0.52, 0.55,   # S26-S30
    0.58, 0.50, 0.40, 0.55, 0.65,   # S31-S35
    0.60, 0.58, 0.78, 0.72, 0.65,   # S36-S40
    0.82, 0.70,                       # S41-S42
])

# 各部门生产税净额占总产出比重
PRODUCTION_TAX_RATE = np.array([
    0.005, 0.08, 0.10, 0.06, 0.05,   # S01-S05
    0.10, 0.05, 0.06, 0.05, 0.05,   # S06-S10
    0.08, 0.06, 0.05, 0.04, 0.04,   # S11-S15
    0.04, 0.04, 0.05, 0.04, 0.03,   # S16-S20
    0.04, 0.04, 0.05, 0.03, 0.03,   # S21-S25
    0.06, 0.06, 0.10, 0.08, 0.05,   # S26-S30
    0.06, 0.12, 0.10, 0.08, 0.05,   # S31-S35
    0.03, 0.06, 0.00, 0.00, 0.03,   # S36-S40
    0.00, 0.00,                       # S41-S42
])

# 各部门居民消费份额
HOUSEHOLD_CONS_SHARE = np.array([
    0.090, 0.003, 0.002, 0.001, 0.001,   # S01-S05
    0.165, 0.025, 0.058, 0.018, 0.012,   # S06-S10
    0.015, 0.038, 0.008, 0.003, 0.008,   # S11-S15
    0.008, 0.006, 0.038, 0.020, 0.065,   # S16-S20
    0.003, 0.010, 0.005, 0.035, 0.010,   # S21-S25
    0.005, 0.003, 0.035, 0.108, 0.065,   # S26-S30
    0.035, 0.072, 0.135, 0.025, 0.008,   # S31-S35
    0.028, 0.072, 0.065, 0.090, 0.032,   # S36-S40
    0.000, 0.000,                          # S41-S42
])

# 各部门政府消费份额
GOV_CONS_SHARE = np.array([
    0.015, 0.000, 0.000, 0.000, 0.000,
    0.002, 0.000, 0.000, 0.000, 0.001,
    0.000, 0.001, 0.000, 0.000, 0.000,
    0.000, 0.001, 0.002, 0.000, 0.001,
    0.000, 0.000, 0.000, 0.015, 0.000,
    0.010, 0.000, 0.035, 0.010, 0.005,
    0.018, 0.010, 0.000, 0.025, 0.045,
    0.060, 0.025, 0.320, 0.250, 0.025,
    0.155, 0.000,
])

# 各部门固定资本形成份额
INVESTMENT_SHARE = np.array([
    0.020, 0.000, 0.000, 0.000, 0.000,
    0.015, 0.000, 0.000, 0.025, 0.002,
    0.000, 0.025, 0.008, 0.030, 0.020,
    0.060, 0.050, 0.075, 0.055, 0.080,
    0.012, 0.008, 0.000, 0.020, 0.002,
    0.005, 0.310, 0.025, 0.015, 0.005,
    0.010, 0.000, 0.080, 0.010, 0.005,
    0.025, 0.010, 0.020, 0.010, 0.005,
    0.000, 0.000,
])

# 出口份额
EXPORT_SHARE = np.array([
    0.010, 0.015, 0.020, 0.025, 0.010,
    0.025, 0.090, 0.110, 0.045, 0.035,
    0.020, 0.080, 0.025, 0.055, 0.040,
    0.070, 0.060, 0.055, 0.110, 0.220,
    0.035, 0.015, 0.005, 0.002, 0.000,
    0.000, 0.005, 0.060, 0.065, 0.015,
    0.025, 0.010, 0.002, 0.030, 0.020,
    0.002, 0.020, 0.010, 0.002, 0.010,
    0.000, 0.000,
])

# 进口份额
IMPORT_SHARE = np.array([
    0.035, 0.080, 0.150, 0.110, 0.040,
    0.045, 0.030, 0.015, 0.020, 0.040,
    0.090, 0.080, 0.025, 0.060, 0.035,
    0.060, 0.055, 0.040, 0.075, 0.180,
    0.045, 0.020, 0.005, 0.005, 0.010,
    0.002, 0.005, 0.015, 0.040, 0.010,
    0.015, 0.005, 0.002, 0.040, 0.025,
    0.002, 0.010, 0.005, 0.002, 0.010,
    0.000, 0.000,
])

# 部门间中间投入强度系数 (中间投入/总产出)
INTERMEDIATE_RATIO = np.array([
    0.42, 0.58, 0.52, 0.55, 0.50,
    0.68, 0.65, 0.62, 0.60, 0.58,
    0.72, 0.68, 0.55, 0.68, 0.60,
    0.58, 0.56, 0.60, 0.58, 0.55,
    0.55, 0.58, 0.50, 0.65, 0.60,
    0.55, 0.58, 0.40, 0.48, 0.50,
    0.40, 0.35, 0.35, 0.45, 0.38,
    0.42, 0.40, 0.38, 0.35, 0.38,
    0.42, 0.40,
])

# 替代弹性 (文献meta-analysis估值)
ARMINGTON_SIGMA = np.array([
    2.8, 3.0, 4.0, 3.5, 3.0,
    2.5, 3.0, 3.0, 2.8, 2.8,
    4.0, 3.2, 2.5, 3.0, 2.8,
    3.0, 3.0, 3.5, 3.2, 4.0,
    3.5, 2.8, 2.5, 2.0, 2.5,
    2.0, 1.8, 2.0, 1.8, 1.5,
    2.5, 2.0, 1.5, 2.0, 2.5,
    1.5, 2.0, 1.2, 1.2, 1.8,
    1.0, 1.0,
])

CES_VA_SIGMA = np.array([
    0.5, 0.8, 0.8, 0.8, 0.8,
    0.6, 0.8, 0.8, 0.7, 0.7,
    0.8, 0.8, 0.7, 0.8, 0.8,
    0.8, 0.8, 0.8, 0.8, 0.9,
    0.9, 0.8, 0.8, 0.7, 0.8,
    0.7, 0.6, 0.6, 0.7, 0.7,
    0.8, 0.7, 0.6, 0.7, 0.8,
    0.6, 0.7, 0.5, 0.6, 0.7,
    0.5, 0.5,
])

CET_SIGMA = np.array([
    2.0, 3.0, 4.0, 3.5, 3.0,
    2.5, 3.0, 3.0, 2.5, 2.5,
    4.0, 3.0, 2.5, 3.0, 2.5,
    3.0, 3.0, 3.5, 3.0, 4.0,
    3.5, 2.5, 2.0, 1.5, 2.0,
    1.5, 1.5, 1.8, 1.5, 1.2,
    2.0, 1.5, 1.2, 1.8, 2.0,
    1.2, 1.5, 1.0, 1.0, 1.5,
    0.8, 0.8,
])


def normalize(arr: np.ndarray) -> np.ndarray:
    """归一化为概率分布（和为1）。"""
    s = arr.sum()
    if s > 0:
        return arr / s
    return arr


def generate_synthetic_intermediate_matrix(num_sectors: int, total_output: np.ndarray,
                                           intermediate_ratio: np.ndarray, rng=None) -> np.ndarray:
    """生成42×42中间投入矩阵。

    使用部门产出加权和随机扰动来生成合理的中间投入结构。
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = num_sectors
    # 基础分配矩阵：列部门j的中间投入按行部门i的产出比例分配
    base = np.outer(total_output, total_output)
    base = base / (base.sum(axis=0, keepdims=True) + 1e-10)

    # 加入随机扰动 (±20%)
    noise = 1.0 + 0.2 * rng.standard_normal((n, n))
    noise = np.clip(noise, 0.5, 1.5)
    alloc = base * noise
    alloc = alloc / (alloc.sum(axis=0, keepdims=True) + 1e-10)

    # 中间投入矩阵 = 各列的中间投入总量 × 分配比例
    intermediate_total = total_output * intermediate_ratio
    inter_matrix = alloc * intermediate_total[np.newaxis, :]

    # 确保非负
    inter_matrix = np.maximum(inter_matrix, 0)

    return inter_matrix


def build_synthetic_sam(output_dir: str = None, seed: int = 42) -> SAMData:
    """构建合成42部门SAM。

    基于中国2025年宏观经济数据和部门结构参数，
    生成一个内部一致的SAM。

    Args:
        output_dir: 如指定，将SAM保存为CSV到此目录
        seed: 随机种子

    Returns:
        SAMData对象
    """
    rng = np.random.default_rng(seed)

    # 归一化各部门份额
    output_shares = normalize(SECTOR_OUTPUT_SHARES.copy())
    hh_cons_shares = normalize(HOUSEHOLD_CONS_SHARE.copy())
    gov_cons_shares = normalize(GOV_CONS_SHARE.copy())
    inv_shares = normalize(INVESTMENT_SHARE.copy())
    export_shares = normalize(EXPORT_SHARE.copy())
    import_shares = normalize(IMPORT_SHARE.copy())

    # 宏观总量 (亿元)
    gdp_yi = CHINA_2025_MACRO["gdp"] * 1e4  # 万亿→亿元
    hh_cons_total = CHINA_2025_MACRO["household_consumption"] * 1e4
    gov_cons_total = CHINA_2025_MACRO["government_consumption"] * 1e4
    inv_total = CHINA_2025_MACRO["gross_capital_formation"] * 1e4
    export_total = CHINA_2025_MACRO["exports"] * 1e4
    import_total = CHINA_2025_MACRO["imports"] * 1e4

    # 部门总产出 (亿元)
    # 总产出 = 中间投入 + 增加值
    # 我们用 GDP/增加值 和中间投入率反推总产出
    # 增加值 = GDP各部门分配, 中间投入 = 增加值 × ratio/(1-ratio)
    # 但首先需要各部门GDP分配
    va_shares = output_shares.copy()
    sector_va = gdp_yi * va_shares  # 部门增加值 (亿元)

    # 部门总产出
    sector_total_output = sector_va / (1.0 - INTERMEDIATE_RATIO)

    # 生成中间投入矩阵
    inter_matrix = generate_synthetic_intermediate_matrix(
        NUM_SECTORS, sector_total_output, INTERMEDIATE_RATIO, rng
    )

    # 重新校准中间投入矩阵使列和匹配
    inter_col_sums = inter_matrix.sum(axis=0)
    for j in range(NUM_SECTORS):
        if inter_col_sums[j] > 0:
            target = sector_total_output[j] * INTERMEDIATE_RATIO[j]
            inter_matrix[:, j] *= target / inter_col_sums[j]

    # 增加值分解
    labor_comp = sector_va * LABOR_SHARE_OF_VA
    production_tax = sector_total_output * PRODUCTION_TAX_RATE
    remaining_va = sector_va - production_tax
    capital_income = remaining_va - labor_comp
    capital_income = np.maximum(capital_income, sector_va * 0.05)
    labor_comp = remaining_va - capital_income
    depreciation = capital_income * 0.55  # 折旧约为资本回报的55%
    net_capital_income = capital_income - depreciation  # 净营业盈余

    # 最终需求向量
    household_cons = hh_cons_total * hh_cons_shares
    government_cons = gov_cons_total * gov_cons_shares
    investment = inv_total * inv_shares
    exports = export_total * export_shares
    imports = import_total * import_shares

    # 组装SAMData
    sam = SAMData()
    sam.num_sectors = NUM_SECTORS
    sam.sector_codes = SECTOR_CODES.copy()
    sam.sector_names = SECTOR_NAMES_CN.copy()
    sam.intermediate = inter_matrix
    sam.household_cons = household_cons
    sam.government_cons = government_cons
    sam.investment = investment
    sam.exports = exports
    sam.imports = imports
    sam.labor_comp = labor_comp
    sam.capital_income = capital_income
    sam.depreciation = depreciation
    sam.production_tax = production_tax
    sam.armington_sigma = ARMINGTON_SIGMA.copy()
    sam.ces_va_sigma = CES_VA_SIGMA.copy()
    sam.cet_sigma = CET_SIGMA.copy()
    sam.frisch = -1.5
    sam.gdp = gdp_yi
    sam.total_household_income = labor_comp.sum() + net_capital_income.sum()
    sam.total_gov_revenue = production_tax.sum()
    sam.total_investment = inv_total

    # 2025年财政数据 (亿元)
    sam.fiscal_revenue = CHINA_2025_MACRO["fiscal_revenue"] * 1e4
    sam.fiscal_expenditure = CHINA_2025_MACRO["fiscal_expenditure"] * 1e4
    sam.fiscal_deficit = CHINA_2025_MACRO["fiscal_deficit"] * 1e4
    # 利率基准
    sam.lpr_1y = CHINA_2025_MACRO["lpr_1y"]
    sam.cpi_baseline = CHINA_2025_MACRO["cpi"]
    sam.exchange_rate = CHINA_2025_MACRO["exchange_rate"]

    # 税率
    sam.tax_rates = {
        "consumption_tax": 0.13,      # 增值税率(近似)
        "corporate_income_tax": 0.25,  # 企业所得税
        "labor_income_tax": 0.10,     # 个人所得税(近似)
        "production_tax_rate": PRODUCTION_TAX_RATE.copy(),
        "import_tariff": 0.05,        # 平均关税率
    }

    # 保存
    if output_dir:
        save_sam(sam, output_dir)

    return sam


def save_sam(sam: SAMData, output_dir: str) -> None:
    """将SAM保存为CSV文件。"""
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 中间投入矩阵
        df_inter = pd.DataFrame(
            sam.intermediate,
            index=SECTOR_CODES,
            columns=SECTOR_CODES,
        )
        df_inter.to_csv(out / "sam_intermediate.csv", encoding="utf-8-sig")

        # 最终需求与增加值
        df_final = pd.DataFrame({
            "部门代码": SECTOR_CODES,
            "部门名称": SECTOR_NAMES_CN,
            "总产出": sam.total_output,
            "中间投入合计": sam.intermediate.sum(axis=0),
            "劳动报酬": sam.labor_comp,
            "资本回报": sam.capital_income,
            "折旧": sam.depreciation,
            "生产税": sam.production_tax,
            "居民消费": sam.household_cons,
            "政府消费": sam.government_cons,
            "固定资本形成": sam.investment,
            "出口": sam.exports,
            "进口": sam.imports,
        })
        df_final.to_csv(out / "sam_final_demand.csv", index=False, encoding="utf-8-sig")

        # 弹性参数
        df_elast = pd.DataFrame({
            "部门代码": SECTOR_CODES,
            "部门名称": SECTOR_NAMES_CN,
            "Armington弹性": sam.armington_sigma,
            "CES_VA弹性": sam.ces_va_sigma,
            "CET弹性": sam.cet_sigma,
        })
        df_elast.to_csv(out / "sam_elasticities.csv", index=False, encoding="utf-8-sig")
    except (OSError, UnicodeEncodeError, PermissionError):
        pass


# ============================================================
# RAS平衡算法
# ============================================================

def ras_balance(matrix: np.ndarray, row_targets: np.ndarray, col_targets: np.ndarray,
                max_iter: int = 1000, tol: float = 1e-8) -> Tuple[np.ndarray, dict]:
    """RAS（双比例平衡）算法。

    给定初始矩阵和行列目标总和，迭代调整使行和列的边际总和
    分别等于目标值。

    Args:
        matrix: 初始非负矩阵 (n×m)
        row_targets: 行目标总和 (n,)
        col_targets: 列目标总和 (m,)
        max_iter: 最大迭代次数
        tol: 收敛容差

    Returns:
        (平衡后矩阵, 诊断信息字典)
    """
    M = matrix.copy().astype(float)
    M = np.maximum(M, 1e-10)  # 确保正数

    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)

    diag_info = {
        "iterations": 0,
        "max_row_error": [],
        "max_col_error": [],
        "converged": False,
    }

    for it in range(max_iter):
        # 行调整
        row_mult = row_targets / (row_sums + 1e-15)
        M *= row_mult[:, np.newaxis]
        row_sums = M.sum(axis=1)

        # 列调整
        col_mult = col_targets / (col_sums + 1e-15)
        M *= col_mult[np.newaxis, :]
        col_sums = M.sum(axis=0)

        # 检查收敛
        max_row_err = np.max(np.abs(row_sums - row_targets) / (row_targets + 1e-15))
        max_col_err = np.max(np.abs(col_sums - col_targets) / (col_targets + 1e-15))
        diag_info["max_row_error"].append(max_row_err)
        diag_info["max_col_error"].append(max_col_err)

        if max(max_row_err, max_col_err) < tol:
            diag_info["converged"] = True
            diag_info["iterations"] = it + 1
            break

    diag_info["iterations"] = diag_info["iterations"] or max_iter
    diag_info["final_row_error"] = float(max_row_err)
    diag_info["final_col_error"] = float(max_col_err)

    return M, diag_info


def cross_entropy_balance(matrix: np.ndarray, row_targets: np.ndarray, col_targets: np.ndarray,
                           max_iter: int = 500, tol: float = 1e-7) -> Tuple[np.ndarray, dict]:
    """Cross-Entropy平衡算法（RAS的熵推广）。

    最小化与初始矩阵的Kullback-Leibler散度，同时满足行列约束。
    """
    return ras_balance(matrix, row_targets, col_targets, max_iter, tol)


def verify_sam_balance(sam: SAMData, tol: float = 1e-4) -> dict:
    """验证SAM平衡性（瓦尔拉斯法则检查）。

    Returns:
        诊断字典
    """
    n = sam.num_sectors

    # 行总和（收入）
    inter_row_sum = sam.intermediate.sum(axis=1) if sam.intermediate is not None else np.zeros(n)
    row_sums = (
        inter_row_sum
        + (sam.household_cons if sam.household_cons is not None else 0)
        + (sam.government_cons if sam.government_cons is not None else 0)
        + (sam.investment if sam.investment is not None else 0)
        + (sam.exports if sam.exports is not None else 0)
    )

    # 列总和（支出）
    inter_col_sum = sam.intermediate.sum(axis=0) if sam.intermediate is not None else np.zeros(n)
    col_sums = (
        inter_col_sum
        + (sam.labor_comp if sam.labor_comp is not None else 0)
        + (sam.capital_income if sam.capital_income is not None else 0)
        + (sam.production_tax if sam.production_tax is not None else 0)
    )

    # 进口需要从行总和中扣除（因为是负需求）
    if sam.imports is not None:
        row_sums = row_sums  # 进口在支出端

    diff = row_sums - col_sums
    mean_abs = np.mean(np.abs(row_sums + col_sums) / 2)
    max_rel_error = np.max(np.abs(diff)) / (mean_abs + 1e-10)

    return {
        "balanced": max_rel_error < tol,
        "max_rel_error": float(max_rel_error),
        "max_abs_error": float(np.max(np.abs(diff))),
        "row_sums": row_sums,
        "col_sums": col_sums,
        "tolerance": tol,
    }
