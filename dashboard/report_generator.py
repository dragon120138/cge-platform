# -*- coding: utf-8 -*-
"""
AI报告生成器 — 双专家模式
  persona="macro"    → 审慎中立的财政部宏观专家
  persona="industry" → 产业政策专家，聚焦42部门细分影响
"""

import os, json, time, numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

HERMES_DIR = Path(os.environ.get("HOME", "C:/Users/Administrator")) / "AppData" / "Local" / "hermes"

def _load_api_config(user_api_key: str = "") -> Dict[str, str]:
    """加载API配置。优先使用用户传入的key，其次读.env，最后读环境变量。"""
    config = {"api_key": "", "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash"}

    # 1. 用户直接传入的key（最高优先级）
    if user_api_key:
        config["api_key"] = user_api_key.strip()

    # 2. 从.env文件读取
    if not config["api_key"]:
        env_path = HERMES_DIR / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                if key.strip() == "GLM_API_KEY" and val.strip():
                    config["api_key"] = val.strip().strip('"').strip("'")

    # 3. 环境变量
    if not config["api_key"]:
        config["api_key"] = os.environ.get("GLM_API_KEY", "")

    # 4. 从Hermes profiles配置读取
    if not config["api_key"]:
        try:
            import json
            settings_path = HERMES_DIR / "profiles" / "fuben" / "settings.json"
            if settings_path.exists():
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
                custom_providers = settings.get("custom_providers", {})
                for name, prov in custom_providers.items():
                    if prov.get("api_key"):
                        config["api_key"] = prov["api_key"]
                        if prov.get("base_url"):
                            config["base_url"] = prov["base_url"]
                        if prov.get("model"):
                            config["model"] = prov["model"]
                        break
        except Exception:
            pass

    return config


# =====================================================================
#  数据收集
# =====================================================================
def _collect_report_data(
    static_result, dyn_result, shock_params, sam_summary,
    sector_names, closure, shock_timing, policy_timings=None
) -> Dict:
    policies = []
    if abs(shock_params.get("consumption_tax_change", 0)) > 1e-6:
        val = shock_params["consumption_tax_change"] * 100
        d = "提高" if val > 0 else "降低"
        timing = _timing_cn(policy_timings.get("consumption_tax", shock_timing)) if policy_timings else _timing_cn(shock_timing)
        policies.append(f"消费税{d}{abs(val):.1f}个百分点（{timing}）")
    if abs(shock_params.get("gov_spending_change", 0)) > 1e-6:
        val = shock_params["gov_spending_change"] * 100
        d = "增加" if val > 0 else "减少"
        gov_total = sam_summary.get("government_consumption", 0) / 1e4
        amount = gov_total * abs(val) / 100
        timing = _timing_cn(policy_timings.get("gov_spending", shock_timing)) if policy_timings else _timing_cn(shock_timing)
        policies.append(f"政府支出{d}{abs(val):.0f}%（约{amount:,.0f}亿元，{timing}）")
    if abs(shock_params.get("corporate_tax_change", 0)) > 1e-6:
        val = shock_params["corporate_tax_change"] * 100
        d = "提高" if val > 0 else "降低"
        timing = _timing_cn(policy_timings.get("corporate_tax", shock_timing)) if policy_timings else _timing_cn(shock_timing)
        policies.append(f"企业所得税{d}{abs(val):.1f}个百分点（{timing}）")
    if abs(shock_params.get("production_tax_change", 0)) > 1e-6:
        val = shock_params["production_tax_change"] * 100
        d = "提高" if val > 0 else "降低"
        timing = _timing_cn(policy_timings.get("production_tax", shock_timing)) if policy_timings else _timing_cn(shock_timing)
        policies.append(f"生产税{d}{abs(val):.1f}个百分点（{timing}）")
    # 利率政策
    if abs(shock_params.get("interest_rate_change", 0)) > 1e-6:
        val_bp = shock_params["interest_rate_change"] * 10000  # fractional pp → bp
        d = "加息" if val_bp > 0 else "降息"
        timing = _timing_cn(policy_timings.get("interest_rate", shock_timing)) if policy_timings else _timing_cn(shock_timing)
        policies.append(f"{d}{abs(val_bp):.0f}基点（{timing}）")
    # TFP冲击
    tfp_shock = shock_params.get("tfp_shock", 0)
    if isinstance(tfp_shock, dict) and tfp_shock:
        # 定向TFP
        items = []
        for idx, val in tfp_shock.items():
            d = "提升" if val > 0 else "下降"
            items.append(f"{sector_names[idx]}{d}{abs(val)*100:.1f}%")
        timing = _timing_cn(policy_timings.get("tfp", shock_timing)) if policy_timings else _timing_cn(shock_timing)
        policies.append(f"定向TFP调整（{'，'.join(items)}，{timing}）")
    elif isinstance(tfp_shock, (int, float)) and abs(tfp_shock) > 1e-6:
        d = "提升" if tfp_shock > 0 else "下降"
        timing = _timing_cn(policy_timings.get("tfp", shock_timing)) if policy_timings else _timing_cn(shock_timing)
        policies.append(f"全行业TFP{d}{abs(tfp_shock)*100:.1f}%（{timing}）")
    # 定向投资
    if "targeted_sector" in shock_params and abs(shock_params.get("targeted_investment", 0)) > 1e-6:
        idx = shock_params["targeted_sector"]
        val = shock_params["targeted_investment"] * 100
        d = "扩张" if val > 0 else "收缩"
        timing = _timing_cn(policy_timings.get("targeted", shock_timing)) if policy_timings else _timing_cn(shock_timing)
        policies.append(f"{sector_names[idx]}定向投资{d}{abs(val):.0f}%（{timing}）")

    policy_desc = "；".join(policies) if policies else "无政策变化（基准情形）"

    gdp_path = dyn_result["deviation"]["gdp"]
    max_gdp_q = int(np.argmax(np.abs(gdp_path)) + 1)
    max_gdp_val = float(gdp_path[max_gdp_q - 1] * 100)

    if sector_names is None:
        from cge_core.sectors import SECTOR_NAMES_CN
        sector_names = SECTOR_NAMES_CN[:42]

    sector_output_chg = static_result.pct_changes.get("qo", np.zeros(42)) * 100
    sector_price_chg = static_result.pct_changes.get("po", np.zeros(42)) * 100
    sector_cons_chg = static_result.pct_changes.get("qh", np.zeros(42)) * 100

    # 全部42部门的详细数据
    all_sectors = []
    for i in range(42):
        all_sectors.append({
            "code": f"S{i+1:02d}",
            "name": sector_names[i],
            "output_chg": round(float(sector_output_chg[i]), 4),
            "price_chg": round(float(sector_price_chg[i]), 4),
            "cons_chg": round(float(sector_cons_chg[i]), 4),
        })

    top_beneficiaries = np.argsort(sector_output_chg)[-10:][::-1]
    top_sufferers = np.argsort(sector_output_chg)[:10]

    beneficiaries_str = "、".join([
        f"{sector_names[i]}（{sector_output_chg[i]:+.2f}%）"
        for i in top_beneficiaries if sector_output_chg[i] > 0.001
    ]) or "无明显受益部门"

    sufferers_str = "、".join([
        f"{sector_names[i]}（{sector_output_chg[i]:+.2f}%）"
        for i in top_sufferers if sector_output_chg[i] < -0.001
    ]) or "无明显受损部门"

    closure_cn = {"keynesian": "凯恩斯模式（就业可变）", "neoclassical": "新古典模式（充分就业）"}

    # 财政平衡数据
    fiscal_path = dyn_result.get("fiscal_path", [])
    fiscal_baseline = sam_summary.get("fiscal_deficit", 0) / 1e4  # 万亿
    fiscal_end = float(fiscal_path[-1]) if len(fiscal_path) > 0 else 0.0
    fiscal_max_worsen = float(min(fiscal_path)) if len(fiscal_path) > 0 else 0.0
    fiscal_max_improve = float(max(fiscal_path)) if len(fiscal_path) > 0 else 0.0

    # 信心指数数据
    conf_path = dyn_result.get("confidence_path", {})
    conf_consumer_end = float(conf_path.get("consumer", [0])[-1]) if conf_path else 0.0
    conf_enterprise_end = float(conf_path.get("enterprise", [0])[-1]) if conf_path else 0.0
    conf_investor_end = float(conf_path.get("investor", [0])[-1]) if conf_path else 0.0
    conf_consumer_max = float(max(conf_path.get("consumer", [0]))) if conf_path else 0.0
    conf_enterprise_max = float(max(conf_path.get("enterprise", [0]))) if conf_path else 0.0
    conf_investor_max = float(max(conf_path.get("investor", [0]))) if conf_path else 0.0

    return {
        "policy_desc": policy_desc,
        "closure_cn": closure_cn.get(closure, closure),
        "gdp_chg": static_result.gdp_change * 100,
        "cpi_chg": static_result.cpi_change * 100,
        "emp_chg": static_result.employment_change * 100,
        "wel_chg": static_result.welfare_change * 100,
        "fiscal_balance_pct": static_result.fiscal_balance_pct,
        "confidence_consumer": static_result.confidence_consumer,
        "confidence_enterprise": static_result.confidence_enterprise,
        "confidence_investor": static_result.confidence_investor,
        "walras": static_result.walras_check,
        "gdp_baseline": sam_summary.get("gdp", 0) / 1e4,
        "fiscal_baseline_deficit": fiscal_baseline,
        "fiscal_end": fiscal_end,
        "fiscal_max_worsen": fiscal_max_worsen,
        "fiscal_max_improve": fiscal_max_improve,
        "fiscal_path": [round(float(v), 3) for v in fiscal_path],
        "conf_consumer_end": conf_consumer_end,
        "conf_enterprise_end": conf_enterprise_end,
        "conf_investor_end": conf_investor_end,
        "conf_consumer_max": conf_consumer_max,
        "conf_enterprise_max": conf_enterprise_max,
        "conf_investor_max": conf_investor_max,
        "conf_consumer_path": [round(float(v), 2) for v in conf_path.get("consumer", [])],
        "conf_enterprise_path": [round(float(v), 2) for v in conf_path.get("enterprise", [])],
        "conf_investor_path": [round(float(v), 2) for v in conf_path.get("investor", [])],
        "max_gdp_q": max_gdp_q,
        "max_gdp_val": max_gdp_val,
        "end_gdp": float(gdp_path[-1] * 100),
        "gdp_path": [round(x * 100, 3) for x in gdp_path],
        "beneficiaries": beneficiaries_str,
        "sufferers": sufferers_str,
        "all_sectors": all_sectors,
        "timestamp": datetime.now().strftime("%Y年%m月%d日 %H:%M"),
    }


def _timing_cn(t):
    return {"permanent": "永久实施", "temporary": "一次性脉冲", "anticipated": "预告后实施"}.get(t, t)


# =====================================================================
#  报告生成
# =====================================================================
def generate_policy_report(
    static_result,
    dyn_result: Dict,
    shock_params: Dict,
    sam_summary: Dict,
    sector_names: list = None,
    closure: str = "keynesian",
    shock_timing: str = "permanent",
    policy_timings: Dict = None,
    use_ai: bool = True,
    persona: str = "macro",
    api_key: str = "",
) -> str:
    """生成政策分析报告。

    Args:
        api_key: 用户提供的API密钥（最高优先级）。留空则从.env/环境变量读取。
    """
    data = _collect_report_data(
        static_result, dyn_result, shock_params, sam_summary,
        sector_names, closure, shock_timing, policy_timings
    )

    if use_ai:
        try:
            report = _call_llm(data, persona, api_key=api_key)
            if report:
                return report
            else:
                return _template_report(data, persona) + "\n\n> ⚠ 未配置API密钥，已使用模板生成。请在侧边栏输入GLM API Key后重试。"
        except Exception as e:
            print(f"[报告生成] AI调用失败: {e}，使用模板生成")
            return _template_report(data, persona) + f"\n\n> ⚠ API调用异常：{e}"

    return _template_report(data, persona)


# =====================================================================
#  LLM 调用
# =====================================================================
PERSONAS = {
    "macro": {
        "system": (
            "你是一位审慎、中立、客观的财政部宏观经济分析专家。"
            "你的分析风格严谨、数据驱动、不偏不倚——既不盲目乐观也不过度悲观，"
            "而是客观呈现政策利弊与风险。你的读者是财政部的司局级官员，"
            "他们需要的是冷静的判断而非宣传式表述。"
            "你善于从宏观均衡视角解读CGE模型结果，"
            "关注财政可持续性、物价稳定、就业质量等维度。"
        ),
        "structure": (
            "### 一、政策摘要（100字以内）\n"
            "用一段话概括政策组合及其总体宏观影响，语气审慎。\n\n"
            "### 二、宏观经济态势（300字）\n"
            "分析GDP、就业、CPI、居民福利的变化方向与幅度，解释传导机制。"
            "数字要具体引用。注意区分短期效应与中长期趋势。\n\n"
            "### 三、财政平衡与物价研判（300字）\n"
            "重点分析财政收支平衡：基准赤字水平、政策对财政平衡的影响（静态+12季度动态路径），"
            "指出最大恶化/改善时点。评估CPI变化对物价稳定的影响。"
            "给出审慎的财政可持续性风险提示。\n\n"
            "### 四、信心指数与预期效应（250字）\n"
            "分析消费者、企业、投资者三部门信心指数的变化（静态+动态路径），"
            "解释信心传导机制：GDP→消费信心→消费行为→GDP的正反馈循环。"
            "如政策为预告型，分析预告期内的预期效应。\n\n"
            "### 五、12季度动态过渡路径分析（200字）\n"
            "解读12季度(3年)过渡路径：效应何时达峰？是否收敛？有无中期效应？"
            "对中期(第4-8季度)和长期(第9-12季度)的态势分别研判。\n\n"
            "### 六、政策风险与建议（250字）\n"
            "给出3条审慎的政策建议，每条须指出潜在风险和退出机制。\n"
        ),
    },
    "industry": {
        "system": (
            "你是一位资深产业政策专家，专注于分析宏观经济政策对中国42个细分行业的差异化影响。"
            "你熟悉国家统计局42部门分类体系和部门间投入产出关联，"
            "善于从中间投入传导、要素替代效应、需求侧拉动等角度解读部门层面的变化。"
            "你的读者是发改委产业司和工信部的技术官员，"
            "他们需要的是精确到行业的产业链分析，而非泛泛的宏观数据。"
        ),
        "structure": (
            "### 一、产业影响总览（100字以内）\n"
            "概括政策组合对42部门的总体影响格局：受益面 vs 受压面。\n\n"
            "### 二、产业链传导机制（300字）\n"
            "从投入产出表角度解释政策冲击如何沿着产业链传导。"
            "哪些上游原材料部门最先受益？哪些下游消费品部门受影响最大？"
            "中间投入关联如何放大或缓冲冲击？\n\n"
            "### 三、重点受益行业深度分析（300字）\n"
            "选取产出增幅最大的5个部门，逐一分析受益原因：\n"
            "是需求侧拉动还是成本侧改善？是直接效应还是间接传导？\n"
            "该行业当前的产能利用率是否足以承接需求增长？\n\n"
            "### 四、受压行业风险提示（250字）\n"
            "选取产出降幅最大的部门（如有），分析受压原因和风险等级。"
            "是否存在产业链关联风险（如某行业收缩导致下游供应紧张）？\n\n"
            "### 五、产业政策建议（200字）\n"
            "基于部门层面的分析，给出3条精准的产业政策建议。"
            "建议要落实到具体行业，不要泛泛而谈。\n"
        ),
    },
}


def _call_llm(data: Dict, persona: str = "macro", api_key: str = "") -> Optional[str]:
    api_config = _load_api_config(api_key)
    if not api_config["api_key"]:
        return None

    prompt = _build_prompt(data, persona)
    p_config = PERSONAS.get(persona, PERSONAS["macro"])

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_config["api_key"], base_url=api_config["base_url"])
        resp = client.chat.completions.create(
            model=api_config["model"],
            messages=[
                {"role": "system", "content": p_config["system"]},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=3000,
        )
        return resp.choices[0].message.content
    except ImportError:
        return _call_llm_requests(api_config, prompt, p_config["system"])


def _call_llm_requests(api_config, prompt, system_prompt):
    import requests
    url = f"{api_config['base_url']}/chat/completions"
    headers = {"Authorization": f"Bearer {api_config['api_key']}", "Content-Type": "application/json"}
    payload = {
        "model": api_config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 3000,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code == 429:
        raise RuntimeError("API额度不足（429）")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _build_prompt(data: Dict, persona: str = "macro") -> str:
    p_config = PERSONAS.get(persona, PERSONAS["macro"])

    # 42部门详细数据（产业专家需要完整列表）
    sector_detail = ""
    if persona == "industry":
        lines = []
        for s in data["all_sectors"]:
            lines.append(f"  {s['code']} {s['name']}: 产出{s['output_chg']:+.3f}% 价格{s['price_chg']:+.3f}% 消费{s['cons_chg']:+.3f}%")
        sector_detail = f"\n## 42部门完整数据（产出/价格/消费变化%）\n" + "\n".join(lines)

    return f"""请基于以下CGE模型模拟结果，撰写一份中文政策分析报告。

## 模拟场景
- 政策内容：{data['policy_desc']}
- 模型设定：{data['closure_cn']}
- 基准GDP：{data['gdp_baseline']:,.0f}万亿元

## 核心结果（静态均衡效应）
- GDP变化：{data['gdp_chg']:+.3f}%
- CPI变化：{data['cpi_chg']:+.3f}%
- 就业变化：{data['emp_chg']:+.3f}%
- 居民福利变化（等效变异EV）：{data['wel_chg']:+.3f}%

## 财政平衡
- 基准财政赤字：{data['fiscal_baseline_deficit']:+.2f}万亿元
- 政策对财政平衡的静态影响：{data['fiscal_balance_pct']:+.2f}个百分点
- 12季度财政平衡动态路径（pp）：{data['fiscal_path']}
- 路径最大恶化：{data['fiscal_max_worsen']:+.2f}pp；最大改善：{data['fiscal_max_improve']:+.2f}pp；第12季度：{data['fiscal_end']:+.2f}pp

## 信心指数（基准=50，正值=改善）
- 消费者信心变化（静态）：{data['confidence_consumer']:+.2f}点；12季度路径：{data['conf_consumer_path']}
- 企业信心变化（静态）：{data['confidence_enterprise']:+.2f}点；12季度路径：{data['conf_enterprise_path']}
- 投资者信心变化（静态）：{data['confidence_investor']:+.2f}点；12季度路径：{data['conf_investor_path']}

## 12季度(3年)动态路径
- GDP季度偏差路径（%）：{data['gdp_path']}
- 最大效应出现在第{data['max_gdp_q']}季度（{data['max_gdp_val']:+.3f}%）
- 第12季度GDP偏差：{data['end_gdp']:+.3f}%

## 部门影响
- 最受益部门（前10）：{data['beneficiaries']}
- 最受压部门（前10）：{data['sufferers']}
{sector_detail}

## 技术指标
- 瓦尔拉斯法则残差：{data['walras']:.2e}

---

请按以下结构撰写报告（总计1000-1500字）：

{p_config['structure']}

请直接输出报告正文（Markdown格式），不要输出多余的解释。"""


def _template_report(data: Dict, persona: str = "macro") -> str:
    gdp_dir = "上升" if data['gdp_chg'] > 0 else "下降" if data['gdp_chg'] < 0 else "不变"
    emp_dir = "增加" if data['emp_chg'] > 0 else "减少" if data['emp_chg'] < 0 else "不变"
    cpi_dir = "上升" if data['cpi_chg'] > 0 else "下降" if data['cpi_chg'] < 0 else "不变"

    title = "宏观经济分析报告" if persona == "macro" else "产业影响分析报告"

    if persona == "macro":
        return f"""# {title}

> 生成时间：{data['timestamp']}
> 分析引擎：Johansen对数线性化CGE模型 · 42部门 · 基准年2025 · {data['closure_cn']}

---

## 一、政策摘要

本报告分析了以下政策的宏观经济影响：**{data['policy_desc']}**。
模拟结果显示，该政策组合将使GDP{gdp_dir}{abs(data['gdp_chg']):.3f}%，就业{emp_dir}{abs(data['emp_chg']):.3f}%，CPI{cpi_dir}{abs(data['cpi_chg']):.3f}%。

## 二、宏观经济态势

在{data['closure_cn']}下：

- **GDP**：变化{data['gdp_chg']:+.3f}%，基准GDP为{data['gdp_baseline']:,.0f}万亿元，对应绝对变化约{data['gdp_baseline']*data['gdp_chg']/100:,.2f}万亿元。
- **就业**：变化{data['emp_chg']:+.3f}%。
- **物价**：CPI变化{data['cpi_chg']:+.3f}%。
- **居民福利**：等效变异(EV)为{data['wel_chg']:+.3f}%，{"表明居民整体福利改善。" if data['wel_chg']>0 else "表明居民整体福利受损。"}

## 三、财政平衡与物价研判

- **基准财政赤字**：{data['fiscal_baseline_deficit']:+.2f}万亿元。
- **政策对财政平衡的静态影响**：{data['fiscal_balance_pct']:+.2f}个百分点。{"政策导致财政平衡改善。" if data['fiscal_balance_pct']>0 else "政策导致财政平衡恶化，赤字扩大。" if data['fiscal_balance_pct']<0 else "财政平衡基本不变。"}
- **12季度动态路径**：最大恶化{data['fiscal_max_worsen']:+.2f}pp，最大改善{data['fiscal_max_improve']:+.2f}pp，第12季度{data['fiscal_end']:+.2f}pp。
- CPI变化{data['cpi_chg']:+.3f}%，{"存在物价上行压力。" if data['cpi_chg']>0.1 else "物价基本稳定。" if abs(data['cpi_chg'])<0.1 else "物价有所回落。"}

## 四、信心指数与预期效应

- **消费者信心**：变化{data['confidence_consumer']:+.2f}点，12季度峰值{data['conf_consumer_max']:+.2f}点。
- **企业信心**：变化{data['confidence_enterprise']:+.2f}点，12季度峰值{data['conf_enterprise_max']:+.2f}点。
- **投资者信心**：变化{data['confidence_investor']:+.2f}点，12季度峰值{data['conf_investor_max']:+.2f}点。
- {"三部门信心均改善，预期传导渠道畅通。" if all([data['confidence_consumer']>0, data['confidence_enterprise']>0, data['confidence_investor']>0]) else "部分部门信心分化，需关注预期传导阻滞风险。"}

## 五、12季度动态过渡路径

- GDP效应在第{data['max_gdp_q']}季度达到峰值（{data['max_gdp_val']:+.3f}%）。
- 第12季度GDP偏差{data['end_gdp']:+.3f}%，{"趋于新均衡。" if abs(data['end_gdp'])<abs(data['max_gdp_val']) else "效应仍在持续。"}

## 六、政策风险与建议

1. 关注受益部门的产能瓶颈与通胀传导风险。
2. 评估财政可持续性，基准赤字{data['fiscal_baseline_deficit']:+.2f}万亿元，政策影响{data['fiscal_balance_pct']:+.2f}pp，必要时设计退出机制。
3. 动态效应在第{data['max_gdp_q']}季度达峰，应在此窗口期加强宏观审慎管理。

---

> 本报告由模板自动生成（AI接口暂不可用）。瓦尔拉斯残差{data['walras']:.2e}。
"""
    else:
        # 产业专家模板
        sectors_by_impact = sorted(data["all_sectors"], key=lambda x: x["output_chg"], reverse=True)
        top10_str = "\n".join([f"- {s['code']} {s['name']}: 产出{s['output_chg']:+.3f}%" for s in sectors_by_impact[:10]])
        bottom10_str = "\n".join([f"- {s['code']} {s['name']}: 产出{s['output_chg']:+.3f}%" for s in sectors_by_impact[-10:]])

        return f"""# {title}

> 生成时间：{data['timestamp']}
> 分析引擎：Johansen对数线性化CGE模型 · 42部门

---

## 一、产业影响总览

政策组合**{data['policy_desc']}**对42部门产生差异化影响。总体格局：{data['beneficiaries']}等部门受益明显。

## 二、产业链传导机制

政策冲击通过中间投入关联沿产业链传导。上游原材料和能源部门通常最先响应，下游消费品部门通过需求侧拉动间接受益。

## 三、重点受益行业（产出增幅前10）

{top10_str}

## 四、受压行业（产出降幅前10）

{bottom10_str}

## 五、产业政策建议

1. 针对受益最大的行业，评估产能承接能力，防止过度扩张。
2. 对受压行业，考虑过渡期支持措施。
3. 关注产业链关联效应，防范局部收缩引发的供应链风险。

---

> 本报告由模板自动生成（AI接口暂不可用）。瓦尔拉斯残差{data['walras']:.2e}。
"""
