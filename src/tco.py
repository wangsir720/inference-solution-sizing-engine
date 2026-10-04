# -*- coding: utf-8 -*-
"""三年 TCO：**电费单列**。

支撑采购决策的成本口径：算得清、点得回。

## 两条不可让步的规则

1. **电费单列** —— 推理场景电费常占运营成本大头，混在总额里客户看不清。
2. **硬件采购价未查到就列为未计入项，不用估算值凑总额** ——
   加速卡单价属商业信息，厂商未公开。编一个价格算出一个漂亮的 TCO，
   在客户面前一问就穿帮。这比少算更可信。
"""
from __future__ import annotations

from .token_per_watt import CHASSIS_OVERHEAD

# ---- C-01 登记的占位参数 ----
DEFAULT_POWER_PRICE = 0.8   # 元/度，须按项目所在地替换
DEFAULT_PUE = 1.3
HOURS_PER_YEAR = 8760
YEARS = 3


class TcoError(Exception):
    pass


def compute(sizing: dict, tpw: dict, util: float = 0.6,
            power_price: float = DEFAULT_POWER_PRICE,
            pue: float = DEFAULT_PUE,
            hardware_price: float | None = None,
            ops_cost_per_year: float | None = None) -> dict:
    """三年 TCO。电费与硬件分列。"""
    cards = sizing.get("cards")
    if not cards:
        raise TcoError("卡数为空，无法测算 TCO")
    tdp = sizing.get("accel_tdp_w")
    if tdp is None:
        raise TcoError("加速卡 TDP 未登记，无法测算电费")
    if power_price <= 0 or pue <= 0:
        raise TcoError("电价与 PUE 必须为正数")
    if util is not None and not 0 < util <= 1:
        raise TcoError("利用率须在 0 与 1 之间，实际为 %r" % util)

    system_w = tdp * cards * (1 + CHASSIS_OVERHEAD)
    effective_hours = HOURS_PER_YEAR * (util or 1.0)
    annual_kwh = system_w * effective_hours / 1000
    annual_power = annual_kwh * power_price
    annual_pue_power = annual_power * (pue - 1)  # 制冷等 PUE 附加部分

    lines = []
    lines.append({
        "category": "运营", "item": "电费（IT 功耗）",
        "qty": round(annual_kwh, 1), "unit": "kWh/年",
        "unit_price": power_price, "unit_price_source": "占位值，须按项目所在地电价替换",
        "annual_cny": round(annual_power, 2),
        "derivation": ("整机 %.0f W × %.0f h/年 × 利用率 %.2f ÷ 1000 = %.1f kWh/年；"
                       "× %.2f 元/度 = %.2f 元/年"
                       % (system_w, HOURS_PER_YEAR, util or 1.0, annual_kwh,
                          power_price, annual_power)),
    })
    lines.append({
        "category": "运营", "item": "电费（PUE 附加：制冷等）",
        "qty": round(annual_kwh, 1), "unit": "kWh/年",
        "unit_price": round(power_price * (pue - 1), 4), "unit_price_source": "PUE %.1f" % pue,
        "annual_cny": round(annual_pue_power, 2),
        "derivation": "IT 电费 × (PUE %.1f − 1) = %.2f 元/年" % (pue, annual_pue_power),
    })
    if ops_cost_per_year is not None:
        lines.append({
            "category": "运营", "item": "运维人力",
            "qty": 1, "unit": "年", "unit_price": ops_cost_per_year,
            "unit_price_source": "由调用方传入实际运维合同价",
            "annual_cny": round(float(ops_cost_per_year), 2),
            "derivation": "按实际运维合同价计入",
        })

    unpriced = []
    if hardware_price is None:
        unpriced.append({
            "item": "加速卡与整机硬件采购",
            "reason": "厂商未公开统一售价，属商业信息；本项目不编造单价",
            "action": "取得正式报价后传入 hardware_price 参数重算",
        })
    else:
        total_hw = hardware_price * cards
        lines.append({
            "category": "资本", "item": "加速卡采购",
            "qty": cards, "unit": "卡", "unit_price": hardware_price,
            "unit_price_source": "由调用方传入正式报价",
            "annual_cny": 0.0,
            "derivation": "一次性投入 %.2f 元（三年摊销见下方说明）" % total_hw,
        })

    annual_total = sum(x["annual_cny"] for x in lines)
    power_total = annual_power + annual_pue_power

    per_million = None
    if tpw.get("available"):
        from .token_per_watt import cost_per_million_tokens
        per_million = cost_per_million_tokens(tpw, util or 1.0, pue, power_price)

    return {
        "cards": cards,
        "system_power_w": round(system_w, 1),
        "utilization": util,
        "pue": pue,
        "power_price": power_price,
        "years": YEARS,
        "lines": lines,
        "unpriced_items": unpriced,
        "annual_power_cost_cny": round(power_total, 2),
        "power_share_of_known": round(power_total / annual_total, 4) if annual_total else None,
        "annual_known_cny": round(annual_total, 2),
        "three_year_known_cny": round(annual_total * YEARS, 2),
        "per_million_tokens": per_million,
        "trace": ("年电费 = %.0f W × 8760 h × %.2f 利用率 ÷ 1000 = %.1f kWh；"
                  "× %.2f 元 = %.2f 元/年；PUE %.1f 附加 %.2f 元/年；"
                  "已计入部分合计 %.2f 元/年 → 三年 %.2f 元"
                  % (system_w, util or 1.0, annual_kwh, power_price, annual_power,
                     pue, annual_pue_power, annual_total, annual_total * YEARS)),
        "completeness_note": (
            "**合计不完整**：加速卡采购价未公开，未计入。"
            "对客户报 TCO 时必须补上硬件价，否则这个数字没有决策价值。"
            if hardware_price is None else
            "已含硬件采购；不含机房建设、网络设备与实施人力"),
        "assumption_refs": ["C-01", "C-02", "B-01"],
    }
