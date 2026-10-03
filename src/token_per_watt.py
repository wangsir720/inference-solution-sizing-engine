# -*- coding: utf-8 -*-
"""每瓦 Token 吞吐量测算（★ 本项目核心指标）。

这个指标回答的是客户最直接的问题：**「我这张卡，每瓦电费能跑多少 Token？」**

## 为什么这个指标对客户有意义

推理成本 = 硬件折旧 + 电费。电费的计算方式只有一个：

```
年电费 = 整机功率(W) × 年运行小时 × 利用率 ÷ 1000 × PUE × 电价(元/度)
```

功率是**固定的**（卡买回来就那样），能变的只有吞吐。
因此**单位功率的吞吐能力，直接决定单位 Token 的电费成本**。

```
每 Token 电费 ≈ 1 ÷ (每瓦 Token 吞吐 × 1000 × PUE × 利用率 × 电价)
```

**每瓦 Token 吞吐越高，每 Token 电费越低。** 这就是推理专用芯片与通用 GPU 的核心差异点。

## 本模块的诚实边界

吞吐基线与优化系数**全部是占位值**（`data/assumptions.md` B-02）——
本项目无加速卡，无法实测。因此本模块输出的是
**「在给定假设下的模型推演结果」**，不是性能预测。

每份输出都带 `assumption_refs`，让使用方清楚哪些是实测可替换的参数。
"""
from __future__ import annotations

from .catalog import Row, get_model, load_engines
from .tp_ep import EP_EFFICIENCY, TP_EFFICIENCY

# ---- B-02 登记的占位参数 ----
# 单卡解码 TPS 基线：按参数量 70 亿对应 60 TPS（bf16）线性折算
TPS_AT_7B = 60.0
REFERENCE_PARAMS_B = 7.0
# 机箱与散热开销（整机 TDP = 卡 TDP × 卡数 × 1.15）
CHASSIS_OVERHEAD = 0.15
# B-03 优化系数（不叠加，取最大值）
OPT_QUANT = 1.6        # FP8/INT8 量化
OPT_SPEC_DECODE = 1.8  # 投机解码
OPT_PREFIX_CACHE = 1.5  # 前缀缓存命中
OPT_CONT_BATCH = 1.4   # 连续批处理

OPT_LABELS = {
    "quantization": "FP8/INT8 量化",
    "speculative_decoding": "投机解码",
    "prefix_cache": "前缀缓存命中",
    "continuous_batching": "连续批处理",
}


class TokenPerWattError(Exception):
    pass


def single_card_tps(model: Row) -> tuple[float, str]:
    """单卡解码吞吐估算（token/秒）。返回 (值, 推导说明)。"""
    params = model.n("params_b")
    if params is None:
        raise TokenPerWattError("模型 %s 未登记参数量" % model.key)
    tps = TPS_AT_7B * params / REFERENCE_PARAMS_B
    return tps, ("按 70 亿参数 bf16 单卡 60 token/s 线性折算："
                 "60 × %.1fB ÷ 7.0B = %.1f token/s（**占位值，须用目标卡实测替换**）"
                 % (params, tps))


def best_optimization(engines: list[str]) -> tuple[float, list[str]]:
    """按已选推理引擎挑出可用的优化项，返回 (最大系数, 命中项)。"""
    applied = []
    best = 1.0
    for name in engines:
        row = load_engines().get(name)
        if row is None:
            continue
        if row.b("quant_modes") and "FP8" in (row.s("quant_modes") or ""):
            applied.append(OPT_LABELS["quantization"])
            best = max(best, OPT_QUANT)
        if row.s("engine") in ("TensorRT-LLM",) or "投机" in (row.s("note") or ""):
            applied.append(OPT_LABELS["speculative_decoding"])
            best = max(best, OPT_SPEC_DECODE)
        if "RadixAttention" in (row.s("note") or "") or "前缀" in (row.s("note") or ""):
            applied.append(OPT_LABELS["prefix_cache"])
            best = max(best, OPT_PREFIX_CACHE)
        if row.b("batching"):
            applied.append(OPT_LABELS["continuous_batching"])
            best = max(best, OPT_CONT_BATCH)
    # 去重保序
    seen, uniq = set(), []
    for a in applied:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return best, uniq


def compute(sizing: dict, model_id: str, engines: list[str] | None = None,
            util: float = 0.6) -> dict:
    """计算每瓦 Token 吞吐量。"""
    cards = sizing.get("cards")
    if not cards:
        raise TokenPerWattError("卡数为空，无法计算每瓦吞吐")
    if util is not None and not 0 < util <= 1:
        raise TokenPerWattError("利用率须在 0 与 1 之间，实际为 %r" % util)

    model = get_model(model_id)
    tdp = sizing.get("accel_tdp_w")
    if tdp is None:
        return {
            "available": False,
            "reason": "加速卡 %s 未登记 TDP，无法计算每瓦吞吐" % sizing.get("accel_model"),
            "assumption_refs": ["B-01"],
        }

    tps_one, tps_trace = single_card_tps(model)
    eff = EP_EFFICIENCY if bool(model.b("is_moe")) else TP_EFFICIENCY
    opt_factor, opt_items = best_optimization(engines or [])

    raw_total = tps_one * cards * eff * opt_factor * (util or 1.0)
    system_tdp = tdp * cards * (1 + CHASSIS_OVERHEAD)

    return {
        "available": True,
        "accel_model": sizing.get("accel_model"),
        "cards": cards,
        "single_card_tps": round(tps_one, 2),
        "parallel_efficiency": eff,
        "optimization_factor": opt_factor,
        "optimizations_applied": opt_items,
        "utilization": util,
        "total_tps": round(raw_total, 2),
        "system_tdp_w": round(system_tdp, 1),
        "token_per_watt": round(raw_total / system_tdp, 5),
        "trace": ("单卡 %.1f token/s（%s）× %d 卡 × 并行效率 %.2f × 优化系数 %.1f%s "
                  "× 利用率 %.2f = 整机 %.1f token/s；整机功耗 = %.0f W × %d 卡 × "
                  "1.15 机箱开销 = %.0f W；**每瓦 Token 吞吐 = %.1f / %.0f = %.4f token/s/W**"
                  % (tps_one, tps_trace, cards, eff, opt_factor,
                     ("（%s）" % "、".join(opt_items)) if opt_items else "",
                     util or 1.0, raw_total, tdp, cards, system_tdp,
                     raw_total, system_tdp, raw_total / system_tdp)),
        "caveats": [
            "单卡 TPS 基线为占位值，非实测",
            "优化系数不叠加，取最大值 —— 多项优化作用于不同瓶颈，相乘会严重高估",
            "不含 TTFT 影响：本指标只计 decode 吞吐，首 token 延迟需单独评估",
        ],
        "assumption_refs": ["B-01", "B-02", "B-03"],
    }


def cost_per_million_tokens(tpw: dict, util: float, pue: float,
                            price_per_kwh: float) -> dict | None:
    """由每瓦吞吐推算每百万 token 电费。返回 None 表示前置数据不足。"""
    if not tpw.get("available"):
        return None
    t = tpw["token_per_watt"]
    if not t or t <= 0:
        return None
    if util <= 0 or pue <= 0 or price_per_kwh <= 0:
        raise TokenPerWattError("利用率、PUE、电价必须为正数")
    # 每百万 token 需要的电量：1e6 / (t × 1000 W) 度
    kwh = 1e6 / (t * 1000)
    cost = kwh * pue * price_per_kwh
    return {
        "token_per_watt": t,
        "kwh_per_million_tokens": round(kwh, 3),
        "cost_per_million_tokens_cny": round(cost, 4),
        "trace": ("每瓦 %.4f token/s → 每百万 token 需 %.1f W·h；"
                  "× PUE %.1f × 电价 %.2f 元/度 = %.4f 元/百万 token"
                  % (t, kwh, pue, price_per_kwh, cost)),
        "note": "仅电费，不含硬件折旧与运维人力",
        "assumption_refs": ["B-01", "C-01"],
    }
