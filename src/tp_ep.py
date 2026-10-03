# -*- coding: utf-8 -*-
"""并行策略建议：张量并行（TP）vs 专家并行（EP）。

对应 JD 任职资格第 2 条的「分布式系统（张量并行/专家并行）」。

## 为什么要分两种

| 维度 | 张量并行（TP） | 专家并行（EP） |
|---|---|---|
| 适用模型 | **稠密模型**（每层权重都参与计算） | **MoE 模型**（仅激活部分专家） |
| 切分对象 | 权重矩阵按行/列切 | 按专家切 |
| 通信量 | **大** —— 每层需 all-reduce | 小 —— 仅路由时 all-to-all |
| 通信频率 | 每层 2 次 | 每 MoE 层 1 次 |
| 负载均衡风险 | 低 | **高** —— 热门专家会过载 |
| 瓶颈 | 互联带宽 | 专家负载不均 + 显存碎片 |

**选错并行方式的后果**：稠密模型用 EP 无法切分（没有专家可切）；
MoE 模型用 TP 会让通信量远超必要，吞吐被互联拖死。

## 通信开销估算

```
TP 通信量 ≈ 2 × layers × batch_tokens × hidden_dim × dtype_bytes × (TP-1)/TP
EP 通信量 ≈ 2 × moe_layers × batch_tokens × topk × hidden_dim × dtype_bytes
```

本模块输出的是**量级估算**（GiB/推理步），用于对比两种策略的相对代价，
不作为绝对性能预测。
"""
from __future__ import annotations

from .catalog import get_model

# B-02 并行效率（占位待实测替换）
TP_EFFICIENCY = 0.85
EP_EFFICIENCY = 0.90
# MoE 路由默认激活专家数
DEFAULT_TOPK = 2
# MoE 层占总层数的比例（占位待实测替换）：
# 主流 MoE 模型通常每 1 层或每几层设置一个 MoE 层，attention/FFN 仍为 dense。
# **这个比例直接决定 EP 相对 TP 的通信优势，是本模块最敏感的假设。**
DEFAULT_MOE_LAYER_RATIO = 0.25


class ParallelError(Exception):
    pass


def strategy_for(model) -> dict:
    """按模型结构推荐并行策略，并说明为什么。"""
    is_moe = model.b("is_moe")
    if is_moe is None:
        return {
            "recommended": "unknown",
            "reason": "模型 %s 未登记是否为 MoE，无法推荐并行策略" % model.key,
            "efficiency": None, "tradeoff": "", "assumption_refs": [],
        }
    if is_moe:
        return {
            "recommended": "expert_parallel",
            "efficiency": EP_EFFICIENCY,
            "reason": ("模型 %s 是 MoE 架构，专家按 token 动态激活。"
                       "EP 只在 MoE 层做 all-to-all，通信量远小于逐层 all-reduce 的 TP；"
                       "用 TP 切 MoE 会让互联成为瓶颈" % model.key),
            "tradeoff": "代价是**专家负载不均**：热门专家过载会拖慢整批推理，"
                        "需要容量因子或冗余专家来缓解",
            "assumption_refs": ["B-02"],
        }
    return {
        "recommended": "tensor_parallel",
        "efficiency": TP_EFFICIENCY,
        "reason": ("模型 %s 是稠密模型，每层权重都参与计算，没有专家可切，"
                   "只能用 TP 按权重矩阵切分" % model.key),
        "tradeoff": "代价是**每层需 2 次 all-reduce**，通信量随 TP 度数上升，"
                    "卡数越多对互联带宽要求越高",
        "assumption_refs": ["B-02"],
    }


def comm_volume(model_id: str, cards: int, batch_tokens: int) -> dict:
    """估算 TP 与 EP 两种策略的每步通信量（GiB）。

    只做**同模型内的相对比较**（TP vs EP），不做跨模型比较——
    不同模型的 hidden 尺寸不同，跨模型比通信量没有意义。
    """
    if cards < 1:
        raise ParallelError("卡数必须为正整数，实际为 %r" % cards)
    if batch_tokens < 1:
        raise ParallelError("批 token 数必须为正整数，实际为 %r" % batch_tokens)
    model = get_model(model_id)
    layers = model.n("layers")
    kv_heads = model.n("kv_heads")
    head_dim = model.n("head_dim")
    if layers is None or kv_heads is None or head_dim is None:
        return {
            "available": False,
            "reason": "模型结构参数不完整，无法估算通信量",
            "assumption_refs": ["A-01"],
        }
    # 近似 hidden size：GQA 场景下 h ≈ kv_heads × head_dim × 分组倍数，
    # 分组倍数未在公开资料中逐项给出，用 4 作占位并显式标注。
    # 注意：这是**占位近似**，对 MoE 与稠密模型的分组倍数可能不同，
    # 因此本模块的结论只用于同一模型内 TP/EP 的相对比较。
    hidden = kv_heads * head_dim * 4
    dtype_bytes = 2 if (model.s("dtype") or "bf16") in ("bf16", "fp16") else 1
    is_moe = bool(model.b("is_moe"))

    # TP：每层 2 次 all-reduce（attention + FFN 各一次），通信量含 (cards-1)/cards 因子
    #     —— all-reduce 是全互联归约，卡越多每步通信量越大
    tp_bytes = 2 * layers * batch_tokens * hidden * dtype_bytes * (cards - 1) / cards
    # EP：**只有 MoE 层**需要 all-to-all（把 token 路由到被激活的专家）；
    #     attention 与 FFN 仍是 dense 计算，走 TP 路径。
    #     这才是 EP 通信量低的根本原因：MoE 层占比小，通信次数少。
    #     moe_ratio = MoE 层数占总层数（DeepSeek/Mixtral 类模型通常每 1 或几层一个 MoE）
    moe_ratio = DEFAULT_MOE_LAYER_RATIO
    moe_layers = max(1, int(round(layers * moe_ratio)))
    # 非 MoE 模型没有专家可路由，EP 通信量恒为 0
    ep_bytes = (moe_layers * batch_tokens * hidden * dtype_bytes * DEFAULT_TOPK
                if is_moe else 0.0)

    ratio = None
    if is_moe and ep_bytes > 0 and tp_bytes > 0:
        ratio = round(ep_bytes / tp_bytes, 4)

    return {
        "available": True,
        "is_moe": is_moe,
        "hidden_approx": hidden,
        "hidden_note": "按 kv_heads × head_dim × 4 近似，GQA 分组倍数未登记（占位）",
        "tp_gib_per_step": round(tp_bytes / 1024 ** 3, 4),
        "ep_gib_per_step": round(ep_bytes / 1024 ** 3, 4),
        "ep_over_tp": ratio,
        "trace": ("TP: 2(all-reduce/层) × %d 层 × %d token × %d hidden × %d Byte × "
                  "(%d-1)/%d = %.4f GiB/步；%s"
                  % (int(layers), batch_tokens, int(hidden), dtype_bytes,
                     cards, cards, tp_bytes / 1024 ** 3,
                     ("EP: 1(all-to-all/层) × %d 个 MoE 层（占 %d 层 × %.0f%%）× "
                      "%d token × %d hidden × %d Byte × topk %d = %.4f GiB/步；"
                      "**同模型内 EP/TP = %.2f**（EP 的优势来自只有 MoE 层需要通信，"
                      "attention/FFN 仍是 dense）"
                      % (moe_layers, int(layers), DEFAULT_MOE_LAYER_RATIO * 100,
                         batch_tokens, int(hidden), dtype_bytes,
                         DEFAULT_TOPK, ep_bytes / 1024 ** 3, ratio)) if is_moe and ratio
                     else "非 MoE 模型，EP 不适用")),
        "caveat": ("hidden 尺寸为占位近似（kv_heads × head_dim × 4），"
                   "对 MoE 与稠密模型的 GQA 分组倍数可能不同。"
                   "结论只用于**同一模型内** TP 与 EP 的相对比较，不做跨模型比较。"),
        "assumption_refs": ["B-02"],
    }
