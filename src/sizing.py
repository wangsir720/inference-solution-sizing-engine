# -*- coding: utf-8 -*-
"""卡数与显存测算 —— **显式计算 KV Cache**。

对应 JD 任职资格第 2 条（推理全栈基础）中的显存与分布式基础。

## 为什么这个模块非做不可

推理场景的显存瓶颈**常常不在权重而在 KV Cache**：

| 场景 | 权重占用 | KV Cache 占用 | 谁是瓶颈 |
|---|---|---|---|
| 短上下文低并发 | 大 | 小 | 权重 |
| 长上下文高并发 | 不变 | 随 `并发 × 序列长度` 线性增长 | **KV Cache** |

只算权重会系统性低估显存需求 → 卡数算少了 → 交付时 OOM。
这是推理选型与训练选型最本质的差别。

## 公式来源

`data/assumptions.md` A-01 登记了完整公式与依据，本模块只实现，不重新推导。
"""
from __future__ import annotations

from .catalog import Row, get_accelerator, get_model

# ---- A-01 登记的常量 ----
BYTES_PER_PARAM = {"bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
# KV Cache 的 K 与 V 各一份
KV_MULTIPLIER = 2
# A-01 激活与临时缓冲余量系数
RUNTIME_VRAM_FACTOR = 1.10
# A-02 显存余量下限
VRAM_MARGIN = 0.05
# A-01 KV Cache 每元素字节数（bf16 KV Cache = 2 Byte）
KV_DTYPE_BYTES = 2


class SizingError(Exception):
    pass


def weight_gib(model: Row) -> float:
    """权重显存（GiB）。dtype 未知则报错，不猜。"""
    params = model.n("params_b")
    if params is None:
        raise SizingError("模型 %s 未登记参数量" % model.key)
    dtype = (model.s("dtype") or "").lower()
    if dtype not in BYTES_PER_PARAM:
        raise SizingError("模型 %s 的 dtype %r 未登记每参数字节数，不做估算"
                          % (model.key, dtype))
    return params * 1e9 * BYTES_PER_PARAM[dtype] / 1024 ** 3


def kv_cache_gib(model: Row, input_len: int, output_len: int,
                 kv_dtype_bytes: int = 2) -> float | None:
    """KV Cache 显存（GiB）。

    结构参数（层数 / kv_head 数 / head_dim）任一缺失即返回 None ——
    缺失时不做估算，因为 KV Cache 恰好是本项目最不能估错的量。
    """
    layers = model.n("layers")
    kv_heads = model.n("kv_heads")
    head_dim = model.n("head_dim")
    if layers is None or kv_heads is None or head_dim is None:
        return None
    tokens = input_len + output_len
    nbytes = KV_MULTIPLIER * layers * kv_heads * head_dim * tokens * kv_dtype_bytes
    return nbytes / 1024 ** 3


def vram_breakdown(model: Row, input_len: int, output_len: int) -> dict:
    """显存构成拆解。KV Cache 算不出时显式标 None 并说明。"""
    w = weight_gib(model)
    kv_dtype_bytes = KV_DTYPE_BYTES
    kv = kv_cache_gib(model, input_len, output_len, kv_dtype_bytes)
    if kv is None:
        need = w * RUNTIME_VRAM_FACTOR
        trace = ("权重 %.2f GiB；**KV Cache 未计入：模型结构参数（层数/kv_heads/head_dim）"
                 "在公开资料中未完整给出，本估算为下界**；×余量系数 %.2f = %.2f GiB"
                 % (w, RUNTIME_VRAM_FACTOR, need))
    else:
        need = (w + kv) * RUNTIME_VRAM_FACTOR
        trace = ("权重 %.2f GiB + KV Cache %.2f GiB（%d token × 2(K,V) × %d 层 × "
                 "%d kv_head × %d head_dim × %d Byte）= %.2f GiB；×余量系数 %.2f = %.2f GiB"
                 % (w, kv, input_len + output_len, int(model.n("layers")),
                    int(model.n("kv_heads")), int(model.n("head_dim")),
                    kv_dtype_bytes, kv, RUNTIME_VRAM_FACTOR, need))
    return {
        "weight_gib": round(w, 3),
        "kv_cache_gib": round(kv, 3) if kv is not None else None,
        "required_gib": round(need, 3),
        "kv_share": (round(kv / (w + kv), 4) if kv and (w + kv) else None),
        "trace": trace,
        "assumption_refs": ["A-01"],
    }


def size(scenario: dict, accel_id: str) -> dict:
    """测算指定加速卡上的卡数需求。"""
    w = scenario.get("workload")
    if not isinstance(w, dict):
        raise SizingError("场景缺少 workload")
    model_id = w.get("model_id")
    if not model_id:
        raise SizingError("场景缺少 workload.model_id")
    in_len = w.get("input_len_tokens")
    out_len = w.get("output_len_tokens")
    if in_len is None or out_len is None:
        raise SizingError("场景缺少 input_len_tokens 或 output_len_tokens")
    if int(in_len) < 0 or int(out_len) < 0:
        raise SizingError("token 数不能为负数")

    model = get_model(model_id)
    accel = get_accelerator(accel_id)
    mem = vram_breakdown(model, int(in_len), int(out_len))

    vram = accel.n("vram_gib")
    if vram is None:
        return {
            "scenario_id": scenario.get("scenario_id"),
            "model_id": model_id, "accel_model": accel_id,
            "cards": None, "verdict": "unknown",
            "memory": mem,
            "reason": "加速卡 %s 未登记单卡显存，公开资料未查到该参数" % accel_id,
            "accel_cite": accel.cite(), "model_cite": model.cite(),
            "assumption_refs": ["A-01", "A-02"],
        }

    need_per_card = vram * (1 - VRAM_MARGIN)
    raw_cards = mem["required_gib"] / need_per_card
    # A-02：向上取整到整机粒度
    cards = int(raw_cards) if raw_cards == int(raw_cards) else int(raw_cards) + 1
    if cards < 1:
        cards = 1

    total_vram = cards * vram
    if mem["required_gib"] <= total_vram:
        verdict = "supported"
        reason = ("单卡可用显存 %.2f GiB（总 %.2f × %.0f%% 余量），需求 %.2f GiB，余量 %.1f%%"
                  % (need_per_card, vram, VRAM_MARGIN * 100,
                     mem["required_gib"], (total_vram - mem["required_gib"]) / total_vram * 100))
    else:
        verdict = "needs_scaling"
        reason = ("按当前卡数总显存 %.2f GiB 仍低于需求 %.2f GiB，"
                  "需增加卡数或改用更大显存的型号" % (total_vram, mem["required_gib"]))

    return {
        "scenario_id": scenario.get("scenario_id"),
        "model_id": model_id,
        "accel_model": accel_id,
        "cards": cards,
        "cards_raw": round(raw_cards, 3),
        "accel_vram_gib": vram,
        "accel_tdp_w": accel.n("tdp_w"),
        "memory": mem,
        "total_vram_gib": round(total_vram, 2),
        "verdict": verdict,
        "reason": reason,
        "accel_cite": accel.cite(),
        "model_cite": model.cite(),
        "assumption_refs": ["A-01", "A-02"],
    }
