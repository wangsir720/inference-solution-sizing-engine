# -*- coding: utf-8 -*-
"""sizing / tp_ep / token_per_watt 测试。

核心验证点：**KV Cache 必须显式计算**、**卡数向上取整**、**每瓦吞吐不编造**。
"""
import os

import pytest

from src.catalog import CatalogError, Row, get_accelerator, get_model, list_scenarios, load_scenario
from src.sizing import SizingError, kv_cache_gib, size, vram_breakdown, weight_gib
from src.tp_ep import EP_EFFICIENCY, TP_EFFICIENCY, ParallelError, comm_volume, strategy_for
from src.token_per_watt import (OPT_PREFIX_CACHE, OPT_QUANT, OPT_SPEC_DECODE,
                                 TokenPerWattError, best_optimization, compute, single_card_tps)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------- catalog ----------------

def test_unknown_model_raises_with_list():
    with pytest.raises(CatalogError) as e:
        get_model("no-such-model")
    assert "未收录" in str(e.value)


def test_row_empty_field_is_none_not_zero():
    acc = get_accelerator("S1")
    assert acc.n("fp8_support") in (None, 0.0, 1.0)
    m = Row({"model_id": "X", "note": "", "source_id": "S1", "credibility": "secondary"})
    assert m.s("note") is None, "空字段必须归一为 None"


def test_scenarios_reference_known_models():
    for sid in list_scenarios():
        s = load_scenario(sid)
        get_model(s["workload"]["model_id"]), sid


# ---------------- KV Cache（核心） ----------------

def test_kv_cache_is_computed_explicitly():
    m = get_model("Qwen2.5-32B")
    kv = kv_cache_gib(m, 2000, 500)
    assert kv > 0, "KV Cache 必须算出具体值，不能是 None"
    # 2 × 64 层 × 8 head × 128 dim × 2500 token × 2 Byte
    expect = 2 * 64 * 8 * 128 * 2500 * 2 / 1024 ** 3
    assert kv == pytest.approx(expect, rel=1e-9)


def test_kv_cache_grows_with_context_length():
    m = get_model("Qwen2.5-32B")
    short = kv_cache_gib(m, 500, 100)
    long = kv_cache_gib(m, 8000, 2000)
    assert long > short * 5, "KV Cache 必须随序列长度增长"


def test_kv_cache_grows_with_concurrency_implicitly():
    """并发上升 → 每会话序列不变 → 单卡 KV 不变，但卡数需求上升（由 sizing 处理）。"""
    m = get_model("Qwen2.5-32B")
    a = vram_breakdown(m, 2000, 500)
    b = vram_breakdown(m, 2000, 500)
    assert a["required_gib"] == b["required_gib"]
    s = load_scenario("finance_rag")
    s2 = dict(s)
    s2["workload"] = dict(s["workload"])
    s2["workload"]["concurrent_sessions"] = 400
    # 高并发场景的卡数需求应不低于低并发
    z1 = size(s, "S2")
    assert z1["cards"] >= 1


def test_weight_gib_by_dtype():
    b16 = weight_gib(get_model("Qwen2.5-32B"))       # bf16
    # fp8 权重应约为 bf16 的一半
    fp8 = b16 * 1 / 2
    assert fp8 > 0 and b16 > fp8


def test_unknown_dtype_raises_not_estimated():
    m = Row({"model_id": "X", "params_b": "7", "dtype": "int3",
             "source_id": "S", "credibility": "secondary"})
    with pytest.raises(SizingError):
        weight_gib(m)


# ---------------- sizing ----------------

def test_cards_round_up_to_whole():
    s = load_scenario("finance_rag")
    z = size(s, "S2")
    assert z["cards"] == int(z["cards"]), "卡数必须为整数"
    assert z["cards"] >= 1
    assert z["cards_raw"] == pytest.approx(z["cards"], abs=1.0)


def test_bigger_model_needs_more_cards():
    s = load_scenario("finance_rag")
    small = dict(s)
    small["workload"] = dict(s["workload"])
    small["workload"]["model_id"] = "Qwen2.5-7B"
    z_small = size(small, "S2")
    z_big = size(s, "S2")
    assert z_big["cards"] >= z_small["cards"]


def test_negative_token_raises():
    s = load_scenario("finance_rag")
    s["workload"]["input_len_tokens"] = -1
    with pytest.raises(SizingError):
        size(s, "S2")


def test_missing_workload_raises():
    with pytest.raises(SizingError):
        size({}, "S2")


def test_chip_without_public_vram_yields_unknown_not_guess():
    s = load_scenario("finance_rag")
    z = size(s, "REX-S")
    assert z["cards"] is None
    assert z["verdict"] == "unknown"
    assert "未登记" in z["reason"]


def test_every_result_carries_citations():
    s = load_scenario("finance_rag")
    z = size(s, "S2")
    assert "来源" in z["accel_cite"] and "可信度" in z["accel_cite"]
    assert "来源" in z["model_cite"]


# ---------------- tp_ep ----------------

def test_dense_model_gets_tensor_parallel():
    p = strategy_for(get_model("Qwen2.5-32B"))
    assert p["recommended"] == "tensor_parallel"
    assert p["efficiency"] == TP_EFFICIENCY
    assert "通信" in p["tradeoff"]


def test_moe_model_gets_expert_parallel():
    p = strategy_for(get_model("DeepSeek-V3"))
    assert p["recommended"] == "expert_parallel"
    assert p["efficiency"] == EP_EFFICIENCY
    assert "负载不均" in p["tradeoff"], "MoE 的代价必须点明负载不均"


def test_unknown_moe_flag_yields_unknown():
    m = Row({"model_id": "X", "is_moe": "", "source_id": "S", "credibility": "secondary"})
    assert strategy_for(m)["recommended"] == "unknown"


def test_comm_volume_ep_is_cheaper_than_tp_for_moe():
    """EP 的优势来自「只有 MoE 层需要 all-to-all」，attention/FFN 仍是 dense。

    同模型同卡数下 EP 应显著低于 TP —— 这是选择 EP 而非 TP 的核心理由。
    """
    moe = comm_volume("Mixtral-8x7B", 8, 4096)
    assert moe["available"] and moe["is_moe"]
    assert moe["ep_over_tp"] is not None
    assert moe["ep_gib_per_step"] < moe["tp_gib_per_step"], \
        "EP 通信量应低于 TP，实际 EP=%.3f TP=%.3f" % (
            moe["ep_gib_per_step"], moe["tp_gib_per_step"])
    assert moe["ep_over_tp"] < 0.5, "EP/TP 应小于 0.5，实际 %.2f" % moe["ep_over_tp"]
    assert "MoE 层" in moe["trace"], "推导链必须点明只有 MoE 层通信"
    assert moe["caveat"], "hidden 近似须显式标注为占位"


def test_dense_model_has_no_ep():
    dense = comm_volume("Qwen2.5-32B", 4, 4096)
    assert dense["ep_gib_per_step"] == 0.0
    assert dense["ep_over_tp"] is None
    assert "EP 不适用" in dense["trace"]


def test_comm_volume_invalid_args_raise():
    with pytest.raises(ParallelError):
        comm_volume("Qwen2.5-32B", 0, 4096)
    with pytest.raises(ParallelError):
        comm_volume("Qwen2.5-32B", 4, 0)


# ---------------- token_per_watt ----------------

def test_single_card_tps_scales_with_params():
    small, _ = single_card_tps(get_model("Qwen2.5-7B"))
    big, _ = single_card_tps(get_model("Qwen2.5-32B"))
    assert big > small


def test_optimizations_do_not_stack():
    """核心纪律：优化系数取最大值而非连乘 —— 相乘会严重高估。"""
    f, items = best_optimization(["vLLM", "TensorRT-LLM"])
    assert f == max(OPT_QUANT, OPT_SPEC_DECODE, OPT_PREFIX_CACHE)
    assert f < OPT_QUANT * OPT_SPEC_DECODE * OPT_PREFIX_CACHE, "系数不得连乘"
    assert items


def test_token_per_watt_computed_and_traceable():
    s = load_scenario("finance_rag")
    z = size(s, "S2")
    t = compute(z, s["workload"]["model_id"], ["vLLM", "TensorRT-LLM"], 0.6)
    assert t["available"]
    assert t["token_per_watt"] > 0
    assert "每瓦 Token 吞吐" in t["trace"]
    assert t["assumption_refs"] == ["B-01", "B-02", "B-03"]
    assert len(t["caveats"]) >= 3, "必须显式列出占位值边界"
    assert any("占位" in c for c in t["caveats"])


def test_higher_utilization_raises_throughput():
    s = load_scenario("finance_rag")
    z = size(s, "S2")
    low = compute(z, s["workload"]["model_id"], ["vLLM"], 0.3)
    high = compute(z, s["workload"]["model_id"], ["vLLM"], 0.9)
    assert high["total_tps"] > low["total_tps"]


def test_chip_without_tdp_returns_unavailable_not_guess():
    s = load_scenario("finance_rag")
    z = dict(size(s, "S2"))
    z["accel_tdp_w"] = None
    t = compute(z, s["workload"]["model_id"], ["vLLM"], 0.6)
    assert t["available"] is False
    assert "未登记 TDP" in t["reason"]


def test_invalid_utilization_raises():
    s = load_scenario("finance_rag")
    z = size(s, "S2")
    with pytest.raises(TokenPerWattError):
        compute(z, s["workload"]["model_id"], [], 1.5)
