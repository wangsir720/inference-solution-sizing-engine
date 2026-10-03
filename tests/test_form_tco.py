# -*- coding: utf-8 -*-
"""form_validator / tco / 端到端测试。

核心验证点：**一致性校验必须真能发现跨层冲突**（用故意做错的规格证明）、
**硬件价未查到时不凑总额**。
"""
import copy
import json
import os

import pytest

from src import cli
from src.form_validator import (CONSISTENCY_RULES, REQUIRED_FIELDS, FormValidationError,
                                completeness, validate)
from src.sizing import size
from src.tco import TcoError, compute as tco_compute
from src.token_per_watt import compute as tpw_compute

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOOD_FORM = {
    "chip": {"model": "S2", "arch": "DSA", "tdp_w": 400, "vram_gib": 64,
             "memory_type": "LPDDR", "memory_bw_gbs": 1200,
             "slot_support": 8, "intra_interconnect": "专用互联"},
    "board": {"model": "S2-卡", "quantity": 4, "slots_used": 1, "tdp_w": 400,
              "vram_gib": 64, "interconnect": "专用互联", "form_factor": "双宽全高",
              "power_connector": "3000W"},
    "server": {"model": "4U", "slots_available": 4, "fan_modules": 4,
               "fan_capacity_w": 500, "total_tdp_w": 1600, "chassis_overhead_w": 200,
               "interconnect_topology": "全互联"},
    "appliance": {"model": "4U 一体机", "power_budget_w": 2000, "cooling_type": "风冷",
                  "form_factor": "4U 机架", "rack_units": 4, "networking": "双 200G",
                  "target_scenarios": "金融 RAG"},
    "target_model_required_gib": 250.0,
    "strategy_required_interconnect": "专用互联",
}


def _form():
    return copy.deepcopy(GOOD_FORM)


# ---------------- 一致性校验（核心） ----------------

def test_validator_finds_slot_mismatch():
    """芯片支持 8 槽，但服务器只提供 4 槽 —— 上层配置不得超过芯片能力。"""
    f = _form()
    f["board"]["quantity"] = 6        # 插 6 块板卡
    f["server"]["slots_available"] = 4  # 服务器只有 4 槽 -> 装不下
    r = validate(f)
    assert any(i["rule"] == "C-01" for i in r["issues"]), \
        "板卡数超过服务器可用槽位必须被发现，实际问题：%s" % r["issues"]


def test_validator_finds_board_exceeding_chip_slots():
    f = _form()
    f["chip"]["slot_support"] = 2   # 芯片只支持 2 槽
    f["board"]["slots_used"] = 4    # 单板要占 4 槽 -> 超出芯片能力
    r = validate(f)
    assert any(i["rule"] == "C-01" for i in r["issues"])


def test_validator_finds_power_budget_overflow():
    f = _form()
    f["appliance"]["power_budget_w"] = 1000  # 实际需要 4×400+200=1800
    r = validate(f)
    issues = [i for i in r["issues"] if i["rule"] == "C-02"]
    assert issues, "功耗超预算必须被发现"
    assert issues[0]["severity"] == "P0"


def test_validator_finds_cooling_shortfall():
    f = _form()
    f["server"]["fan_modules"] = 2       # 2 × 500 = 1000W < 1600W
    r = validate(f)
    assert any(i["rule"] == "C-03" for i in r["issues"])


def test_validator_finds_vram_shortfall():
    f = _form()
    f["target_model_required_gib"] = 500.0   # 实有 4×64 = 256
    r = validate(f)
    assert any(i["rule"] == "C-04" for i in r["issues"])


def test_validator_finds_interconnect_mismatch():
    f = _form()
    f["board"]["interconnect"] = "PCIe 5.0"
    r = validate(f)
    assert any(i["rule"] == "C-05" for i in r["issues"])


def test_good_form_has_no_issues():
    r = validate(_form())
    assert r["issue_count"] == 0, r["issues"]
    assert r["p0_count"] == 0


def test_at_least_three_distinct_issue_types_detectable():
    """验收标准：校验器至少能发现 3 类跨层不一致。"""
    detected = set()
    for mutate in (
        lambda f: (f["board"].__setitem__("quantity", 6),
                   f["server"].__setitem__("slots_available", 4)),
        lambda f: f["appliance"].__setitem__("power_budget_w", 1000),
        lambda f: f["server"].__setitem__("fan_modules", 2),
        lambda f: f.__setitem__("target_model_required_gib", 500.0),
        lambda f: f["board"].__setitem__("interconnect", "PCIe 5.0"),
    ):
        f = _form()
        mutate(f)
        detected |= {i["rule"] for i in validate(f)["issues"]}
    assert len(detected) >= 3, "应能发现至少 3 类问题，实际 %d" % len(detected)


def test_missing_layer_raises():
    f = _form()
    del f["appliance"]
    with pytest.raises(FormValidationError):
        validate(f)


def test_completeness_counts_zero_as_filled():
    f = _form()
    f["chip"]["memory_bw_gbs"] = 0
    c = completeness(f)
    assert "memory_bw_gbs" not in c["per_layer"]["chip"]["missing"], \
        "0 是具体取值，不算缺失"


def test_all_five_rules_have_why():
    for rule in CONSISTENCY_RULES:
        assert rule["why"] and rule["name"] and rule["severity"]
    assert len(CONSISTENCY_RULES) == 5


# ---------------- TCO ----------------

def _tco(**kw):
    s = cli.load_scenario("finance_rag")
    z = size(s, "S2")
    t = tpw_compute(z, s["workload"]["model_id"], ["vLLM", "TensorRT-LLM"], 0.6)
    return tco_compute(z, t, **kw)


def test_hardware_price_missing_is_not_filled_with_zero():
    """核心纪律：硬件价未公开就不计入合计，不按 0 元凑总额。"""
    r = _tco(util=0.6)
    assert r["unpriced_items"], "硬件价未查到时必须列入未计入项"
    assert "加速卡" in r["unpriced_items"][0]["item"]
    assert "不完整" in r["completeness_note"]


def test_power_cost_is_listed_separately():
    r = _tco(util=0.6)
    categories = [l["category"] for l in r["lines"]]
    assert "运营" in categories
    items = [l["item"] for l in r["lines"]]
    assert any("电费" in i for i in items), "电费必须单列"
    assert r["annual_power_cost_cny"] > 0


def test_three_year_is_twelve_times_annual():
    r = _tco(util=0.6)
    assert r["three_year_known_cny"] == pytest.approx(r["annual_known_cny"] * 3, rel=0.01)


def test_hardware_price_included_when_given():
    r = _tco(util=0.6, hardware_price=50000.0)
    assert not r["unpriced_items"], "给了报价就不该再列未计入"
    assert any(l["category"] == "资本" for l in r["lines"])


def test_per_million_token_cost_computed():
    r = _tco(util=0.6)
    pm = r["per_million_tokens"]
    assert pm and pm["cost_per_million_tokens_cny"] > 0
    assert "PUE" in pm["trace"]
    assert "不含硬件折旧" in pm["note"]


def test_higher_power_price_raises_cost():
    low = _tco(util=0.6, power_price=0.5)
    high = _tco(util=0.6, power_price=1.0)
    assert high["annual_power_cost_cny"] > low["annual_power_cost_cny"]


def test_invalid_inputs_raise():
    for kw in ({"power_price": 0}, {"pue": 0}, {"util": 2.0}, {"util": 0}):
        with pytest.raises(TcoError):
            _tco(**kw)


# ---------------- 端到端 ----------------

def _doc(name, sid="finance_rag"):
    assert cli.main(["--scenario", sid]) == 0
    with open(os.path.join(cli.OUTPUT_DIR, sid, name), encoding="utf-8") as f:
        return f.read()


def test_all_artifacts_generated():
    assert cli.main(["--all"]) == 0
    for sid in cli.list_scenarios():
        for name in cli.ARTIFACTS:
            p = os.path.join(cli.OUTPUT_DIR, sid, name)
            assert os.path.isfile(p), "缺少 %s/%s" % (sid, name)
            assert os.path.getsize(p) > 0, "产物为空 %s/%s" % (sid, name)


def test_proposal_has_required_sections():
    doc = _doc("solution_proposal.md")
    for h in ("## 1. 需求与约束", "## 2. 显存测算", "## 3. 配置建议",
              "## 4. 每瓦 Token 吞吐量", "## 5. 三年 TCO", "## 6. 本方案未做的事"):
        assert h in doc, "方案书缺少章节：%s" % h


def test_proposal_shows_kv_cache_explicitly():
    doc = _doc("solution_proposal.md")
    assert "KV Cache" in doc
    assert "推理场景的显存瓶颈常在 KV Cache" in doc


def test_token_per_watt_section_states_caveats():
    doc = _doc("solution_proposal.md")
    assert "占位值" in doc, "必须声明吞吐基线是占位值"
    assert "未实测" in doc or "非实测" in doc


def test_bid_section_declares_draft_status():
    doc = _doc("bid_section.md")
    assert "须由技术负责人复核后投标" in doc
    assert "未计入" in doc or "不完整" in doc


def test_form_spec_backfilled_from_sizing():
    """规格书的卡数必须来自测算结果，否则规格书与方案书自相矛盾。"""
    cli.main(["--scenario", "finance_rag"])
    with open(os.path.join(cli.OUTPUT_DIR, "finance_rag", "result.json"),
              encoding="utf-8") as f:
        r = json.load(f)
    assert r["form"]["board"]["quantity"] == r["sizing"]["cards"]
    assert r["form"]["board"]["tdp_w"] == r["sizing"]["accel_tdp_w"]


def test_whitepaper_lists_competitors_without_ranking():
    cli.main(["--compare"])
    with open(os.path.join(cli.OUTPUT_DIR, "whitepaper.md"), encoding="utf-8") as f:
        doc = f.read()
    for m in ("S2", "S3", "A100-80G", "L40S", "MI300X"):
        assert m in doc, "白皮书缺少型号 %s" % m
    assert "不做优劣评价" in doc
    for banned in ("优于", "领先于", "吊打", "远超"):
        assert banned not in doc, "出现竞品贬低表述：%s" % banned


def test_customer_understanding_cost_generated():
    doc = _doc("customer_understanding_cost.md")
    assert "客户会问" in doc and "有风险" in doc
    assert "不要" in doc, "应给出禁忌表述"


def test_reports_declare_synthetic_scenario():
    cli.main(["--all"])
    for sid in cli.list_scenarios():
        for name in ("solution_proposal.md", "bid_section.md", "form_spec.md"):
            doc = _doc(name, sid)
            assert "自拟" in doc or "示例数据" in doc, "%s/%s 缺自拟声明" % (sid, name)


def test_no_sensitive_fields():
    cli.main(["--all"])
    banned = ["合同金额", "客户名称", "内部报价", "账号密码", "有限公司"]
    for sid in cli.list_scenarios():
        for name in ("solution_proposal.md", "bid_section.md", "form_spec.md"):
            doc = _doc(name, sid)
            for b in banned:
                assert b not in doc, "%s/%s 出现 %s" % (sid, name, b)


def test_generation_is_reproducible():
    cli.main(["--scenario", "finance_rag"])
    with open(os.path.join(cli.OUTPUT_DIR, "finance_rag", "result.json"),
              encoding="utf-8") as f:
        a = json.load(f)
    cli.main(["--scenario", "finance_rag"])
    with open(os.path.join(cli.OUTPUT_DIR, "finance_rag", "result.json"),
              encoding="utf-8") as f:
        b = json.load(f)
    assert a["token_per_watt"] == b["token_per_watt"]
    assert a["sizing"]["cards"] == b["sizing"]["cards"]
