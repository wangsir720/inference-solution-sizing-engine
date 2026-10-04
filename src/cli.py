# -*- coding: utf-8 -*-
"""命令行入口：场景 → 选型建议书 + 标书技术章节 + 白皮书 + 形态规格书。

    python -m src.cli              # 全部场景
    python -m src.cli --scenario finance_rag
    python -m src.cli --accel S2   # 指定加速卡（默认 S2）
    python -m src.cli --compare    # 竞品规格对照（不做优劣评价）
"""
from __future__ import annotations

import argparse
import os
import sys

from .catalog import list_scenarios, load_accelerators, load_scenario
from .doc_assembler import (render_bid_section, render_customer_cost, render_form_spec,
                            render_proposal, render_whitepaper, save, save_json)
from .form_validator import validate
from .sizing import size
from .tco import compute as tco_compute
from .token_per_watt import compute as tpw_compute
from .tp_ep import comm_volume, strategy_for
from .catalog import get_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
DEFAULT_ACCEL = "S2"
DEFAULT_ENGINES = ["vLLM", "TensorRT-LLM"]
DEFAULT_UTIL = 0.6

ARTIFACTS = ["solution_proposal.md", "bid_section.md", "whitepaper.md",
             "form_spec.md", "result.json"]

# 示例四层形态规格（示例数据，非真实产品配置）
DEMO_FORM = {
    "chip": {"model": "S2", "arch": "DSA", "tdp_w": 400, "vram_gib": 64,
             "memory_type": "LPDDR", "memory_bw_gbs": 1200,
             "slot_support": 8, "intra_interconnect": "专用互联"},
    "board": {"model": "S2-加速卡", "quantity": 4, "slots_used": 1, "tdp_w": 400, "vram_gib": 64,
              "interconnect": "专用互联", "form_factor": "双宽全高", "power_connector": "3000W"},
    "server": {"model": "4U 机架服务器", "slots_available": 4, "fan_modules": 4,
               "fan_capacity_w": 500, "total_tdp_w": 1600, "chassis_overhead_w": 200,
               "interconnect_topology": "全互联"},
    "appliance": {"model": "4U 一体机", "power_budget_w": 2000, "cooling_type": "风冷",
                  "form_factor": "4U 机架", "rack_units": 4, "networking": "双 200G",
                  "target_scenarios": "金融 RAG 推理"},
    "target_model_required_gib": 70.0,
    "strategy_required_interconnect": "专用互联",
}


def _run_one(sid: str, accel: str, engines: list[str], util: float) -> dict:
    scenario = load_scenario(sid)
    z = size(scenario, accel)
    model = get_model(scenario["workload"]["model_id"])
    parallel = strategy_for(model)
    comm = comm_volume(scenario["workload"]["model_id"], max(z.get("cards") or 1, 1),
                       scenario["workload"]["concurrent_sessions"]
                       * scenario["workload"]["input_len_tokens"]) \
        if z.get("cards") else {"available": False, "reason": "卡数未定"}
    tpw = tpw_compute(z, scenario["workload"]["model_id"], engines, util)
    tco = tco_compute(z, tpw, util)

    # 四层形态规格：把测算出的卡数与功耗回填，保证规格与测算一致
    form = _form_from_sizing(DEMO_FORM, z, scenario)
    validation = validate(form)

    base = os.path.join(OUTPUT_DIR, sid)
    save(render_proposal(scenario, z, parallel, tpw, tco, comm),
         os.path.join(base, "solution_proposal.md"))
    save(render_bid_section(scenario, z, tpw, tco),
         os.path.join(base, "bid_section.md"))
    save(render_form_spec(form, validation), os.path.join(base, "form_spec.md"))
    save(render_customer_cost(), os.path.join(base, "customer_understanding_cost.md"))
    # 白皮书按场景生成：本场景所选型号的规格对照 + 每瓦 Token 推演
    save(render_whitepaper(_spec_rows([accel]), _tpw_rows([(scenario, accel, engines, util)])),
         os.path.join(base, "whitepaper.md"))
    save_json({"scenario": scenario, "sizing": z, "parallel": parallel,
               "comm": comm, "token_per_watt": tpw, "tco": tco,
               "form": form, "validation": validation},
              os.path.join(base, "result.json"))
    return {"scenario": scenario, "sizing": z, "parallel": parallel, "comm": comm,
            "token_per_watt": tpw, "tco": tco, "form": form, "validation": validation}


def _form_from_sizing(base: dict, z: dict, scenario: dict) -> dict:
    """把测算结果回填到四层形态规格。

    这一步很关键：**规格书里的卡数与功耗必须来自测算结果**，
    否则规格书与方案书会自相矛盾 —— 那正是四层形态一致性要防的问题。
    """
    form = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    cards = z.get("cards")
    if cards:
        form["board"]["quantity"] = cards
    tdp = z.get("accel_tdp_w")
    if tdp:
        form["board"]["tdp_w"] = tdp
        form["chip"]["tdp_w"] = tdp
        total = tdp * cards
        chassis = form["server"].get("chassis_overhead_w") or 0
        form["server"]["total_tdp_w"] = total
        form["appliance"]["power_budget_w"] = total + chassis
        need_fans = max(1, -(-(total + chassis) // (form["server"].get("fan_capacity_w") or 500)))
        form["server"]["fan_modules"] = need_fans
    if z.get("memory", {}).get("required_gib"):
        form["target_model_required_gib"] = z["memory"]["required_gib"]
    form["appliance"]["target_scenarios"] = scenario.get("name", "")
    return form


def _spec_rows(model_ids: list[str]) -> list[dict]:
    """指定型号的规格行（供白皮书用）。查不到的字段显式标「未查到」。"""
    accs = load_accelerators()
    entries = []
    for mid in model_ids:
        a = accs.get(mid)
        if not a:
            continue
        entries.append({
            "model": mid, "arch": a.s("arch") or "未查到",
            "vram": "%.0f GiB" % a.n("vram_gib") if a.n("vram_gib") else "未查到",
            "mem": a.s("memory_type") or "未查到",
            "bw": "%.0f" % a.n("memory_bw_gbs") if a.n("memory_bw_gbs") else "未查到",
            "tdp": "%.0f" % a.n("tdp_w") if a.n("tdp_w") else "未查到",
            "ic": a.s("intra_interconnect") or "未查到",
        })
    return entries


def _tpw_rows(items: list[tuple]) -> list[dict]:
    """每瓦 Token 推演行。卡数或 TDP 缺失时如实标 0，不估算。"""
    rows = []
    for scenario, accel_id, engines, util in items:
        z = size(scenario, accel_id)
        row = {"model": accel_id, "cards": z.get("cards") or 0, "w": 0, "tps": 0, "tpw": 0}
        if z.get("cards"):
            t = tpw_compute(z, scenario["workload"]["model_id"], engines, util)
            if t.get("available"):
                row.update({"w": t["system_tdp_w"], "tps": t["total_tps"],
                            "tpw": t["token_per_watt"]})
        rows.append(row)
    return rows


def _whitepaper_inputs():
    """兼容旧调用：全部在售/对照型号的规格行。"""
    return _spec_rows(["S2", "S3", "A100-80G", "L40S", "MI300X"]), []


def _compare() -> int:
    """竞品规格对照 + 每瓦 Token 推演。只做并列，不做优劣评价。"""
    entries = _spec_rows(["S2", "S3", "A100-80G", "L40S", "MI300X"])
    s = load_scenario("finance_rag")
    items = [(s, mid, DEFAULT_ENGINES, DEFAULT_UTIL)
             for mid in ("S2", "S3", "A100-80G", "L40S", "MI300X")]
    rows = _tpw_rows(items)
    save(render_whitepaper(entries, rows), os.path.join(OUTPUT_DIR, "whitepaper.md"))
    print("已生成：output/whitepaper.md")
    print("对照型号数 %d，其中完成每瓦吞吐推演 %d 个"
          % (len(entries), sum(1 for r in rows if r["tpw"] > 0)))
    return 0


def _summary(res: dict) -> None:
    z, tpw, tco, v = res["sizing"], res["token_per_watt"], res["tco"], res["validation"]
    print("场景 %s ｜ 加速卡 %s" % (z["scenario_id"], z["accel_model"]))
    print("  卡数 %s ｜ 判定 %s" % (z.get("cards"), z.get("verdict")))
    mem = z["memory"]
    print("  显存 权重 %.2f + KV Cache %s = 需求 %.2f GiB"
          % (mem["weight_gib"],
             "%.2f" % mem["kv_cache_gib"] if mem["kv_cache_gib"] is not None else "未查到",
             mem["required_gib"]))
    print("  并行策略 %s" % res["parallel"]["recommended"])
    if tpw.get("available"):
        print("  每瓦 Token 吞吐 %.4f token/s/W（整机 %.0f W / %.0f token/s）"
              % (tpw["token_per_watt"], tpw["system_tdp_w"], tpw["total_tps"]))
    print("  年电费 %.2f 元（已计入部分三年 %.2f 元）"
          % (tco["annual_power_cost_cny"], tco["three_year_known_cny"]))
    if tco.get("per_million_tokens"):
        print("  每百万 Token 电费 %.2f 元"
              % tco["per_million_tokens"]["cost_per_million_tokens_cny"])
    if tco["unpriced_items"]:
        print("  未计入：%s" % "、".join(u["item"] for u in tco["unpriced_items"]))
    print("  形态一致性：%d 条问题（P0 %d 项），字段完整率 %.0f%%"
          % (v["issue_count"], v["p0_count"], v["field_completeness"]["rate"] * 100))
    print("  已生成：output/%s/{%s}" % (z["scenario_id"], ", ".join(ARTIFACTS)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.cli",
                                 description="AI 推理方案选型与成本测算引擎")
    ap.add_argument("--scenario")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--accel", default=DEFAULT_ACCEL, help="加速卡型号，默认 S2")
    ap.add_argument("--compare", action="store_true", help="竞品规格对照与每瓦吞吐推演")
    ap.add_argument("--util", type=float, default=DEFAULT_UTIL, help="利用率，默认 0.6")
    args = ap.parse_args(argv)

    if args.compare:
        return _compare()
    if args.list or not (args.scenario or args.all):
        print("可用场景：")
        for s in list_scenarios():
            print("  - %s" % s)
        print("可用加速卡：%s" % "、".join(sorted(load_accelerators())))
        return 0

    try:
        targets = list_scenarios() if args.all else [args.scenario]
        for sid in targets:
            _summary(_run_one(sid, args.accel, DEFAULT_ENGINES, args.util))
    except Exception as e:
        print("执行失败：%s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
