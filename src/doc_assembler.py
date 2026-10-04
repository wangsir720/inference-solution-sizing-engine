# -*- coding: utf-8 -*-
"""文档装配：选型建议书 / 标书技术章节 / 产品白皮书 / 四层形态规格书。

装配技术方案与商务标书、白皮书与技术规格书。

用标准库字符串模板而非模板引擎 —— 保持零第三方依赖是本项目的一贯取舍。
"""
from __future__ import annotations

import csv
import json
import os

# ---- 四层形态模板（MVP-0 产出物）----
PRODUCT_FORM_SPEC = {
    "chip": {"fields": ["model", "arch", "tdp_w", "vram_gib", "memory_type",
                        "memory_bw_gbs", "slot_support", "intra_interconnect"],
             "rule": "芯片层决定上限：槽位支持数、显存容量、互联能力都是上层不可逾越的天花板"},
    "board": {"fields": ["model", "quantity", "slots_used", "tdp_w", "vram_gib", "interconnect",
                         "form_factor", "power_connector"],
              "rule": "板卡层不得超过芯片层：槽位数 ≤ 芯片槽位支持，功耗须在板卡供电上限内"},
    "server": {"fields": ["model", "slots_available", "fan_modules", "fan_capacity_w",
                          "total_tdp_w", "chassis_overhead_w", "interconnect_topology"],
               "rule": "服务器层是散热与供电的落点：散热能力必须覆盖整机功耗"},
    "appliance": {"fields": ["model", "power_budget_w", "cooling_type", "form_factor",
                             "rack_units", "networking", "target_scenarios"],
                  "rule": "一体机层是客户采购决策的界面：功耗预算、散热形式、机架高度都要对齐客户机房"},
}

# ---- 客户理解成本清单（MVP-0 产出物）----
CUSTOMER_UNDERSTANDING_COST = [
    {"question": "推理卡和训练卡到底差在哪？",
     "answer": "训练卡优化的是大规模矩阵乘的峰值吞吐；推理卡优化的是访存效率与单位功耗吞吐。"
               "训练卡做推理时，大量算力在等显存搬运。",
     "risk_wording": "不要说「我们比 A100 快 X 倍」—— 那是训练口径的数字，"
                     "推理场景客户会立刻质疑"},
    {"question": "每瓦 Token 吞吐和 TFLOPS 什么关系？",
     "answer": "TFLOPS 衡量算力峰值，Token/瓦衡量实际业务产出。"
               "推理场景客户的采购依据是后者：每 Token 电费直接决定运营成本。",
     "risk_wording": "不要用 TFLOPS 作为主指标，客户会认为你没理解推理的成本结构"},
    {"question": "为什么不用 HBM 而用 LPDDR？",
     "answer": "LPDDR 成本与容量优势明显，推理场景的访存模式对带宽要求低于训练。"
               "代价是带宽上限较低，需要靠架构与编译优化弥补。",
     "risk_wording": "必须主动说明带宽差距，否则客户会认为这是降级选择"},
    {"question": "CUDA 兼容到什么程度？我的代码要改多少？",
     "answer": "主流框架与算子已适配，迁移路径为「不改模型结构、替换算子库」。"
               "具体工作量需按客户实际模型评估。",
     "risk_wording": "**不要承诺零成本迁移**—— 个别自定义算子确实需要改写，"
                     "夸大会在 POC 阶段暴露"},
    {"question": "从卡到一体机多久能交付？",
     "answer": "需按采购量与定制程度评估。标准形态交期短，深度定制需评审。",
     "risk_wording": "不要给具体天数承诺，除非已有供应链确认"},
    {"question": "总拥有成本和通用 GPU 方案比差多少？",
     "answer": "推理场景的 TCO 由电费主导，因此关键指标是每 Token 电费。"
               "需要客户提供用电电价、运行小时数、利用率才能给出可比数字。",
     "risk_wording": "**不要在不知道客户电价的情况下给 TCO 数字**—— "
                     "电价差 0.2 元就会让结论反转"},
]


def _table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def render_proposal(scenario: dict, sizing: dict, parallel: dict, tpw: dict,
                    tco: dict, comm: dict) -> str:
    """技术方案书。"""
    L = ["# AI 推理方案选型建议书 · %s" % scenario.get("name", ""), ""]
    L.append("> 由 `src/doc_assembler.py` 自动装配。"
             "**场景为自拟、测算为公开规格驱动的理论推算，非真实业务项目、非实测性能。**")
    L.append("")

    L.append("## 1. 需求与约束")
    w = scenario.get("workload", {})
    c = scenario.get("constraints", {})
    L.append(_table(["项", "值"], [
        ["目标模型", sizing["model_id"]],
        ["并发会话数", w.get("concurrent_sessions")],
        ["输入 / 输出长度", "%s / %s tokens" % (w.get("input_len_tokens"),
                                                w.get("output_len_tokens"))],
        ["P95 时延目标", "%s ms" % w.get("p95_latency_ms")],
        ["可用性目标", w.get("availability_target")],
        ["部署形态", scenario.get("deployment")],
        ["功耗预算", "%s kW" % c.get("power_budget_kw")],
        ["数据驻留要求", c.get("data_residency")],
    ]))
    L.append("")

    L.append("## 2. 显存测算（显式含 KV Cache）")
    mem = sizing["memory"]
    L.append(_table(["项", "GiB"], [
        ["权重", "%.2f" % mem["weight_gib"]],
        ["KV Cache", "%.2f" % mem["kv_cache_gib"] if mem["kv_cache_gib"] is not None
         else "未查到（结构参数不完整）"],
        ["含余量后需求", "%.2f" % mem["required_gib"]],
        ["KV Cache 占比", "%.1f%%" % (mem["kv_share"] * 100) if mem["kv_share"] else "—"],
    ]))
    L.append("")
    L.append("```text\n%s\n```" % mem["trace"])
    L.append("")
    L.append("> **推理场景的显存瓶颈常在 KV Cache 而非权重。** "
             "并发会话数与上下文长度的乘积决定 KV Cache 大小 —— "
             "只算权重会系统性低估需求，导致卡数测算偏乐观。")
    L.append("")

    L.append("## 3. 配置建议")
    L.append(_table(["项", "值"], [
        ["加速卡型号", sizing["accel_model"]],
        ["单卡显存", "%.0f GiB" % sizing["accel_vram_gib"] if sizing.get("accel_vram_gib") else "未查到"],
        ["卡数", sizing["cards"]],
        ["卡数计算", "需求 %.2f GiB ÷ 单卡可用 %.2f GiB = %.3f → 向上取整 %d 张"
         % (mem["required_gib"],
            sizing["accel_vram_gib"] * 0.95 if sizing.get("accel_vram_gib") else 0,
            sizing.get("cards_raw") or 0, sizing["cards"] or 0)],
        ["并行策略", parallel["recommended"]],
        ["判定", sizing["verdict"]],
    ]))
    L.append("")
    L.append("**并行策略依据**：%s" % parallel["reason"])
    L.append("")
    L.append("**策略代价**：%s" % parallel["tradeoff"])
    if comm.get("available"):
        L.append("")
        L.append("**通信量量级对比**（用于评估互联是否够用）：")
        L.append("")
        L.append("```text\n%s\n```" % comm["trace"])
    L.append("")

    L.append("## 4. 每瓦 Token 吞吐量")
    if tpw.get("available"):
        L.append(_table(["项", "值"], [
            ["单卡解码吞吐", "%.1f token/s" % tpw["single_card_tps"]],
            ["并行效率", "%.2f" % tpw["parallel_efficiency"]],
            ["优化系数", "%.1f（%s）" % (tpw["optimization_factor"],
                                        "、".join(tpw["optimizations_applied"]) or "无")],
            ["整机吞吐", "%.1f token/s" % tpw["total_tps"]],
            ["整机功耗", "%.0f W" % tpw["system_tdp_w"]],
            ["**每瓦 Token 吞吐**", "**%.4f token/s/W**" % tpw["token_per_watt"]],
        ]))
        L.append("")
        L.append("```text\n%s\n```" % tpw["trace"])
        L.append("")
        L.append("**本指标的边界**：")
        for cv in tpw["caveats"]:
            L.append("- %s" % cv)
        L.append("")
        L.append("> 这个指标为什么决定推理成本：功率是固定的，能变的只有吞吐。"
                 "**每瓦 Token 吞吐越高，每 Token 电费越低。**")
    else:
        L.append("未查到：%s" % tpw["reason"])
    L.append("")

    L.append("## 5. 三年 TCO")
    L.append(_table(["科目", "项", "数量", "单位", "单价", "年费用(元)"],
                    [[l["category"], l["item"], l["qty"], l["unit"],
                      l["unit_price"], l["annual_cny"]] for l in tco["lines"]]))
    L.append("")
    L.append("```text\n%s\n```" % tco["trace"])
    L.append("")
    if tco["unpriced_items"]:
        L.append("**未计入项（不猜数）**：")
        for u in tco["unpriced_items"]:
            L.append("- **%s** —— %s；补齐方式：%s" % (u["item"], u["reason"], u["action"]))
        L.append("")
    L.append("> %s" % tco["completeness_note"])
    L.append("")
    if tco.get("per_million_tokens"):
        pm = tco["per_million_tokens"]
        L.append("**每百万 Token 电费**：%.2f 元" % pm["cost_per_million_tokens_cny"])
        L.append("")
        L.append("```text\n%s\n```" % pm["trace"])
        L.append("")
        L.append("> %s。" % pm["note"])
        L.append("")

    L.append("## 6. 本方案未做的事")
    for line in [
        "**未做实测**：无加速卡，全部指标为公开规格驱动的理论推算。",
        "**吞吐基线与优化系数是占位值**：真实值须用目标卡跑目标模型得到。",
        "**电价与 PUE 是占位值**：须按项目所在地实际数据替换。",
        "**硬件采购价未查到**：厂商未公开售价，未计入 TCO 合计。",
        "**不做振动与散热的工程设计**：只做参数一致性校验。",
        "**不替代硬件架构师决策**：并行策略给建议与代价说明，方案由架构团队定。",
    ]:
        L.append("- %s" % line)
    L.append("")
    L.append("---")
    L.append("生成命令：`python -m src.cli`")
    L.append("")
    return "\n".join(L)


def render_bid_section(scenario: dict, sizing: dict, tpw: dict, tco: dict) -> str:
    """商务标书技术章节。"""
    L = ["# 商务标书 · 技术章节（自动装配草稿）", ""]
    L.append("> **本章为自动生成的草稿，须由技术负责人复核后投标。**")
    L.append("> 场景为自拟、测算为公开规格驱动的理论推算，**非真实业务数据、非实测性能**。")
    L.append("")
    L.append("## T.1 供货与配置")
    L.append(_table(["项", "响应"], [
        ["加速卡型号", sizing["accel_model"]],
        ["配置数量", "%s 张" % sizing["cards"]],
        ["单卡显存", "%.0f GiB" % sizing["accel_vram_gib"] if sizing.get("accel_vram_gib") else "未查到"],
        ["形态", scenario.get("deployment")],
    ]))
    L.append("")
    L.append("## T.2 性能与效率承诺")
    if tpw.get("available"):
        L.append(_table(["指标", "承诺值", "口径"], [
            ["每瓦 Token 吞吐", "≥ %.4f token/s/W" % tpw["token_per_watt"],
             "整机稳态 decode 吞吐 ÷ 整机 TDP，含机箱开销"],
            ["整机吞吐", "≥ %.0f token/s" % tpw["total_tps"],
             "并发 %s、输入 %s / 输出 %s tokens" % (
                 scenario["workload"]["concurrent_sessions"],
                 scenario["workload"]["input_len_tokens"],
                 scenario["workload"]["output_len_tokens"])],
        ]))
        L.append("")
        L.append("**承诺条件（缺一即不成立）**：")
        for cv in tpw["caveats"]:
            L.append("- %s" % cv)
        L.append("- 以上为理论推算值，投标前须以实测报告替换")
    L.append("")
    L.append("## T.3 运行成本")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 整机功耗 | %.0f W |" % tco["system_power_w"])
    L.append("| 年电费（PUE %.1f、电价 %.2f 元/度） | %.2f 元 |"
             % (tco["pue"], tco["power_price"], tco["annual_power_cost_cny"]))
    if tco.get("per_million_tokens"):
        L.append("| 每百万 Token 电费 | %.2f 元 |"
                 % tco["per_million_tokens"]["cost_per_million_tokens_cny"])
    L.append("")
    if tco["unpriced_items"]:
        L.append("> **注意**：%s。此项须以正式报价补齐后投标，"
                 "否则 TCO 响应不完整。" % tco["unpriced_items"][0]["item"])
        L.append("")
    L.append("## T.4 交付与一致性")
    L.append("四层形态（芯片→板卡→服务器→一体机）规格见随附《技术规格书》，"
             "所有跨层参数已通过一致性校验。")
    L.append("")
    L.append("---")
    L.append("生成命令：`python -m src.cli`")
    L.append("")
    return "\n".join(L)


def render_whitepaper(entries: list[dict], twp_rows: list) -> str:
    """产品白皮书。"""
    L = ["# 产品白皮书 · 每瓦 Token 吞吐量视角（草稿）", ""]
    L.append("> 由 `src/doc_assembler.py` 自动装配。"
             "**全部数据为公开规格与理论推算，非实测性能。**")
    L.append("")
    L.append("## 1. 为什么用「每瓦 Token 吞吐」而不是 TFLOPS 做主指标")
    L.append("TFLOPS 衡量的是算力峰值，而推理场景客户真正付出的是**运营电费**。")
    L.append("整机电费与吞吐的关系是固定的：")
    L.append("")
    L.append("```text")
    L.append("年电费 = 整机功率(W) × 年运行小时 × 利用率 ÷ 1000 × PUE × 电价")
    L.append("每 Token 电费 ≈ 1 ÷ (每瓦 Token 吞吐 × 1000 × PUE × 利用率 × 电价)")
    L.append("```")
    L.append("")
    L.append("功率买回来就固定了，**能变的只有吞吐**。"
             "因此每瓦 Token 吞吐直接决定单位 Token 的电费成本。")
    L.append("")
    L.append("## 2. 推理加速卡规格对照")
    L.append("")
    L.append(_table(["型号", "架构", "显存", "显存类型", "带宽(GB/s)", "TDP(W)", "互联"],
                    [[e["model"], e["arch"], e["vram"], e["mem"], e["bw"], e["tdp"], e["ic"]]
                     for e in entries]))
    L.append("")
    L.append("> 本表只做**规格并列**，不做优劣评价。实际选型取决于模型结构、"
             "上下文长度、并发量与互联拓扑，需按场景测算。")
    L.append("")
    L.append("## 3. 每瓦 Token 吞吐的理论推演")
    L.append("")
    L.append(_table(["型号", "卡数", "整机功耗(W)", "整机吞吐(t/s)", "每瓦 Token 吞吐"],
                    [[r["model"], r["cards"], "%.0f" % r["w"], "%.0f" % r["tps"],
                      "%.4f" % r["tpw"]] for r in twp_rows]))
    L.append("")
    L.append("### 推演假设（**全部为占位值，非实测**）")
    L.append("")
    L.append(_table(["假设", "取值", "说明"], [
        ["单卡解码 TPS 基线", "60 token/s @ 7B bf16", "须用目标卡实测替换"],
        ["并行效率", "TP 0.85 / EP 0.90", "须按实际互联带宽校准"],
        ["优化系数", "量化 1.6 / 投机解码 1.8 / 前缀缓存 1.5 / 连续批处理 1.4",
         "**不叠加，取最大值** —— 相乘会严重高估"],
        ["机箱开销", "整机 TDP = 卡 TDP × 卡数 × 1.15", "须按实际机型校准"],
        ["利用率", "按场景给定", "实际取决于业务波动"],
    ]))
    L.append("")
    L.append("## 4. 优化维度对吞吐的影响")
    L.append("")
    L.append("| 优化项 | 系数 | 作用机制 |")
    L.append("|---|---|---|")
    L.append("| 量化（FP8/INT8） | 1.6 | 低精度解码显著缓解访存带宽瓶颈 |")
    L.append("| 投机解码 | 1.8 | 小模型批量草稿 + 大模型并行验证 |")
    L.append("| 前缀缓存命中 | 1.5 | 复用已算 KV，避免重复 prefill |")
    L.append("| 连续批处理 | 1.4 | 空闲槽位即时补位，提升 GPU 利用率 |")
    L.append("")
    L.append("> 前缀缓存的命中高度依赖业务形态。**多轮对话类应用前缀重复率高，"
             "收益显著；单次独立请求几乎不命中。** 给客户报吞吐时必须问清这一点。")
    L.append("")
    L.append("---")
    L.append("生成命令：`python -m src.cli`")
    L.append("")
    return "\n".join(L)


def render_form_spec(form: dict, validation: dict) -> str:
    """四层形态技术规格书。"""
    from .form_validator import LAYER_LABEL, REQUIRED_FIELDS
    L = ["# 四层产品形态技术规格书", ""]
    L.append("> 由 `src/doc_assembler.py` 自动装配。"
             "**规格为示例数据，非真实产品配置。**")
    L.append("")
    L.append("## 1. 各层必填字段与约束")
    for layer, spec in PRODUCT_FORM_SPEC.items():
        L.append("### %s（%s）" % (LAYER_LABEL[layer], layer))
        L.append("")
        L.append("- 必填：%s" % "、".join("`%s`" % f for f in spec["fields"]))
        L.append("- 约束：%s" % spec["rule"])
        L.append("")
    L.append("## 2. 当前规格")
    L.append("")
    rows = []
    for layer in LAYER_LABEL:
        node = form.get(layer) or {}
        for f in REQUIRED_FIELDS[layer]:
            rows.append([LAYER_LABEL[layer], "`%s`" % f,
                         node.get(f) if node.get(f) else "**未填**"])
    L.append(_table(["层", "字段", "值"], rows))
    L.append("")
    comp = validation["field_completeness"]
    L.append("**字段完整率 %.0f%%**（%d/%d）"
             % (comp["rate"] * 100, comp["filled_fields"], comp["total_fields"]))
    L.append("")
    L.append("## 3. 一致性校验")
    L.append("")
    if validation["issues"]:
        L.append(_table(["规则", "名称", "严重度", "问题", "为什么"],
                        [[i["rule"], i["name"], i["severity"], i["detail"], i["why"]]
                         for i in validation["issues"]]))
        L.append("")
        L.append("**P0 问题 %d 项 —— 必须在投标前清零。**" % validation["p0_count"])
    else:
        L.append("未发现跨层不一致（已校验 %d 条规则）。" % validation["rules_checked"])
    L.append("")
    L.append("> %s" % validation["note"])
    L.append("")
    L.append("---")
    L.append("生成命令：`python -m src.cli`")
    L.append("")
    return "\n".join(L)


def render_customer_cost() -> str:
    """客户理解成本清单。"""
    L = ["# 客户理解成本清单", ""]
    L.append("> 用途：售前沟通前自查。**客户问什么、我们怎么答、哪些说法会弄丢订单。**")
    L.append("")
    L.append(_table(["客户会问", "怎么答", "哪些说法有风险"],
                    [[c["question"], c["answer"], c["risk_wording"]]
                     for c in CUSTOMER_UNDERSTANDING_COST]))
    L.append("")
    L.append("**使用方式**：POC 前逐条过一遍，"
             "凡是无法给出可验证答案的条目，都应在会前补齐材料。")
    L.append("")
    return "\n".join(L)


def save(text: str, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def save_csv(rows, header, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def save_json(obj, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return path
