# -*- coding: utf-8 -*-
"""四层产品形态一致性校验：芯片 → 板卡 → 服务器 → 一体机。

对客户理解成本、技术/商务、振动设计、散热设计、模块、服务器、
超节点以及一体机产品形态形成**一致性标准**。

## 这个模块解决什么真实问题

四层形态是**由下往上推导**的：芯片能插几槽 → 板卡能插几张 → 机箱能插几块板 →
一体机的功耗与散热够不够。**任何一层改了，上层的数字就全错了。**

典型的一致性断裂：

| 类别 | 例子 | 后果 |
|---|---|---|
| 槽位不一致 | 芯片支持 8 槽互联，板卡只做 4 槽 | 芯片能力浪费，售价白高 |
| 功耗预算不一致 | 一体机按 4 卡 × 400W 设计，板卡实际 450W | 散热压不住，降频或不达标 |
| 散热容量不一致 | 4 个 700W 模组配 2 个风扇模组 | 直接过热保护 |
| 显存与模型不匹配 | 标称 144GB 但只按 96GB 做规划 | 客户按标称采购，交付时跑不了目标模型 |
| 互联能力不一致 | 超节点用 PCIe 而非专用互联 | 并行效率达不到设计值 |

**这些错误在投标阶段发现是改图纸，交付后���现是退货。**
"""
from __future__ import annotations

# 四层形态的必填字段
FORM_LAYERS = ["chip", "board", "server", "appliance"]
LAYER_LABEL = {"chip": "芯片", "board": "板卡", "server": "服务器", "appliance": "一体机"}

# 一致性规则：每条给出检查项、判定逻辑、依据
CONSISTENCY_RULES = [
    {
        "id": "C-01",
        "name": "槽位数量逐层收敛",
        "layers": ["chip", "board", "server"],
        "check": "chip.slot_support >= board.slots_used >= server.slots_available",
        "why": "板卡槽位不得超过芯片支持的槽位，服务器可用槽位不得少于板卡数量",
        "severity": "P0",
    },
    {
        "id": "C-02",
        "name": "功耗预算逐层收敛",
        "layers": ["chip", "board", "server", "appliance"],
        "check": "Σ(板卡TDP) + 机箱开销 <= appliance.power_budget",
        "why": "一体机的功耗预算由散热能力决定，超出即无法稳定运行",
        "severity": "P0",
    },
    {
        "id": "C-03",
        "name": "散热容量匹配",
        "layers": ["server", "appliance"],
        "check": "server.fan_modules * server.fan_capacity_w >= server.total_tdp_w",
        "why": "散热能力不足会触发降频，实测性能达不到标称值",
        "severity": "P0",
    },
    {
        "id": "C-04",
        "name": "显存容量与目标模型匹配",
        "layers": ["board", "appliance"],
        "check": "board.vram_gib * boards >= target_model_required_gib",
        "why": "标称显存与实际可用显存的差距是最常见的交付期返工原因",
        "severity": "P1",
    },
    {
        "id": "C-05",
        "name": "互联方式与并行策略匹配",
        "layers": ["board", "appliance"],
        "check": "board.interconnect == strategy_required_interconnect",
        "why": "张量并行需要高带宽互联，用 PCIe 会让并行效率达不到设计值",
        "severity": "P1",
    },
]


class FormValidationError(Exception):
    pass


def _get(form: dict, layer: str, field: str):
    node = form.get(layer)
    if not isinstance(node, dict):
        return None
    v = node.get(field)
    if v is None or v == "" or v == "-":
        return None
    return v


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def validate(form: dict) -> dict:
    """执行全部一致性规则，返回问题清单。"""
    if not isinstance(form, dict):
        raise FormValidationError("形态规格必须是字典")
    for layer in FORM_LAYERS:
        if layer not in form:
            raise FormValidationError("形态规格缺少层：%s" % LAYER_LABEL[layer])

    issues = []
    for rule in CONSISTENCY_RULES:
        if rule["id"] == "C-01":
            # 注意字段名：板卡用「单板槽位数」slots_used，服务器用「可用槽位」slots_available，
            # 芯片用「槽位支持数」slot_support。三者语义不同，不能互相顶替。
            chip_slot = _num(_get(form, "chip", "slot_support"))
            board_used = _num(_get(form, "board", "slots_used"))
            srv_avail = _num(_get(form, "server", "slots_available"))
            if None not in (chip_slot, board_used) and board_used > chip_slot:
                issues.append(_issue(rule, "板卡使用 %d 槽，芯片仅支持 %d 槽"
                                 % (int(board_used), int(chip_slot))))
            if None not in (board_used, srv_avail) and srv_avail < board_used:
                issues.append(_issue(rule, "服务器可用 %d 槽，少于单板所需 %d 槽"
                                 % (int(srv_avail), int(board_used))))
            # 板卡数量 × 单板槽位 不应超过服务器可用槽位
            board_qty = _num(_get(form, "board", "quantity"))
            if None not in (board_qty, board_used, srv_avail) \
                    and board_qty * board_used > srv_avail:
                issues.append(_issue(
                    rule, "%d 块板卡 × %d 槽 = %d 槽需求，超过服务器可用 %d 槽"
                    % (int(board_qty), int(board_used),
                       int(board_qty * board_used), int(srv_avail))))

        elif rule["id"] == "C-02":
            board_tdp = _num(_get(form, "board", "tdp_w"))
            board_qty = _num(_get(form, "board", "quantity"))
            srv_chassis = _num(_get(form, "server", "chassis_overhead_w")) or 0.0
            budget = _num(_get(form, "appliance", "power_budget_w"))
            if None not in (board_tdp, board_qty) and budget is not None:
                total = board_tdp * board_qty + srv_chassis
                if total > budget:
                    issues.append(_issue(
                        rule, "整机功耗 %.0f W（%d 卡 × %.0f W + 机箱 %.0f W）"
                        "超出预算 %.0f W" % (total, int(board_qty), board_tdp,
                                            srv_chassis, budget)))

        elif rule["id"] == "C-03":
            fans = _num(_get(form, "server", "fan_modules"))
            cap = _num(_get(form, "server", "fan_capacity_w"))
            tdp = _num(_get(form, "server", "total_tdp_w"))
            if None not in (fans, cap, tdp) and fans * cap < tdp:
                issues.append(_issue(rule, "散热能力 %.0f W（%d 模组 × %.0f W）"
                                 "低于整机功耗 %.0f W"
                                 % (fans * cap, int(fans), cap, tdp)))

        elif rule["id"] == "C-04":
            vram = _num(_get(form, "board", "vram_gib"))
            qty = _num(_get(form, "board", "quantity"))
            need = _num(form.get("target_model_required_gib"))
            if None not in (vram, qty) and need is not None and vram * qty < need:
                issues.append(_issue(rule, "整机显存 %.0f GiB（%d × %.0f GiB）"
                                 "低于目标模型需求 %.0f GiB"
                                 % (vram * qty, int(qty), vram, need)))

        elif rule["id"] == "C-05":
            actual = _get(form, "board", "interconnect")
            required = form.get("strategy_required_interconnect")
            if actual and required and actual != required:
                issues.append(_issue(rule, "板卡互联方式为 %s，并行策略要求 %s"
                                 % (actual, required)))

    severity_order = {"P0": 0, "P1": 1, "P2": 2}
    issues.sort(key=lambda i: (severity_order.get(i["severity"], 3), i["rule"]))

    return {
        "issues": issues,
        "issue_count": len(issues),
        "p0_count": sum(1 for i in issues if i["severity"] == "P0"),
        "rules_checked": len(CONSISTENCY_RULES),
        "field_completeness": completeness(form),
        "note": "只做参数一致性校验，不做振动/散热的工程设计（见 assumptions.md D 节）",
    }


def _issue(rule: dict, detail: str) -> dict:
    return {
        "rule": rule["id"],
        "name": rule["name"],
        "severity": rule["severity"],
        "detail": detail,
        "why": rule["why"],
    }


# 各层必填字段（用于完整率计算）
REQUIRED_FIELDS = {
    "chip": ["model", "tdp_w", "vram_gib", "slot_support", "arch"],
    "board": ["model", "quantity", "slots_used", "tdp_w", "vram_gib", "interconnect"],
    "server": ["model", "slots_available", "fan_modules", "fan_capacity_w",
               "total_tdp_w", "chassis_overhead_w"],
    "appliance": ["model", "power_budget_w", "cooling_type", "form_factor"],
}


def _blank(v) -> bool:
    """字段是否为空。数值 0 不算空 —— 0 是个具体取值，不是缺失。"""
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip() or v.strip() == "-"
    return False


def completeness(form: dict) -> dict:
    """四层形态的字段完整率。"""
    per_layer = {}
    total, filled = 0, 0
    for layer, fields in REQUIRED_FIELDS.items():
        node = form.get(layer) or {}
        miss = [f for f in fields if _blank(node.get(f))]
        per_layer[layer] = {
            "total": len(fields), "filled": len(fields) - len(miss),
            "missing": miss,
        }
        total += len(fields)
        filled += len(fields) - len(miss)
    return {
        "per_layer": per_layer,
        "total_fields": total,
        "filled_fields": filled,
        "rate": round(filled / total, 4) if total else 0.0,
    }
