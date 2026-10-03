# -*- coding: utf-8 -*-
"""目录加载：模型规格 / 加速卡规格 / 推理引擎指标。

与本项目其他部分一致的三条纪律：

1. **缺列、缺 source_id、未知 credibility 直接抛错**，不静默兜底。
2. **空字段归一为 None（未查到）**，绝不返回 0 顶替 —— 0 是个具体数值，
   会污染下游所有计算。
3. 数据表外置为 CSV，改数据不用改代码。
"""
from __future__ import annotations

import csv
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_CSV = os.path.join(DATA_DIR, "models", "model_specs.csv")
ACCEL_CSV = os.path.join(DATA_DIR, "hardware", "accelerator_specs.csv")
ENGINE_CSV = os.path.join(DATA_DIR, "frameworks", "engine_matrix.csv")
SCENARIO_DIR = os.path.join(DATA_DIR, "scenarios")

CREDIBILITY_ORDER = {"unverified": 0, "secondary": 1, "official": 2}


class CatalogError(Exception):
    """目录数据缺失或格式错误。"""


def _read(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise CatalogError("数据文件不存在：%s" % path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise CatalogError("数据文件为空：%s" % path)
    for r in rows:
        for req in ("source_id", "credibility"):
            if not (r.get(req) or "").strip():
                raise CatalogError("数据行缺少 %s：%s" % (req, path))
        if r["credibility"].strip() not in CREDIBILITY_ORDER:
            raise CatalogError("未知 credibility %r：%s" % (r["credibility"].strip(), path))
    return rows


def _index(rows: list[dict], key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        k = r[key].strip()
        if k in out:
            prev = out[k]
            if CREDIBILITY_ORDER[r["credibility"].strip()] > \
                    CREDIBILITY_ORDER[prev["credibility"].strip()]:
                out[k] = r
            continue
        out[k] = r
    return out


class Row:
    """目录行包装：空字段一律归一为 None。"""

    def __init__(self, data: dict):
        self.data = data

    @property
    def key(self) -> str:
        return (self.data.get("model_id") or self.data.get("accel_model")
                or self.data.get("engine") or "").strip()

    @property
    def source_id(self) -> str:
        return (self.data.get("source_id") or "").strip()

    @property
    def credibility(self) -> str:
        return (self.data.get("credibility") or "").strip()

    def s(self, field: str):
        """取字符串字段；空串归一为 None。"""
        v = (self.data.get(field) or "").strip()
        return v if v and v != "-" else None

    def n(self, field: str):
        """取数值字段；缺失返回 None。"""
        v = self.s(field)
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return None

    def b(self, field: str) -> bool | None:
        v = self.s(field)
        if v is None:
            return None
        return v.lower() == "true"

    def cite(self) -> str:
        return "%s（来源 %s，可信度 %s）" % (self.key, self.source_id, self.credibility)


def load_models() -> dict[str, Row]:
    return {k: Row(v) for k, v in _index(_read(MODEL_CSV), "model_id").items()}


def load_accelerators() -> dict[str, Row]:
    return {k: Row(v) for k, v in _index(_read(ACCEL_CSV), "accel_model").items()}


def load_engines() -> dict[str, Row]:
    return {k: Row(v) for k, v in _index(_read(ENGINE_CSV), "engine").items()}


def get_model(model_id: str) -> Row:
    m = load_models()
    if model_id not in m:
        raise CatalogError("未收录模型 %r；已收录：%s" % (model_id, ", ".join(sorted(m))))
    return m[model_id]


def get_accelerator(model_id: str) -> Row:
    a = load_accelerators()
    if model_id not in a:
        raise CatalogError("未收录加速卡 %r；已收录：%s" % (model_id, ", ".join(sorted(a))))
    return a[model_id]


def list_scenarios() -> list[str]:
    if not os.path.isdir(SCENARIO_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(SCENARIO_DIR) if f.endswith(".json"))


def load_scenario(sid: str) -> dict:
    path = os.path.join(SCENARIO_DIR, "%s.json" % sid)
    if not os.path.isfile(path):
        raise CatalogError("场景 %r 不存在；可用：%s" % (sid, ", ".join(list_scenarios())))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
