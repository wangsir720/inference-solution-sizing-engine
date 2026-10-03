# 四层产品形态技术规格书

> 由 `src/doc_assembler.py` 自动装配。**规格为示例数据，非真实产品配置。**

## 1. 各层必填字段与约束
### 芯片（chip）

- 必填：`model`、`arch`、`tdp_w`、`vram_gib`、`memory_type`、`memory_bw_gbs`、`slot_support`、`intra_interconnect`
- 约束：芯片层决定上限：槽位支持数、显存容量、互联能力都是上层不可逾越的天花板

### 板卡（board）

- 必填：`model`、`quantity`、`slots_used`、`tdp_w`、`vram_gib`、`interconnect`、`form_factor`、`power_connector`
- 约束：板卡层不得超过芯片层：槽位数 ≤ 芯片槽位支持，功耗须在板卡供电上限内

### 服务器（server）

- 必填：`model`、`slots_available`、`fan_modules`、`fan_capacity_w`、`total_tdp_w`、`chassis_overhead_w`、`interconnect_topology`
- 约束：服务器层是散热与供电的落点：散热能力必须覆盖整机功耗

### 一体机（appliance）

- 必填：`model`、`power_budget_w`、`cooling_type`、`form_factor`、`rack_units`、`networking`、`target_scenarios`
- 约束：一体机层是客户采购决策的界面：功耗预算、散热形式、机架高度都要对齐客户机房

## 2. 当前规格

| 层 | 字段 | 值 |
|---|---|---|
| 芯片 | `model` | S2 |
| 芯片 | `tdp_w` | 400.0 |
| 芯片 | `vram_gib` | 64 |
| 芯片 | `slot_support` | 8 |
| 芯片 | `arch` | DSA |
| 板卡 | `model` | S2-加速卡 |
| 板卡 | `quantity` | 2 |
| 板卡 | `slots_used` | 1 |
| 板卡 | `tdp_w` | 400.0 |
| 板卡 | `vram_gib` | 64 |
| 板卡 | `interconnect` | 专用互联 |
| 服务器 | `model` | 4U 机架服务器 |
| 服务器 | `slots_available` | 4 |
| 服务器 | `fan_modules` | 2.0 |
| 服务器 | `fan_capacity_w` | 500 |
| 服务器 | `total_tdp_w` | 800.0 |
| 服务器 | `chassis_overhead_w` | 200 |
| 一体机 | `model` | 4U 一体机 |
| 一体机 | `power_budget_w` | 1000.0 |
| 一体机 | `cooling_type` | 风冷 |
| 一体机 | `form_factor` | 4U 机架 |

**字段完整率 100%**（21/21）

## 3. 一致性校验

未发现跨层不一致（已校验 5 条规则）。

> 只做参数一致性校验，不做振动/散热的工程设计（见 assumptions.md D 节）

---
生成命令：`python -m src.cli`
