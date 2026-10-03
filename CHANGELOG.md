# CHANGELOG

本文件记录本项目实际发生的变更。只写真实做过的事，不补历史。

## [0.1.0] — 2026-10-03

首个可用版本：卡数与显存测算（显式含 KV Cache）→ 并行策略 → 每瓦 Token 吞吐 →
三年 TCO → 四层形态一致性校验 → 五类文档装配。

### 新增 · 数据层
- `data/models/model_specs.csv` —— 8 个模型的参数量、激活参数量、层数、kv_heads、
  head_dim、是否 MoE（MoE 模型的激活参数与总量分开记录）
- `data/hardware/accelerator_specs.csv` —— 7 个型号的显存、带宽、TDP、互联；
  含 3 个竞品对照（A100-80G / L40S / MI300X，`official` 级）与 2 个 `unverified`
- `data/frameworks/engine_matrix.csv` —— vLLM / TensorRT-LLM / SGLang 的
  分页 KV Cache、前缀复用、量化支持、批处理能力
- `data/scenarios/` —— 3 个推理场景：金融 RAG / 视联网多模态 / 企业 Agent 服务
- `data/sources.md` —— H1–H12 来源登记 + 三级可信度
- `data/assumptions.md` —— A（显存/卡数）· B（每瓦吞吐）· C（TCO）三组假设 +
  D 节「明确不做的事」

### 新增 · 计算层
- `src/catalog.py` —— 目录加载，空字段归一为 None，缺列/未知可信度直接抛错
- `src/sizing.py` —— **KV Cache 显式计算** + 卡数向上取整到整机粒度
- `src/tp_ep.py` —— 按是否 MoE 推荐 TP/EP，说明依据与代价；通信量量级对比
- `src/token_per_watt.py` —— 每瓦 Token 吞吐（★ 核心指标）+ 每百万 Token 电费
- `src/tco.py` —— **电费单列** + PUE 附加；硬件价未查到则列为未计入项
- `src/form_validator.py` —— 芯片→板卡→服务器→一体机 五条跨层一致性规则
- `src/doc_assembler.py` —— 方案书 / 标书技术章节 / 白皮书 / 形态规格书 /
  客户理解成本清单
- `src/cli.py` —— `--all` / `--scenario` / `--accel` / `--util` / `--compare`

### 新增 · 测试
- 55 项单元与端到端测试，含边界：负数 token、缺字段、未知型号、
  未公布显存（判 unknown 不猜）、**优化系数不叠加**、
  **硬件价缺失不按 0 凑总额**、四层校验至少发现 3 类冲突、
  白皮书不得出现竞品贬低用词、生成结果可复现

### 三个关键设计决策
1. **KV Cache 必须显式算** —— 推理场景显存瓶颈常在 KV cache 而非权重，
   只算权重会系统性低估，导致卡数偏乐观、交付时 OOM。
2. **硬件采购价未查到就列为未计入项** —— 加速卡单价属商业信息，
   不用估算值凑总额。编价格算出的 TCO 在客户面前一问就穿帮。
3. **优化系数取最大值而非连乘** —— 多项优化作用于不同瓶颈，
   相乘（1.6×1.8×1.5×1.4 = 6.0 倍）在任何单卡上都不可能。

### 自查发现并修掉的问题
1. **EP 通信量公式缺「仅 MoE 层通信」** —— 初版对全部层算 all-to-all，
   导致 EP/TP 恒为 1.14（EP 反而更贵），与工程事实矛盾。
   修正为只对 MoE 层计通信，EP/TP = 0.28。
2. **非 MoE 模型的 EP 通信量未置零** —— 改公式时丢了 `is_moe` 守卫，
   稠密模型也算出 EP 通信量。
3. **C-01 校验字段名与形态模板不一致** —— 校验器读 `board.slots_used`，
   而字段规范里是 `quantity`，导致槽位冲突检测从未生效。
   已在三处（校验规则 / 字段规范 / 示例数据）统一字段名并补上板卡数 × 槽位校验。
4. **白皮书只由 `--compare` 生成到根目录** —— 但 `ARTIFACTS` 按场景列出，
   改为按场景生成。
5. **数值字段误调 `.strip()`** —— 规格回填后字段变成 float 导致崩溃；
   改为 `_blank()` 判定，**0 是具体取值不算缺失**。

### 已知问题
- 吞吐基线与全部优化系数为占位值，非实测
- 无真卡实测，输出为理论推算
- 加速卡单价未公开，TCO 合计不完整
- 电价与 PUE 为占位值
- MoE 层占比假设（0.25）直接决定 EP/TP 比值，是本模块最敏感参数
- hidden 尺寸为占位近似（kv_heads × head_dim × 4）
- 曦望规格多为 unverified，REX-S 单卡显存标「未查到」故判 unknown
