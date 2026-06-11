# 方案:PixelPrune × vLLM 0.18.0 集成与评测(Qwen3.5-9B)

> 目标:在 vLLM 0.18.0 上让 PixelPrune(视觉 token 预测编码剪枝)对我们的 Qwen3.5-9B VLM 生效,
> **先量化 prefill / TTFT 收益与精度代价**,再决定是否进生产、是否与 DFlash 叠加。
>
> 代码落地仓:vLLM fork `Bensong0506/vllm` 新分支 `pixelprune-integration`(本方案文件留在 speculators 仓)。

## 0. 关键认知(决定了工作量)

**PixelPrune 的 vLLM 集成已经存在,不需要从零实现算法。** 它通过 `vllm.general_plugins`
entry point 在每个 vLLM 进程启动时自动 monkey-patch Qwen3.5 / Qwen3-VL。所以本方案做的是:

1. 跑通 + 在**我们的模型和图**上量收益(Phase 1);
2. 若值得,再把它**沉淀进 fork** 做成可控、可与 DFlash 合栈的一等公民(Phase 2)。

所以本期重点其实是**评测**,不是造轮子。

## 1. 它现在是怎么接进 vLLM 的(现状,来自 PixelPrune 源码)

- **入口**(`setup.py`):`vllm.general_plugins → pixelprune.patches.vllm_bootstrap:maybe_apply_patches`;
  `pip install -e .` 后每个 vLLM 进程(driver / EngineCore / TP worker)自动调用,`PIXELPRUNE_ENABLED` 控开关。
- **patch 的 vLLM 位置**(`qwen3_5_vllm.py` 复用 `qwen3_vl_vllm.py`):
  - `Qwen3VLMultiModalProcessor`:`_mmp_init / _call_hf / _fields / _prompt_updates` —— 视觉 token 计数 + prompt 占位符
  - `Qwen3_VisionTransformer`:`_vt_forward` —— 在 ViT 输入处丢弃冗余 patch
  - `Qwen3_5ForConditionalGeneration`(+ `Qwen3_5MoeForConditionalGeneration`):
    `_parse_and_validate_image_input`、`_process_image_input`、`get_mrope_input_positions`
    —— 图像输入解析 + **MRoPE 位置按剪枝后 token 重算**
- **剪枝核心**(`core.py` + `methods/pred_2d.py`):
  `compute_merged_keep_indices(pixel_values, image_grid_thw, spatial_merge_size, method, metric, threshold)`
  → 每张图保留的 merged-token 索引;`merged_indices_to_patch_indices()` 展开到 patch 级喂 ViT。
- **env 旋钮**:`PIXELPRUNE_ENABLED` / `PIXELPRUNE_METHOD=pred_2d` / `PIXELPRUNE_METRIC=max` / `PIXELPRUNE_THRESHOLD=0.0` / `PIXELPRUNE_VERBOSE`
- **依赖钉死**:`vllm==0.18.0`、`transformers==4.57.6`(Qwen3.5+HF 需 `transformers>=5.2.0`)。

## 2. 前置确认(Phase 0,必须先过,否则白干)

- [ ] **架构匹配(成败关键)**:vLLM 0.18.0 含 `vllm.model_executor.models.qwen3_5`,且我们的
      Qwen3.5-9B 的 `config.json` `architectures` 在 vLLM 里解析为 `Qwen3_5ForConditionalGeneration`
      或 `Qwen3_5MoeForConditionalGeneration`(启动日志看 `Resolved architecture`)。
      ⚠️ 我们的 Qwen3.5-9B 是内部模型、backbone 是 Qwen3-Next —— 必须确认它走的是 `qwen3_5` 多模态封装,
      否则 plugin 的 patch 根本挂不上。
- [ ] **transformers 兼容**:降到 vLLM 0.18.0 后,Qwen3.5 的 processor / 权重要能在
      `transformers==4.57.6`(或 vLLM 0.18 要求的版本)下正常加载(我们之前在 0.22 上跑过)。
- [ ] **测试图像集**:PixelPrune 只在**冗余图**(文档 / 截图 / GUI / 图表)上大赚,自然图小赚。
      测试集 = 我们真实负载图(`MM_MEDIA_DIR`)+ 一组文档图(看收益上限)。
- [ ] **不与 DFlash 同跑**:本期 PixelPrune 单独验证。注意 PixelPrune 和 DFlash 都改 `get_mrope_input_positions`,
      合栈是 Phase 3 的事,先不碰。

## 3. 集成策略:两段式

### Phase 1 — 最小验证(不改 vLLM 源码,先量收益)

环境(你来搭):`pip install vllm==0.18.0` + `pip install -e PixelPrune`。

我写的交付物(放 fork 分支 `pixelprune-integration` 的 `benchmarks/pixelprune/`):

- `serve_qwen35_pixelprune.sh` —— 起 vLLM 0.18.0 server(Qwen3.5-9B),镜像我们现有 serve flags
  (`--enforce-eager`、`--max-model-len 32768`、`--limit-mm-per-prompt`、`--attention-backend` 等),
  通过 `PIXELPRUNE_ENABLED` / `PIXELPRUNE_THRESHOLD` 切开关与 τ。
- `bench_pixelprune.py` —— 对同一批图发请求,测 **TTFT / e2e 延迟 / decode 吞吐 / 视觉 token 数**
  (开 `PIXELPRUNE_VERBOSE` 读压缩比),A/B:enabled vs disabled,τ∈{0, 0.02, 0.05}。
- `accuracy_check.py` —— τ=0 看输出是否≈一致;τ>0 在小 doc 子集上测精度 delta。
  (注意:PixelPrune 自带的 VLMEvalKit eval 是 **HF 后端**,不测 vLLM,所以 vLLM 评测要我们自己写。)

**决策门**:若我们负载图上 TTFT 收益不显著(经验上 <~10%)或精度掉太多 → 停在 Phase 1,
结论是"对我们的图不划算",不进 fork。

### Phase 2 — 沉淀进 fork(仅当 Phase 1 收益值)

把剪枝做成 fork 里的一等公民,两选一(推荐 B):

- **A. Vendor**:把 `pixelprune/` 拷进 fork + 在 fork 注册 plugin entry point。改动小,但仍是 monkey-patch 形态。
- **B.(推荐)原生集成**:把剪枝逻辑直接写进 fork 的 `qwen3_5.py` / 视觉模型 / processor,用 env 或
  serving flag 控制,去掉 monkey-patch。更干净,且能和 DFlash 的 M-RoPE 改动**统一在一处维护**。

这一步 fork 分支需真正基于 vLLM **v0.18.0**(从 upstream tag 拉),并处理与 DFlash M-RoPE patch 的合并。

### Phase 3(后续,不在本期)—— 与 DFlash 合栈 / Ascend 移植

- 合栈:两处 MRoPE 改动核对,确认 PixelPrune(prefill 剪枝)不破坏 DFlash 的 verifier hidden-state 路径;
  **τ=0 无损 → 不改 target 输出 → 理论上不伤 DFlash acceptance**,先从 τ=0 验证。
- Ascend:PixelPrune 仅 CUDA 验证过;剪枝本体是像素空间预处理(可移植),但 ViT 内变长 patch 处理绑在
  vLLM CUDA 路径,需单独评估 vllm-ascend 移植成本。

## 4. 评测指标与方法

| 维度 | 指标 |
|---|---|
| 性能 | **TTFT**(主,≈视觉编码 + LLM prefill)、e2e 延迟、decode 吞吐、视觉 token 压缩比 |
| 质量 | τ=0 输出一致性;τ>0 任务精度 delta(DocVQA / ChartQA / OCRBench 子集) |
| 变量 | `PIXELPRUNE_ENABLED` × τ∈{0,0.02,0.05} × 图像类型(真实负载图 vs 文档图);固定 temp=0、同 prompt/max_tokens |

预期:文档 / 截图类大赚,自然图小赚;τ=0 近无损,τ=0.05 约 1% 精度换更高压缩。

## 5. 仓库与分支

- **方案(本文件)** → speculators 仓 `notes/pixelprune-vllm-integration-plan.md`(当前分支)。
- **代码** → vLLM fork `Bensong0506/vllm` 新分支 `pixelprune-integration`
  (Phase 1 放 `benchmarks/pixelprune/`;Phase 2 再动模型源码)。

## 6. 里程碑

1. **Phase 0**:你在 0.18.0 env 里过架构/transformers/图像集三项前置确认。
2. **Phase 1**:我写 benchmark 脚本 → 首轮 A/B 数 → 决策门。
3. **(条件)Phase 2**:原生集成进 fork。
4. **(后续)Phase 3**:DFlash 合栈 / Ascend。

---

*待办:你 review 本方案 + 过 Phase 0 三项确认后,我去 `Bensong0506/vllm` 开 `pixelprune-integration` 分支,写 Phase 1 的三个脚本。*
