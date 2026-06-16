# HANDOFF 2026-06-16 — MTP soup win + 122B MTP running + 122B DFlash queued

Read me first: `git show test_result:HANDOFF.md`. Repo `Bensong0506/speculators`.
Intranet is **pull-only**; results come back via `output_log` / `output_debug` on a
branch. Deliver runnable scripts (commit + pull), don't make the user hand-copy.

## Branch map (each line = a separate concern, don't mix)
- **`mtp-training`** (tip `59beae7`) — MTP training + shared infra. Has: distill
  `--concurrency`, MTP soup tooling, RUN.md 122B auto-detect, the **MoE
  `intermediate_size` fix (`86cca58`)**. **122B MTP training is RUNNING here.**
  (The 122B DFlash launcher was here then reverted → it now lives only on the
  branch below.)
- **`dflash-122b-distill`** (tip `ccf78aa`) — **122B DFlash training home.** Has the
  DFlash 122B launcher + RUN.md DFlash section + the `intermediate_size` fix.
  NOT yet run.
- **`test_result`** (tip `4af8e8a`) — reports, `output_log` (results回传), customer
  HTML deck. Reports: `dflash_100k_three_way_report.md`,
  `speculative_decoding_overview.md`, customer `spec_decoding_customer_report.html`.

## Done this session
1. **DFlash 100k three-way eval analyzed + report** (`test_result` `03774ed`):
   in-domain ALLaVA trained-DFlash 73.8 tok/s, mean-accept **2.666 (+37% vs orig)**,
   first-pos 0.810; OOD MMStar 76.1 tok/s, mean-accept **2.416 (+15%)**. DFlash
   fastest on both; **domains both up**. + `speculative_decoding_overview.md` (`21bfede`).
2. **Distill was sequential (concurrency=1)** → added `--concurrency` (order-preserving
   thread pool) to `scripts/distill_allava_with_qwen.py` + wired into distill shells
   (`mtp-training` `963eadf`). Big distill speedup.
3. **MTP OOD fix = WiSE-FT / weight soup** (`mtp-training` `ccf8052`): `stitch_mtp.py
   --alpha` (blend finetuned+native head), `sweep_mtp_soup_alpha.sh`, `SKIP_ORIGINAL`.
   **User ran α=0.5 — comprehensive WIN** (see Key results). ckpt:
   `output/mtp_qwen3.5_9b_mm_distilled/mtp_bf16_lr3e5_100k_0612_0823/checkpoints/checkpoint_best`.
4. **Customer HTML deck** (Huawei red/white). **Canonical = LOCAL HD file** (8 slides,
   1280×720 fullscreen-scaling, includes the soup conclusion + an "MTP vs DFlash"
   intro slide): `~/Documents/Codex/2026-05-27/spec_decoding_customer_report.html`
   (on the user's Mac). The GitHub copy on `test_result` (`4af8e8a`) is the OLDER
   7-slide version — STALE (no soup conclusion). User said they'd upload HD manually;
   `.pptx` export still offered, not done.
5. **122B MTP**: RUN.md auto-detect block (`39b3de9`). First run crashed →
   **fixed `scripts/train.py` MoE bug** (`86cca58`): `create_transformer_layer_config`
   read dense `intermediate_size`; 122B MoE only has `moe_intermediate_size` → added
   fallback (runs for BOTH mtp & dflash). User: **MTP 122B now running.**
6. **122B DFlash launcher** `nohup_dflash_122b_allava_distilled.sh` → moved to
   `dflash-122b-distill`. Warm-starts from a downloaded z-lab 122B DFlash; base reads
   block_size/aux-layers/arch from `FINETUNE_FROM`; CE+fp32+lr3e5; auto-detect data.
7. **DFlash v2 blog** (lmsys 2026-06-15): **SGLang-only, NOT vLLM**; not training-free;
   pretrained v2 drafts only for 397B + Qwen3-4B (NOT 9B/122B). Verdict: can't adopt
   short-term → **stay on DFlash v1** (the vLLM path that already works).

## Key results (the numbers that matter)
- **MTP soup α=0.5 = both domains above native, zero retrain** (the headline win):
  | metric | native | finetuned (α=1) | **soup α=0.5** |
  |---|---|---|---|
  | in-domain accept | 2.916 | 3.142 | **3.147 (+7.9%)** |
  | OOD accept | 2.822 | 2.694 (−4.6% down) | **2.861 (+1.4% up)** |
  Soup recovered the OOD regression AND kept in-domain. MTP is now "comprehensive".
- **DFlash 100k**: in-domain +37% accept / +20% tok/s (2.45× baseline), OOD +15%.
- **Why DFlash > MTP throughput**: DFlash drafts in PARALLEL (1 fwd, `dflash.py`
  parallel_drafting); MTP is AUTOREGRESSIVE (k fwds). At equal accept, parallel wins
  on `T_draft`. (Confirmed in vllm-fork code.)

## Next session — first actions
1. **MTP 122B training is RUNNING** (`mtp-training`) — check its result when done
   (in-domain + OOD accept/throughput). Stitch + eval like the 9B
   (`test_mtp_{allava,mmstar}_orig_vs_trained.sh`).
2. **122B DFlash NOT run yet** (`dflash-122b-distill`): smoke first —
   ```
   git checkout dflash-122b-distill && git pull
   MODEL=/data/wenxuan/Qwen3.5-122B-A10B \
   FINETUNE_FROM=<downloaded z-lab 122B DFlash dir> \
   ALLAVA_IMAGE_ROOT=<real image path on training node> \
   MAX_SAMPLES=50 EPOCHS=1 VALIDATE_INITIAL=0 \
   bash examples/train/nohup_dflash_122b_allava_distilled.sh
   ```
   **Experimental/unvalidated.** Likely failure points: draft arch/weight mismatch on
   warm-start load, or fp32 trainer OOM (`HIDDEN_STATES_DTYPE=bfloat16`). Send log.
3. **Customer deck**: HD version is local-only + canonical; GitHub copy is stale. If
   user wants it on GitHub or as `.pptx`, sync/generate from the local HD file.

## Gotchas / workflow (keep)
- `pkill -f vllm` before runs. Path prefix: training node uses **`/home/wenxuan`**;
  the 122B scripts default image/model paths to `/data/wenxuan` → **set
  `ALLAVA_IMAGE_ROOT` to the real path** (this bit the MTP 122B run).
- **MTP must be bf16** (`HIDDEN_STATES_DTYPE`; fp32 breaks the MTP lm_head). **DFlash
  uses fp32** (separate draft — the winning recipe).
- 122B = Qwen3.5-122B-A10B MoE, heads=32 → **VLLM_TP in {4,8}, not 6**.
- DFlash = parallel draft; MTP = autoregressive. MTP=accept ceiling, DFlash=throughput
  ceiling (soup narrowed it; v2 would close it but is SGLang-only).
- eval `ALLAVA_JSONL` must match the training jsonl (val tail = last 10%).
