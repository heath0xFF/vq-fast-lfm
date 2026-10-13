# E16 — DSpark cycle cost, draft formats, and target-only token flips

Observed 2026-10-13. Follow-up to [E15](E15-dspark-server-recheck.md). No engine patch or serving-default change.

## Results in brief

- **The target alone can change its next token at the same prefix when that prefix is evaluated as a batch.** This reproduced the E15 chat and math differences without a draft model or speculative rollback. It does not diagnose every output difference.
- **Q8_0 is the best tested draft format for these prompts.** E15 also tested Q4_K_M. E16 tested Q4_0 and the official F16 file. F16 fits with smaller buffers, but is slower.
- A Q8/n9 cycle costs about **109 ms for target verification + 46 ms for drafting + 2 ms for feature preparation**, before host overhead. It needs roughly four useful output tokens per cycle to beat plain decoding.
- **Move draft token selection to the CPU:** `--no-spec-draft-backend-sampling` improved short code/math decode speed by **about 7–9%** in two interleaved A/B pairs. Output token IDs and acceptance counts stayed unchanged. This is not a completed-task speed result.

Keep plain decoding as the default. These findings do not establish lossless output or a general DSpark speedup.

## Controls and evidence

Same QAD Q4_0 target, installed engine, HTP0 placement, CPU 2 affinity, one host thread, greedy requests, and prompts as E15. Cache disabled; all requests reported `cache_n=0`. Unless noted: context 4096, batch/microbatch 1024, one server slot, n-max 9. Short timing runs use one warmup and three measured requests per prompt per launch.

- Requests, responses, token probabilities, timings, model/tool hashes, and derived profile summary: [`results/raw/E16-investigate.NfHrhY/`](../../results/raw/E16-investigate.NfHrhY/).
- Commands and server/client logs: sibling `results/raw/E16-*` directories. Test servers bound only to localhost and stopped after each run. Existing GenieX was left untouched.
- Local source inspected at `8bb3cf0d216a5ae48360179f2931daf284be4d01`. Matching this checkout to the installed binary is **not proven**; the binary still reports `commit unknown`. Its hash and the core library hashes match E15.
- Downloaded the official [F16 draft](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark-GGUF/blob/7bc2896af56d82ccc7e156800197408db464d63b/LFM2.5-2.6B-DSpark-F16.gguf), SHA-256 `e198962c08903f3ba29f0ce6bf8e17f5e60bf85ec2f8673e1e2aab03508937e5`. It matches the source ETag. Q4_0 was made from this F16 file with the installed `llama-quantize`, not from Q8.
- No package installation, runtime replacement, system change, or persistent test service. New F16 and Q4_0 files remain in a separate board model directory, about 817 MiB total.

## 1. Same prefix, no draft model, different next token

`scripts/specprobe.py` first generates through a selected output position on a **plain** server, saving the top five token log probabilities. It then sends the original prompt token IDs plus all output IDs before that position as one new, uncached prompt. It asks for one next token. This holds the token sequence fixed while changing how the prefix state is computed.

The ordinary sequential runs matched the saved E15 plain output exactly:

| Output position, one-based | Sequential next token | Batched-prefix next token | Matches E15's speculative choice? |
|---|---|---|---|
| Chat 48 | ` clear` | ` paragraphs` | yes |
| Code 191 | ` should` | ` should` | no; this control did not reproduce that flip |
| Math 229 | ` city` | ` A` | yes |

For chat, the sequential log-probability margin of ` clear` over ` paragraphs` was only **0.0407**. Batched evaluation changed the margin to **−0.0240**. For math, the margin of ` city` over ` A` changed from **+0.0910** to **−0.1238**. These are small changes near a greedy tie, not evidence that the verifier simply trusts draft tokens.

**What this proves:** batch-dependent target computation is sufficient to reproduce two of the observed flips. No draft model or speculative state rollback was involved in these controls.

**What it does not prove:** full-prefix prefill is not the same operation sequence as a ten-token speculative verification. The exact kernel/state operation responsible remains unknown, and the longer code mismatch is unresolved.

An additional `GGML_HEXAGON_MM_SELECT=1` control disabled HMX matrix routing. Chat still changed its next token between sequential and batched-prefix evaluation, now in the opposite direction. Thus, forcing HVX alone is **not** a parity fix. Its math continuation already differed from E15 before the selected position; do not treat its matching next-token result as a fix for the E15 math case.

The inspected source routes one-row matrix work to HVX and larger work to HMX where supported; their arithmetic is not identical. [Upstream issue #25618](https://github.com/ggml-org/llama.cpp/issues/25618) also reports speculative/plain greedy differences. That report is context, not a confirmed diagnosis of this Hexagon build.

## 2. Why nine draft tokens do not mean nine times faster

A fresh QAD-target `llama-bench` screen, three repeats per size:

| Tokens in one target evaluation | Mean time, ms |
|---:|---:|
| 1 | 44.42 |
| 2 | 57.15 |
| 3 | 82.50 |
| 4 | 99.34 |
| 5 | 96.16 |
| 10 | 101.12 |

These are random-token prompt-evaluation measurements, not live verification timings. They confirm the expensive small-batch region. Ten evaluated positions cost more than twice one, even before running the drafter.

Separate operation traces used a 64-token math request, context 2048, batch/microbatch 128, `GGML_HEXAGON_PROFILE=1`, and `-lv 5`. Profiling adds overhead; these are **not** the speed-comparison runs. Initial `-lv 4` attempts did not emit operation rows and are retained under `E16-profile-*`.

Mean NPU operation-batch time:

| Work | Q8 draft setup | F16 draft setup |
|---|---:|---:|
| Target, ten positions | 109.50 ms | 107.67 ms |
| Draft, nine positions | 46.44 ms | 82.03 ms |
| Draft target-feature preparation, ten positions | 1.65 ms | 4.06 ms |

The Q8 trace has 12 ten-position target batches (601 operations each), 13 nine-position draft batches (173 operations), plus separate feature-preparation batches (25 operations). F16 has 13 target batches, 14 draft batches (178 operations), and feature batches of 32 operations. Startup, prompt processing, and smaller tail batches are excluded from the table. Do not average target and feature batches together and call that a model pass.

The draft has two notable costs:

- Its shared 128k-vocabulary output projection takes about **11 ms** per nine-position batch.
- Nine backend `TOP_K` operations take **12.12 ms** together—about **26% of Q8 draft time**. Each selects ten candidates from 128k logits; these are draft candidates, separate from the target request's greedy `top_k=1` setting.

A good request amortizes this work over several accepted tokens. A poor one does not. In E15, code acceptance fell from **102/209 (48.8%)** over 128 outputs to **369/1264 (29.2%)** over 512 outputs. Short chat accepted only **88/340 (25.9%)**. Nine proposed tokens are not nine useful tokens.

## 3. Why Q8, rather than a smaller or full-precision draft?

Default buffers, 128 output tokens, median decode tokens/s:

| Draft | Code | Math | Chat | Experiment |
|---|---:|---:|---:|---|
| Q8_0 | 31.38 | 30.44 | 19.56 | E15 |
| Q4_K_M | 29.66 | 30.07 | 17.54 | E15 |
| Q4_0 | 26.26 | 30.73 | 17.51 | E16 |

These format screens each have one launch, not interleaved format trials. Q4_0's acceptance fell to **97/254** for code and **83/385** for chat. Its math acceptance matched Q8. A smaller weight file alone does not ensure a faster completed speculative cycle.

The official F16 draft **loaded and generated successfully** with `-c 2048 -b 128 -ub 128`. This narrows the historical mapping failure: F16 is not universally unable to fit. A Q8 control used the same smaller buffers:

| Draft, smaller buffers | Code | Math | Chat |
|---|---:|---:|---:|
| Q8_0 | 30.61 | 29.80 | 19.13 |
| F16 | 23.46 | 23.64 | 15.79 |

The profile explains much of this gap. F16 draft work costs **82 vs 46 ms**. Its repeated Markov output projections take about **2.67 ms each**, versus roughly **0.8 ms** for Q8. Its FFN gate/up operations are also slower. Full precision did not improve acceptance on these three prompts: code **100/227**, math **101/230**, chat **88/340**.

All 128-token outputs from these new formats matched E15 Q8/n9 exactly. They still inherit its chat difference from plain decoding. This is an acceptance/speed screen, not a general quality ranking of draft formats.

## 4. Small measured win: CPU draft token selection

Use the existing engine flag:

```text
--no-spec-draft-backend-sampling
```

The target and draft networks stay on HTP0. Only draft sampling moves to the host; this is **not** the historical `-devd none` attempt to put the whole draft network on CPU.

Two interleaved pairs, each configuration launched separately with default buffers:

| Launch order | Code tok/s | Math tok/s |
|---|---:|---:|
| Backend selection A | 30.74 | 29.89 |
| CPU selection A | 33.44 | 32.60 |
| Backend selection B | 31.40 | 30.50 |
| CPU selection B | 33.51 | 32.63 |

Paired gains were **6.7–8.8%** for code and **7.0–9.1%** for math. All token IDs and acceptance counts matched between modes and matched E15 Q8/n9. A separate three-prompt CPU-selection screen gave chat **20.43 tok/s**, still below E15's fast plain baseline; that chat comparison was not interleaved.

This removes one measured cost, not the target's ~100 ms verification floor. No complete-answer, long-output, or broad quality validation of this flag was done. Serving defaults are unchanged.

## Reproduce and continue

Use the E15 runner command with the same QAD target and Q8 draft. Add the flag above for the CPU-selection variant. Use new output filenames for every run.

For a prefix control, start the plain server using the same runner, then:

```bash
python3 specprobe.py E15-QAD-screen.t7Y94X/plain-before.json chat 47 \
  --output prefix-chat.json
```

The index is zero-based. Use `long-plain.json code 190` or `long-plain.json math 228` for the longer cases. The helper checks no speculation, zero cache use, and exact evaluated prefix length. Use `prof_agg.parse()` on the trace logs to reproduce the operation groups in `profile-summary.json`; open logs with `errors='replace'` because debug token fragments can contain incomplete UTF-8.

Next steps:
1. Localize target batch differences with identical-prefix intermediate tensor/logit checks, including convolution state. Do not assume an HMX-only fault or a rollback-only fault.
2. Test complete answers on distinct real tasks, with interleaved plain references and multiple launches. Check both latency and answer quality.
3. If speculation remains useful, optimize the measured small-batch target path. Lower-cost draft sampling helps, but draft quantization alone did not solve the main limit.
