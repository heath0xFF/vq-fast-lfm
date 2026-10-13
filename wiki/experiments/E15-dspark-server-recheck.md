# E15 — DSpark on the live Ventuno Q: QAD target, server path, output checks

Observed 2026-10-13. This is a small screening test, not a general speed or quality result.

Follow-up: [E16](E16-dspark-cost-and-prefix.md) reproduces two token flips with the target alone, profiles cycle costs, tests Q4_0/F16 drafts, and measures a small gain from CPU draft sampling.

## Which draft model?

Liquid's official release is **DSpark**, with a DFlash-style parallel backbone plus Markov and confidence heads. The GGUF architecture is listed as `dflash`, but the model card uses `--spec-type draft-dspark`. No separate DeFlash checkpoint was established by this review.

Sources:
- [Liquid's release](https://www.liquid.ai/blog/lfm2.5-dspark)
- [Official LFM2.5-2.6B DSpark GGUF card](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark-GGUF)

Both Q8_0 and Q4_K_M draft files were already on the board. No downloads, package installs, runtime updates, or changes to the existing environment file were needed.

## Setup

- Target: `LFM2.5-2.6B-QAD-Q4_0.gguf`, not the plain Q4_0 target used in E10.
- Engine: existing `~/lfm-fast/lcpp-master/bin/llama-server`. Its version output is `0.6.0-dev (build 0, commit unknown)`, Clang 21.1.8, Linux arm64. The directory name does not prove the source commit; binary and model SHA-256 hashes are saved with the results.
- Target and draft: HTP0, `-ngl 99 -devd HTP0 -ngld 99`, flash attention, one host thread pinned to CPU 2.
- Context 4096; batch and microbatch 1024; one server slot. Server bound only to localhost.
- All variants use the same `/completion` endpoint, greedy sampling, seed 1, no prompt cache, and returned token IDs. Every response reported `cache_n=0`.
- Prompts: the repository's `code`, `math`, and `chat` prompts, copied unchanged. No echo or repeated-edit benchmarks.
- Each short configuration: one server launch, one warmup per prompt, then three measured requests per prompt. Prompt order repeats code/math/chat. Repeated prompts are suitable for this DSpark test, **not** a cold n-gram-pool test.
- Plain decoding was run before, between, and after the speculative tests. It showed the known launch-to-launch speed split: about 22.4 versus 20.7 tokens/s. Repeats within one process do not replace independent server launches.
- The existing GenieX service was left untouched. No active llama.cpp job or Docker container was found before testing. All test servers were stopped by their own PID afterwards.

The speculative example executable rejects `--spec-type none`; that failed smoke test is retained. The comparisons therefore use the server for **both** plain and speculative decoding.

## First 128 output tokens

Median server-reported decode tokens/s, excluding prompt evaluation:

| Configuration | Code | Math | Chat |
|---|---:|---:|---:|
| Plain, before | 22.44 | 22.43 | 22.41 |
| Q8_0 draft, n-max 9 | **31.38** | **30.44** | 19.56 |
| Q4_K_M draft, n-max 9 | 29.66 | 30.07 | 17.54 |
| Plain, middle | 20.76 | 20.74 | 20.71 |
| Q8_0 draft, n-max 3 | 21.08 | 22.87 | 20.18 |
| Q8_0 draft, n-max 1 | 23.19 | 23.79 | 22.70 |
| Plain, final | 22.33 | 22.34 | 22.37 |

Against the faster initial plain run, Q8_0/n9 gained about 40% on code and 36% on math. Request wall time fell from 5.89 s to 4.30 s and 4.43 s respectively. Chat wall time increased to 6.74 s.

The smaller Q4_K_M draft was not faster. Reducing n-max to 3 lost most of the code/math benefit. n-max 1 was only slightly faster than the fast plain baseline; launch variation prevents a strong conclusion from this screen.

### Token agreement

- All three plain launches returned identical token IDs for each prompt, including the slow launch.
- Code and math matched plain exactly for all 128 tokens, in all measured draft configurations.
- Chat did **not** match plain. Both n9 draft formats first differed at output token **48** (one-based). Q8 n1 and n3 first differed at token **83**. The difference repeated in all three measured requests.
- Apart from the speculative type, the returned generation settings matched.

Speculative verification is intended to preserve the target distribution. This test does **not** establish exact greedy-output parity on the HTP. Different numerical results in single-token and batched kernels, or recurrent-state handling, are possible causes; neither is proven here.

## Longer check: 512 output tokens

One request per prompt/configuration, no separate warmup. These are diagnostic samples, not medians over repeated trials.

| Configuration | Code tok/s | Math tok/s | Code wall time | Math wall time |
|---|---:|---:|---:|---:|
| Plain | 22.20 | 22.20 | 23.28 s | 23.25 s |
| Q8_0 draft, n-max 9 | 21.13 | 27.61 | 24.51 s | 18.77 s |

- Code became about **5% slower** with speculation. Math remained about **24% faster**.
- Code first differed from plain at output token **191**; math at token **229**.
- All runs stopped at the token limit. Even at 512 tokens, the model had not closed its reasoning block. These are decode measurements, **not completed-task latency or correctness scores**.
- Since the longer outputs differ, their speed ratio is not a controlled same-output comparison.

## Recommendation

Keep plain decoding as the default. Q8_0/n9 remains an experiment for suitable workloads, not a universal speed setting. Do not call this HTP path lossless without resolving the output differences.

Next:
1. Reproduce the first token mismatch and inspect target logits/state at the same prefix. Separate numerical batch effects from state-copy/rollback faults.
2. Repeat complete-task runs on more prompts, with several independent server launches and interleaved plain references.
3. Only then optimize small-batch verification or add task-specific speculation settings. Do not use the 128-token code gain to predict whole-response latency.

## Evidence and reproduction

- Full requests, responses, warmups, token IDs, timing fields, and SHA-256 hashes: [`results/raw/E15-QAD-screen.t7Y94X/`](../../results/raw/E15-QAD-screen.t7Y94X/).
- Server commands, logs, and client output: sibling `results/raw/E15-plain-*`, `E15-dspark-*`, and `E15-long-*` directories.
- Failed CLI smoke: `results/raw/E15-smoke.5zz4XK/`; successful server smoke: `results/raw/E15-server-smoke.hdBZht/`.
- Client: `scripts/specbench.py`; offline checks: `tests/test_specbench.py`.

The existing safe runner also accepts a chat model when `MMPROJ` is empty. From the board workspace, after copying the client, runner, and prompts:

```bash
source ~/lfm-fast/lenv-master.sh
MODEL="$M_QAD" MMPROJ='' bash d1_run.sh E15-repeat 2 \
  "--device HTP0 -ngl 99 -fa on -t 1 -c 4096 -b 1024 -ub 1024 -np 1 --spec-type draft-dspark -md $D_Q8 -devd HTP0 -ngld 99 --spec-draft-n-max 9" \
  'python3 specbench.py prompts/chat/code.txt prompts/chat/math.txt prompts/chat/chat.txt --label dspark-q8 --output E15-repeat.json -n 128 -r 3 --warm 1'
```

Use a new output filename on each run. For plain, remove the draft flags and use `--spec-type none`. The client preserves completed responses in a file marked `complete: false` if a later request fails.
