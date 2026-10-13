# Next performance tests

Status: d1 scheduler work remains untested. [E15](experiments/E15-dspark-server-recheck.md) screened speculation; [E16](experiments/E16-dspark-cost-and-prefix.md) measured its cycle cost, reproduced target-only token flips, and tested draft formats/CPU sampling. No VQ inference-engine patches have been made. [J00](experiments/J00-orin-nano-comparison.md) adds the Jetson comparison and a local CUDA virtual-memory workaround.

## Before using the board

- Run `python3 -m unittest discover -s tests -v`. These tests use saved data and mock processes, not network services.
- Inspect the pinned llama.cpp source (`8bb3cf0`) before changing the d1 scheduler. Start with `tools/server/server-decision.cpp` and follow the state-copy and batching code.
- Keep the saved seed-1 references unchanged. Use `d1_calibration.py` to prepare separate calibration inputs. Rebuilding the imatrix and weights still requires a model and inference runtime.
- Confirm ownership of running board workloads before deployment. Do not run two benchmark agents at once.

## 1. Establish a repeatable baseline

Keep the model, projector, build, prompts, and CPU placement fixed. Save their paths and hashes with each run. Start with the published d1-3B Q4_K_M to isolate scheduling changes from quantization changes.

Use at least three separate server launches per configuration, with reference runs between changed configurations. Use distinct states and at least 100 measured requests per launch. Report p50, p95, and decisions/s; do not report only the best launch.

`scripts/d1_run.sh` now keeps separate command, server, and client logs under `~/lfm-fast/results/`. Example, when the board is available:

```bash
bash ~/lfm-fast/d1_run.sh D03-reference 2 '--device HTP0 -ngl 99 -fa on -t 1' \
  'python3 d1bench.py lat -n 100 --warm 5'
```

Use separate profiled runs to count passes and identify expensive operations. `prof_agg.py` reports NPU batches, which are not necessarily whole model passes or requests; match them to server logs. Profiling adds overhead, so do not use those runs for the latency comparison.

## 2. d1-3B: reduce passes per request

Hypothesis: the shared state can be evaluated once, then all question suffixes can be evaluated together. The target is two passes rather than the recorded three or four. About 300 ms is a target, not a prediction backed by a new measurement.

Check the scheduler's actual dependencies before implementing this. Each question needs the same initial attention and convolution state, not the state after another question's suffix.

Accept a change only if:
- Profile and server logs show fewer passes.
- Unprofiled latency improves across repeated launches without a p95 regression.
- Probabilities match the unmodified build within a tolerance measured from repeated baseline runs.
- Question order changes and consecutive unrelated requests do not leak state between questions or requests.

## 3. Omni: separate latency from throughput

For latency, vary state length and question count while keeping the meaning fixed. Record token counts and operation times. The recorded 100–107 ms is total NPU operation time; it is not proof of a fixed 100 ms dequantization cost.

For throughput, test actual cross-request batches of 1, 2, and 4 independent decisions. More HTTP clients alone did not help. Keep attention masks and request state separate; check each answer against its single-request result. Include queue wait in p95 latency.

## 4. Chat: small-batch verification

E16 reproduced the chat/math flips with target-only batched-prefix evaluation and refreshed the QAD cost curve for 1, 2, 3, 4, 5, and 10 positions. Localize the changed target logits/state, including the unresolved code case; forcing HVX did not restore parity. Test `--no-spec-draft-backend-sampling` on complete tasks before adopting its short-prefix gain. A faster kernel must improve complete requests without a quality regression. Keep the deployed QAD model and serving sampler separate from the historical Q4_0 greedy benchmark.

Use useful output length, not repetition loops, to score speculation. Defer CPU/GPU redistribution and general clock sweeps until evidence points to them.

## Quality gate

Do not route to Omni solely on probability confidence. In the saved F16 references, four of the 31 binary answers at ≥99% confidence are wrong under the existing weak labels. Real-task labels and a joint model-residency test are required before adopting a router.
