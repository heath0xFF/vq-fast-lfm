# E13 – Run-to-run bimodality of decode speed (21.0 vs 22.5 t/s)

Identical command (`llama-bench -p 0 -n 128`, pinned, same model) gives **either 21.0 ± 0.02 or 22.5 ± 0.03 t/s**, constant within a process, switching in streaks of a few launches over minutes (≈ 6.7 % apart).
Not explained by: CPU choice (2/3/0/1), thread count, `GGML_HEXAGON_HOSTBUF`, `GGML_HEXAGON_POWER_MODE` (burst/high_performance/sustained/balanced), temperature (36–40 °C everywhere), display scanout (no display connected), visible background processes.
Likely a system-level DDR/interconnect (bandwidth vote / LLCC) or memory-placement state outside user control (no root). → **Always bracket A/B comparisons with a reference run and repeat launches; report both modes.** (Tooling idea: take best-of-N launches.)
