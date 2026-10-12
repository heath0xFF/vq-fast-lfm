# E14 – N-gram speculation, edit-style prompts, and the "second NPU"

master build, pinned cpu2, `-t 1`, greedy, target = geniex Q4_0, `llama-speculative-simple` vs `llama-completion` (plain).

## N-gram speculation (`--spec-type ngram-mod`, no draft model)
| prompt | tokens | plain t/s | ngram-mod default (match 24) | match 10 | match 6 | **match 4** |
|---|---:|---:|---:|---:|---:|---:|
| edittext (fix typos in a paragraph) | first 110 | 22.4 | 22.1–22.2 (0 drafts) | 23.1 | 25.8 | **26.3** (46/46 accepted) |
| editcode (add type hints/docstrings) | first 110 | ~22.0 | 21.8 (0 drafts) | 21.7 | 21.3 | **23.1** (25/25) |
| json / code / chat-like (250 tok) | | 20.8–22 | = plain, 0–12 drafts | | | |
| edittext, 250 tokens | 250 | 20.7 | 39.8–42.4 (159/159) | | | |

- ⚠ The 250-token edittext 2× is mostly an artifact: after the real edit (first ~100 tokens, the model does fix the paragraph correctly) it falls into a repetition loop ("Final output should be a single line…") which n-gram predicts perfectly. Use the 110-token rows as the honest numbers.
- ⚠ The `summarize` prompt in [E10](E10-dspark-grid.md) is similar: the model echoes the input verbatim, so its large speedups (DSpark 42 t/s, n-gram 78 t/s) are best-case echo numbers.
- llama-server, identical request twice: 1st 20.1 t/s (plain 20.6, 14/64 drafts accepted), 2nd **228 t/s** (239/245) because the n-gram pool contains the previous identical output → useful for regenerate/retry, not general.
- Property that matters: with no n-gram match the verify batch is M=1, i.e. plain decoding → **no measurable downside**.
- `scripts/serve_best.sh`: `SPEC=ngram` (n-match 4, n-min 2, n-max 8).

## Second Hexagon NPU?
`GGML_HEXAGON_DEVICES=HTP0,HTP1 llama-bench --list-devices` lists HTP1 (`cdsp1:4`) only because FASTRPC_GET_DOMAINS fails and the backend falls back to a static domain list; without the env var `--device HTP1` is "invalid device". `/dev` has `fastrpc-cdsp`, `fastrpc-cdsp-secure`, `fastrpc-adsp-secure`, `fastrpc-gdsp0(-secure)` – no cdsp1 node. AI Hub also lists `npu_count: 1` for the Ventuno Q. Conclusion: one HTP (I did not get a working session on HTP1). Even with two, decode is DDR-bound.
