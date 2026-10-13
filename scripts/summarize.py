#!/usr/bin/env python3
"""Summarize geniex-bench JSON results (median across measured runs) as a markdown table."""
import json, glob, os, statistics as st, sys
root = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw')
rows = []
for f in sorted(glob.glob(os.path.join(root, '**', '*.json'), recursive=True)):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    runs = d.get('runs') or []
    if not runs or 'decode_tps' not in runs[0]:
        continue
    name = os.path.relpath(os.path.dirname(f), root) if os.path.basename(f) == 'result.json' else os.path.relpath(f, root)[:-5]
    cmd = ''
    c = os.path.join(os.path.dirname(f), 'cmd.txt')
    if os.path.exists(c):
        cmd = ' '.join(l.split(': ', 1)[1].strip() for l in open(c).read().splitlines() if ': ' in l)
        cmd = cmd.replace('--plugin llama_cpp', '').replace(' --output-json', '')
    p = d['params']
    pre = [r['prefill_tps'] for r in runs]; dec = [r['decode_tps'] for r in runs]
    ttft = [r['ttft_us']/1000 for r in runs]
    rows.append((name, d['device'], p['n_prompt'], p['n_gen'], st.median(pre), st.median(dec),
                 (st.pstdev(dec) if len(dec) > 1 else 0), st.median(ttft), cmd))
filt = sys.argv[1] if len(sys.argv) > 1 else ''
print('| experiment | dev | pp | tg | prefill t/s | decode t/s (±sd) | ttft ms | config |')
print('|---|---|---|---|---|---|---|---|')
for r in rows:
    if filt in r[0]:
        print(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.1f} | {r[5]:.2f} (±{r[6]:.2f}) | {r[7]:.0f} | `{r[8]}` |')
