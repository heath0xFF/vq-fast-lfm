#!/usr/bin/env python3
"""Aggregate GGML_HEXAGON_PROFILE=1 logs: per-op usec, split decode (n_tok=1 batches) vs prefill."""
import re, sys, collections
log = sys.argv[1]
ansi = re.compile(r'\x1b\[[0-9;]*m')
batches = []   # list of (total_usec, n_ops, [(op, usec, name)])
cur = []
for line in open(log, errors='ignore'):
    line = ansi.sub('', line)
    if 'profile-op' not in line: continue
    body = line.split('profile-op ', 1)[1].strip()
    f = body.split('|')
    op = f[0]
    m = re.search(r'usec (\d+)', body)
    if not m: continue
    us = int(m.group(1))
    if op == 'OPBATCH':
        n = int(re.search(r'n-ops (\d+)', body).group(1))
        batches.append((us, n, cur)); cur = []
    else:
        cur.append((op, us, f[1] if len(f) > 1 else ''))
def kind(b):
    # decode batches: MUL_MAT src1 'x 2048:1'; prefill: '2048:64'
    for op, us, name in b[2]:
        if op.startswith('MUL_MAT') or op.startswith('RMS_NORM'):
            return None
    return None
dec, pre = [], []
for b in batches:
    txt = ' '.join(n for _, _, n in b[2][:1])
    # use shapes: look at the RMS_NORM+MUL entry for token dim -- parse again
    pass
# simpler: classify by first RMS_NORM field 4 (shape) in raw lines
