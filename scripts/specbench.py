#!/usr/bin/env python3
"""Compare plain/DSpark servers with fixed greedy prompts; save full responses and timings."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request


def complete(url, body):
    request = urllib.request.Request(url.rstrip('/') + '/completion', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    elapsed = (time.perf_counter() - start) * 1000
    timings = result['timings']
    if timings['cache_n'] != 0:
        raise ValueError('prompt cache was used despite cache_prompt=false')
    if not result.get('tokens') or not all(type(token) is int for token in result['tokens']):
        raise ValueError('server did not return output token IDs')
    for key in ('predicted_per_second', 'predicted_ms'):
        if not math.isfinite(timings[key]) or timings[key] <= 0:
            raise ValueError(f'invalid timing: {key}')
    return {'wall_ms': elapsed, 'response': result}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('prompts', nargs='+', type=Path)
    ap.add_argument('--url', default='http://127.0.0.1:8098')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--label', required=True)
    ap.add_argument('-n', type=int, default=128)
    ap.add_argument('-r', type=int, default=3)
    ap.add_argument('--warm', type=int, default=1)
    args = ap.parse_args()
    if args.n < 2 or args.r < 1 or args.warm < 0:
        ap.error('need n >= 2, r >= 1, warm >= 0')
    prompts = {str(path): path.read_text() for path in args.prompts}
    if len(prompts) != len(args.prompts) or any(not text.strip() for text in prompts.values()):
        ap.error('prompt paths must be unique and their contents nonempty')
    options = {'n_predict': args.n, 'temperature': 0, 'top_k': 1, 'seed': 1,
               'cache_prompt': False, 'return_tokens': True}
    data = {'label': args.label, 'request_options': options, 'prompts': prompts,
            'complete': False, 'runs': []}
    # Exclusive creation protects earlier results, including failed runs.
    with args.output.open('x') as output:
        try:
            # shortcut: repeated prompts suit DSpark, use distinct prompts for n-gram pools.
            for repeat in range(args.warm + args.r):
                for path, prompt in prompts.items():
                    run = complete(args.url, {**options, 'prompt': prompt})
                    run.update(prompt=path, warmup=repeat < args.warm)
                    data['runs'].append(run)
                    timing = run['response']['timings']
                    print(f'{args.label} {Path(path).stem} {"warm" if run["warmup"] else "measured"}: '
                          f'{timing["predicted_per_second"]:.2f} tok/s, {run["wall_ms"]:.0f} ms, '
                          f'draft={timing.get("draft_n", 0)}, accepted={timing.get("draft_n_accepted", 0)}', flush=True)
            data['complete'] = True
        finally:
            json.dump(data, output, indent=2)
            output.write('\n')
    for path in prompts:
        runs = [run for run in data['runs'] if run['prompt'] == path and not run['warmup']]
        tps = statistics.median(run['response']['timings']['predicted_per_second'] for run in runs)
        wall = statistics.median(run['wall_ms'] for run in runs)
        print(f'{args.label} {Path(path).stem}: median {tps:.2f} tok/s, {wall:.0f} ms/request')


if __name__ == '__main__':
    main()
