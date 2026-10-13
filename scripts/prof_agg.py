#!/usr/bin/env python3
"""Summarize complete, single-device GGML_HEXAGON_PROFILE=1 logs by token count."""
import argparse
import collections
import re
import statistics

ANSI = re.compile(r'\x1b\[[0-9;]*m')


def parse(lines):
    batches = []
    devices = set()
    for number, line in enumerate(lines, 1):
        line = ANSI.sub('', line)
        if 'profile-op ' not in line:
            continue
        device = re.search(r'\bHTP\d+\b', line)
        if device:
            devices.add(device[0])
        fields = line.split('profile-op ', 1)[1].strip().split('|')
        usec = re.search(r'\busec (\d+)\b', fields[-1])
        if len(fields) < 6 or not usec:
            raise ValueError(f'line {number}: invalid profile entry')
        if fields[0] == 'OPBATCH':
            count = re.search(r'\bn-ops (\d+)\b', fields[2])
            if not count:
                raise ValueError(f'line {number}: missing operation count')
            # The batch header precedes its operations in Hexagon logs.
            batches.append({'usec': int(usec[1]), 'count': int(count[1]), 'ops': []})
        else:
            if not batches:
                raise ValueError(f'line {number}: operation before batch header')
            batches[-1]['ops'].append((fields[0], int(usec[1]), fields[2]))
    if len(devices) > 1:
        raise ValueError('use a single-device profile; mixed HTP logs are not supported')
    if not batches:
        raise ValueError('no profile batches found')
    for index, batch in enumerate(batches, 1):
        if not batch['ops'] or len(batch['ops']) != batch['count']:
            raise ValueError(f'batch {index}: expected {batch["count"]} operations, got {len(batch["ops"])}')
    return batches


def token_count(batch):
    # shortcut: use the first model norm for LFM; extend this for other layouts.
    for op, _, shape in batch['ops']:
        if op.startswith('RMS_NORM'):
            match = re.match(r'\d+:(\d+)(?: x | -> )', shape)
            return int(match[1]) if match else None
    return None


def summarize(batches):
    groups = collections.defaultdict(list)
    for batch in batches:
        groups[token_count(batch)].append(batch)
    for tokens in sorted(groups, key=lambda n: -1 if n is None else n):
        group = groups[tokens]
        totals = collections.Counter()
        calls = collections.Counter()
        for batch in group:
            for op, usec, _ in batch['ops']:
                totals[op] += usec
                calls[op] += 1
        print(f'\ntokens/pass={tokens if tokens is not None else "unknown"}: {len(group)} batches, '
              f'mean batch={statistics.mean(b["usec"] for b in group) / 1000:.3f} ms')
        print('| operation | calls/batch | ms/batch | us/call |')
        print('|---|---:|---:|---:|')
        for op, usec in totals.most_common():
            print(f'| {op} | {calls[op] / len(group):.2f} | {usec / len(group) / 1000:.3f} | {usec / calls[op]:.1f} |')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('log')
    args = ap.parse_args()
    try:
        with open(args.log, errors='replace') as source:
            batches = parse(source)
    except (OSError, ValueError) as error:
        ap.exit(1, f'{error}\n')
    summarize(batches)


if __name__ == '__main__':
    main()
