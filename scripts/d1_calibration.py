#!/usr/bin/env python3
"""Check historical d1 calibration prompts against seed 1; optionally write a filtered copy."""
import argparse
from pathlib import Path

from d1_fidelity import dataset

START = '<|startoftext|><|im_start|>user\n'
QUESTION = '\n\n\nQUESTION:\n'
END = '<|im_end|>\n<|im_start|>assistant'


def filter_prompts(text):
    parts = text.split(START)
    if parts[0] or len(parts) == 1:
        raise ValueError('expected systemone prompts starting with the user marker')
    evaluation = {state for state, _ in dataset(1)}
    kept = []
    removed = 0
    for index, part in enumerate(parts[1:], 1):
        state, separator, _ = part.partition(QUESTION)
        if not state or not separator or not part.rstrip('\n').endswith(END):
            raise ValueError(f'prompt {index}: invalid systemone format')
        if state in evaluation:
            removed += 1
        else:
            kept.append(START + part)
    if not kept:
        raise ValueError('no calibration prompts remain')
    return ''.join(kept), removed, len(kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('--output', type=Path, help='write disjoint prompts to a new file; never overwrite')
    args = ap.parse_args()
    try:
        clean, removed, kept = filter_prompts(args.input.read_text())
        if args.output:
            # Keep the historical calibration input and its imatrix paired.
            with args.output.open('x') as out:
                out.write(clean)
    except (OSError, ValueError) as error:
        ap.exit(1, f'{error}\n')
    print(f'{removed} overlapping prompts; {kept} disjoint prompts')
    if args.output:
        print(f'Wrote {args.output}. Rebuild the imatrix and weights before evaluating them.')
    elif removed:
        ap.exit(1, 'Calibration overlaps the seed-1 evaluation set. Use --output to write a filtered copy.\n')


if __name__ == '__main__':
    main()
