#!/usr/bin/env python3
"""Save matched d1 requests, full responses, and latency samples on either board."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import random
import statistics
import time
import wave

from d1bench import Q, png_red_square, post, rand_state


def requests(mode, count, audio=None):
    if mode == 'text':
        rng = random.Random(1)
        return [{'state': json.dumps(rand_state(rng)), 'questions': Q} for _ in range(count)]
    if mode == 'image':
        return [{'state': None,
                 'images': ['data:image/png;base64,' + base64.b64encode(png_red_square()).decode()],
                 'questions': {'red': {'type': 'noul',
                                      'instructions': 'Is there a red object in this picture?'}}}]
    if audio is None:
        raise ValueError('audio mode requires --audio with a WAV file')
    with wave.open(str(audio), 'rb') as source:
        if not 0 < source.getnframes() / source.getframerate() <= 30:
            raise ValueError('audio must be between zero and 30 seconds')
    return [{'state': 'Recording from a public event.',
             'files': ['data:audio/wav;base64,' + base64.b64encode(audio.read_bytes()).decode()],
             'questions': {
                 'kind': {'type': 'choice', 'instructions': 'What kind of utterance is this?',
                          'criteria': {'request': 'A request to do something',
                                       'question': 'A question asking for information',
                                       'speech': 'A speech to an audience'}},
                 'calm': {'type': 'score', 'instructions': 'How calm is the speaker?',
                          'criteria': ['Agitated', 'Neutral', 'Calm']}}}]


def validate(body, response):
    answers = response['answers']
    if set(answers) != set(body['questions']) or response['usage']['input_tokens'] <= 0:
        raise ValueError('incomplete response or invalid input token count')
    for key, question in body['questions'].items():
        answer = answers[key]
        kind = question['type']
        if answer['type'] != kind:
            raise ValueError('answer type differs from request')
        if kind == 'noul':
            values = [answer['noul']]
        else:
            probabilities = answer['probabilities']
            values = list(probabilities.values())
            if len(values) != len(question['criteria']) or not math.isclose(sum(values), 1, abs_tol=1e-4):
                raise ValueError('invalid probability distribution')
            if kind == 'choice':
                if set(probabilities) != set(question['criteria']) or answer['choice'] not in probabilities:
                    raise ValueError('invalid choice labels')
            elif not 0 <= answer['score'] <= len(question['criteria']) - 1:
                raise ValueError('score outside requested range')
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError('invalid probability')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['text', 'image', 'audio'])
    ap.add_argument('--audio', type=Path)
    ap.add_argument('--url', default='http://127.0.0.1:8098')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--label', required=True)
    ap.add_argument('-n', type=int, default=100)
    ap.add_argument('--warm', type=int, default=5)
    ap.add_argument('--conc', type=int, default=1)
    args = ap.parse_args()
    if args.n < 1 or args.warm < 0 or args.conc < 1:
        ap.error('need n >= 1, warm >= 0, conc >= 1')
    bodies = requests(args.mode, args.n + args.warm, args.audio)
    data = {'label': args.label, 'mode': args.mode, 'concurrency': args.conc,
            'requests': bodies, 'runs': [], 'complete': False}

    def one(index):
        request_index = index if args.mode == 'text' else 0
        body = bodies[request_index]
        response, elapsed = post(args.url, body)
        validate(body, response)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('invalid request latency')
        return {'request_index': request_index, 'warmup': index < args.warm,
                'wall_ms': elapsed, 'response': response}

    with args.output.open('x') as output:
        try:
            for index in range(args.warm):
                data['runs'].append(one(index))
            start = time.perf_counter()
            with ThreadPoolExecutor(args.conc) as executor:
                for result in executor.map(one, range(args.warm, args.warm + args.n)):
                    data['runs'].append(result)
            wall = time.perf_counter() - start
            samples = sorted(run['wall_ms'] for run in data['runs'] if not run['warmup'])
            data['summary'] = {'p50_ms': statistics.median(samples),
                               'p95_ms': samples[math.ceil(.95 * len(samples)) - 1],
                               'requests_per_second': args.n / wall, 'wall_seconds': wall}
            data['complete'] = True
        finally:
            # Keep completed samples if a later request fails; never replace an earlier result.
            json.dump(data, output, indent=2)
            output.write('\n')
    print(args.label, args.mode, json.dumps(data['summary']), flush=True)


if __name__ == '__main__':
    main()
