#!/usr/bin/env python3
"""Compare sequential decoding with a batched prefix on a plain, no-drafter server."""
import argparse
import json
from pathlib import Path
import urllib.request


def post(url, endpoint, body):
    request = urllib.request.Request(url.rstrip('/') + endpoint, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def probe(url, reference, name, index):
    if not reference['complete'] or index < 1:
        raise ValueError('need a complete reference and a zero-based index >= 1')
    paths = [path for path in reference['prompts'] if Path(path).stem == name]
    if len(paths) != 1:
        raise ValueError('prompt name must identify exactly one reference prompt')
    path = paths[0]
    rows = [run for run in reference['runs'] if run['prompt'] == path and not run['warmup']]
    if not rows or len(rows[0]['response']['tokens']) <= index:
        raise ValueError('reference is shorter than the requested position')
    options = {**reference['request_options'], 'cache_prompt': False, 'n_probs': 5,
               'post_sampling_probs': False, 'return_tokens': True}
    sequential = post(url, '/completion', {**options, 'prompt': reference['prompts'][path],
                                         'n_predict': index + 1})
    if any(t != 'none' for t in sequential['generation_settings']['speculative.types'].split(',')):
        raise ValueError('this control requires a server with no speculative decoding')
    tokens = sequential['tokens']
    if len(tokens) != index + 1 or len(sequential['completion_probabilities']) != len(tokens):
        raise ValueError('server did not return the requested tokens and probabilities')
    prompt_tokens = post(url, '/tokenize', {'content': reference['prompts'][path],
                                          'add_special': True, 'parse_special': True})['tokens']
    if len(prompt_tokens) != sequential['timings']['prompt_n']:
        raise ValueError('tokenized prompt length differs from the evaluated prompt')
    # Token IDs preserve the exact prefix; detokenizing and retokenizing may change it.
    prefix = prompt_tokens + tokens[:index]
    batched = post(url, '/completion', {**options, 'prompt': prefix, 'n_predict': 1})
    if len(batched['tokens']) != 1 or len(batched['completion_probabilities']) != 1:
        raise ValueError('server did not return the batched-prefix prediction')
    if sequential['timings']['cache_n'] != 0 or batched['timings']['cache_n'] != 0:
        raise ValueError('unexpected prompt cache use')
    if batched['timings']['prompt_n'] != len(prefix):
        raise ValueError('batched prefix length changed')
    return {'prompt': path, 'index': index, 'request_options': options, 'prefix_tokens': prefix,
            'reference_match': tokens == rows[0]['response']['tokens'][:index + 1],
            'same_next_token': tokens[index] == batched['tokens'][0],
            'sequential': sequential, 'batched_prefix': batched}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reference', type=Path)
    ap.add_argument('prompt')
    ap.add_argument('index', type=int, help='zero-based output position to compare')
    ap.add_argument('--url', default='http://127.0.0.1:8098')
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    reference = json.loads(args.reference.read_text())
    with args.output.open('x') as output:
        result = probe(args.url, reference, args.prompt, args.index)
        json.dump(result, output, indent=2)
        output.write('\n')
    print(f'{args.prompt} token {args.index + 1}: same_next_token={result["same_next_token"]}, '
          f'reference_match={result["reference_match"]}')
    for name, row in [('sequential', result['sequential']['completion_probabilities'][args.index]),
                      ('batched_prefix', result['batched_prefix']['completion_probabilities'][0])]:
        print(name, json.dumps(row))


if __name__ == '__main__':
    main()
