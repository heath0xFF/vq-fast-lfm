"""Offline checks for the speculative-decoding client."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import specbench


def response():
    return {'tokens': [1, 2, 3], 'content': 'test',
            'timings': {'cache_n': 0, 'predicted_per_second': 20.0, 'predicted_ms': 100.0}}


class SpecbenchTests(unittest.TestCase):
    def test_response_and_cache_validation(self):
        for field, value in [(None, None), ('cache_n', 10), ('predicted_ms', 0),
                             ('predicted_per_second', float('nan'))]:
            data = response()
            if field:
                data['timings'][field] = value
            with self.subTest(field=field), mock.patch('urllib.request.urlopen',
                    return_value=io.BytesIO(json.dumps(data).encode())) as post:
                if field:
                    with self.assertRaises(ValueError):
                        specbench.complete('http://localhost:8098', {})
                else:
                    self.assertEqual(specbench.complete('http://localhost:8098', {})['response'], data)
                    self.assertEqual(post.call_args.args[0].full_url, 'http://localhost:8098/completion')

    def test_missing_tokens(self):
        data = response(); data['tokens'] = []
        with mock.patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(data).encode())):
            with self.assertRaises(ValueError):
                specbench.complete('http://localhost:8098', {})

    def test_saved_runs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / 'prompt.txt'; prompt.write_text('question')
            output = Path(tmp) / 'result.json'
            argv = ['specbench', str(prompt), '--label', 'test', '--output', str(output), '-r', '2']
            with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(specbench, 'complete', side_effect=lambda *args: {
                        'wall_ms': 150.0, 'response': response()}) as complete:
                    specbench.main()
                    self.assertEqual(complete.call_count, 3)
                    self.assertFalse(complete.call_args.args[1]['cache_prompt'])
                    self.assertTrue(complete.call_args.args[1]['return_tokens'])
                data = json.loads(output.read_text())
                self.assertTrue(data['complete'])
                self.assertEqual([r['warmup'] for r in data['runs']], [True, False, False])
                with mock.patch.object(specbench, 'complete') as complete:
                    with self.assertRaises(FileExistsError):
                        specbench.main()
                    complete.assert_not_called()

    def test_failure_retains_partial_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / 'prompt.txt'; prompt.write_text('question')
            output = Path(tmp) / 'result.json'
            argv = ['specbench', str(prompt), '--label', 'test', '--output', str(output)]
            with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()), \
                 mock.patch.object(specbench, 'complete', side_effect=[
                     {'wall_ms': 150.0, 'response': response()}, ValueError('failed')]):
                with self.assertRaises(ValueError):
                    specbench.main()
            data = json.loads(output.read_text())
            self.assertFalse(data['complete'])
            self.assertEqual(len(data['runs']), 1)


if __name__ == '__main__':
    unittest.main()
