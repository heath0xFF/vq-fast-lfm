"""Offline checks for the cross-board decision benchmark."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import decisionbench


def response(body):
    answers = {}
    for name, q in body['questions'].items():
        a = {'type': q['type']}
        if q['type'] == 'noul':
            a['noul'] = .5
        else:
            labels = list(q['criteria'])
            a['probabilities'] = {label: float(i == 0) for i, label in enumerate(labels)}
            if q['type'] == 'choice':
                a['choice'] = labels[0]
            else:
                a['score'] = 0
        answers[name] = a
    return {'answers': answers, 'usage': {'input_tokens': 224}}


class DecisionbenchTests(unittest.TestCase):
    def test_requests_and_validation(self):
        bodies = decisionbench.requests('text', 105)
        self.assertEqual(bodies, decisionbench.requests('text', 105))
        self.assertEqual(len({b['state'] for b in bodies}), 105)
        for body in bodies + decisionbench.requests('image', 5):
            decisionbench.validate(body, response(body))
        body = bodies[0]
        for change in ('missing', 'probability', 'choice', 'score', 'tokens'):
            bad = copy.deepcopy(response(body))
            if change == 'missing': del bad['answers']['alert']
            if change == 'probability': bad['answers']['alert']['noul'] = float('nan')
            if change == 'choice': bad['answers']['category']['choice'] = 'unknown'
            if change == 'score': bad['answers']['severity']['score'] = 3
            if change == 'tokens': bad['usage']['input_tokens'] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                decisionbench.validate(body, bad)

    def test_audio_input(self):
        with self.assertRaises(ValueError):
            decisionbench.requests('audio', 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'clip.wav'
            for seconds in (1, 31):
                with wave.open(str(path), 'wb') as out:
                    out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                    out.writeframes(b'\0' * (16000 * seconds * 2))
                if seconds == 1:
                    body = decisionbench.requests('audio', 3, path)[0]
                    decisionbench.validate(body, response(body))
                else:
                    with self.assertRaises(ValueError):
                        decisionbench.requests('audio', 3, path)

    def test_save_summary_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            argv = ['decisionbench', 'text', '--label', 'test', '--output', str(path),
                    '-n', '4', '--warm', '1', '--conc', '2']
            with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(decisionbench, 'post', side_effect=lambda url, body: (response(body), 10)):
                    decisionbench.main()
                result = json.loads(path.read_text())
                self.assertTrue(result['complete'])
                self.assertEqual(len(result['runs']), 5)
                self.assertEqual(result['summary']['p95_ms'], 10)
                self.assertEqual([r['request_index'] for r in result['runs']], list(range(5)))
                with mock.patch.object(decisionbench, 'post') as post:
                    with self.assertRaises(FileExistsError): decisionbench.main()
                    post.assert_not_called()

    def test_preserve_partial_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'partial.json'
            argv = ['decisionbench', 'text', '--label', 'test', '--output', str(path), '--warm', '1']
            reply = response(decisionbench.requests('text', 1)[0])
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(decisionbench, 'post',
                    side_effect=[(reply, 10), ValueError('failed')]):
                with self.assertRaises(ValueError): decisionbench.main()
            data = json.loads(path.read_text())
            self.assertFalse(data['complete'])
            self.assertEqual(len(data['runs']), 1)


if __name__ == '__main__':
    unittest.main()
