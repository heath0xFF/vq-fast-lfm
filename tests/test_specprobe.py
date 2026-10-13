"""Offline checks for the no-drafter prefix control."""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import specprobe


class PrefixProbeTests(unittest.TestCase):
    def setUp(self):
        self.reference = {'complete': True, 'prompts': {'chat.txt': 'hello'}, 'request_options': {},
                          'runs': [{'prompt': 'chat.txt', 'warmup': False,
                                    'response': {'tokens': [10, 11, 12]}}]}
        self.responses = [
            {'tokens': [10, 11], 'completion_probabilities': [{}, {}],
             'generation_settings': {'speculative.types': 'none,none'},
             'timings': {'prompt_n': 2, 'cache_n': 0}},
            {'tokens': [1, 2]},
            {'tokens': [99], 'completion_probabilities': [{}],
             'timings': {'prompt_n': 3, 'cache_n': 0}},
        ]

    def test_same_prefix_and_different_argmax(self):
        with mock.patch.object(specprobe, 'post', side_effect=self.responses) as post:
            result = specprobe.probe('http://localhost', self.reference, 'chat', 1)
        self.assertFalse(result['same_next_token'])
        self.assertTrue(result['reference_match'])
        self.assertEqual(result['prefix_tokens'], [1, 2, 10])
        self.assertEqual(post.call_args_list[-1].args[2]['prompt'], [1, 2, 10])
        self.assertFalse(post.call_args_list[-1].args[2]['cache_prompt'])

    def test_reject_invalid_controls(self):
        for field in ('speculation', 'cache', 'prefix', 'probabilities'):
            responses = copy.deepcopy(self.responses)
            if field == 'speculation':
                responses[0]['generation_settings']['speculative.types'] = 'none,draft-dspark'
            elif field == 'cache':
                responses[2]['timings']['cache_n'] = 1
            elif field == 'prefix':
                responses[2]['timings']['prompt_n'] = 4
            else:
                responses[0]['completion_probabilities'] = []
            with self.subTest(field=field), mock.patch.object(specprobe, 'post', side_effect=responses):
                with self.assertRaises(ValueError):
                    specprobe.probe('http://localhost', self.reference, 'chat', 1)

    def test_reference_position_validation(self):
        with mock.patch.object(specprobe, 'post') as post:
            for name, index in [('chat', 0), ('chat', 3), ('missing', 1)]:
                with self.subTest(name=name, index=index), self.assertRaises(ValueError):
                    specprobe.probe('http://localhost', self.reference, name, index)
            post.assert_not_called()


if __name__ == '__main__':
    unittest.main()
