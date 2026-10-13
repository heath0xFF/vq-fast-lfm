"""Run with python3 -m unittest discover -s tests -v. No models or network required."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import d1_calibration
import d1_fidelity
import prof_agg


class ProfileTests(unittest.TestCase):
    def test_recorded_profile(self):
        with (ROOT / 'results/raw/E02-npu-profile/log.txt').open(errors='replace') as source:
            batches = prof_agg.parse(source)
        self.assertEqual(len(batches), 33)
        decode = [b for b in batches if prof_agg.token_count(b) == 1]
        self.assertEqual(len(decode), 32)
        self.assertEqual(prof_agg.token_count(batches[0]), 64)
        self.assertAlmostEqual(sum(b['usec'] for b in decode) / len(decode) / 1000, 42.57, places=2)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            prof_agg.summarize(batches)
        self.assertIn('tokens/pass=1: 32 batches', output.getvalue())
        self.assertIn('| MUL_MAT_NX | 38.00 | 17.359 |', output.getvalue())

    def test_incomplete_or_invalid_profile(self):
        header = 'HTP0 profile-op OPBATCH|----|n-ops 1|----|----|----|usec 100\n'
        op = 'HTP0 profile-op RMS_NORM|norm|2048:8 -> 2048:8|f32 -> f32|----|usec 50\n'
        for text in ['', header, op, header + op + op, header + 'profile-op invalid\n',
                     header + op + header.replace('HTP0', 'HTP1') + op.replace('HTP0', 'HTP1')]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                prof_agg.parse(text.splitlines())
        self.assertEqual(prof_agg.token_count(prof_agg.parse((header + op).splitlines())[0]), 8)

    def test_unknown_shape_is_not_called_decode(self):
        self.assertIsNone(prof_agg.token_count({'ops': [('GET_ROWS', 10, '4096:1 -> 4096:1')]}))
        self.assertIsNone(prof_agg.token_count({'ops': [
            ('RMS_NORM', 10, '128:32:64 -> 128:32:64'),
            ('RMS_NORM', 10, '2048:1 -> 2048:1'),
        ]}))


class CalibrationTests(unittest.TestCase):
    def test_evaluation_set_is_unchanged(self):
        digest = hashlib.sha256(json.dumps(d1_fidelity.dataset(1), sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, 'c91c7f4ce2fa80987be03e203a7f8b6922f92ec7fc73e13cb1a9a40c87a09ace')

    def test_filter_historical_prompts(self):
        original = (ROOT / 'data/d1/calib.txt').read_text()
        clean, removed, kept = d1_calibration.filter_prompts(original)
        self.assertEqual((removed, kept), (153, 567))
        self.assertEqual(d1_calibration.filter_prompts(clean), (clean, 0, 567))
        for state, _ in d1_fidelity.dataset(1):
            self.assertNotIn(d1_calibration.START + state + d1_calibration.QUESTION, clean)

    def test_invalid_calibration(self):
        all_overlap = (d1_calibration.START + d1_fidelity.dataset(1)[0][0]
                       + d1_calibration.QUESTION + 'Question' + d1_calibration.END + '\n')
        for text in ['', 'wrong format', d1_calibration.START + 'missing question', all_overlap]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                d1_calibration.filter_prompts(text)

    def test_cli_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'clean.txt'
            command = [sys.executable, str(ROOT / 'scripts/d1_calibration.py'),
                       str(ROOT / 'data/d1/calib.txt')]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
            self.assertEqual(subprocess.run(command + ['--output', str(out)], capture_output=True).returncode, 0)
            before = out.read_bytes()
            self.assertNotEqual(subprocess.run(command + ['--output', str(out)], capture_output=True).returncode, 0)
            self.assertEqual(out.read_bytes(), before)


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!{sys.executable}\n' + body)
    path.chmod(0o755)


class RunnerTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self, kind, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / 'scripts'
            scripts.mkdir()
            home = root / 'home'
            lfm = home / 'lfm-fast'
            lfm.mkdir(parents=True)
            mock = root / 'mock'
            pidfile = root / 'child.pid'
            env = {**os.environ, 'HOME': str(home), 'PATH': f'{mock}:{os.environ["PATH"]}',
                   'MOCK_PID': str(pidfile), 'MOCK_ROOT': str(root), 'TMPDIR': str(root), **overrides}
            server_code = '''import os, pathlib, signal, time
if os.environ.get('MOCK_START_FAIL'): raise SystemExit(23)
pathlib.Path(os.environ['MOCK_PID']).write_text(str(os.getpid()))
signal.signal(signal.SIGTERM, lambda *_: exit(0))
while True: time.sleep(0.02)
'''
            board_bin = root / 'build/bin'
            metal_bin = root / 'third_party/llama.cpp-master/build-mac/bin'
            for binary in (board_bin / 'llama-server', metal_bin / 'llama-server'):
                executable(binary, server_code)
            (lfm / 'lenv-master.sh').write_text(f'export LFM="{lfm}"\nexport L="{board_bin.parent}"\n')
            executable(mock / 'curl', '''import os, pathlib
if os.environ.get('MOCK_BUSY'):
    print('{"status":"loading model"}'); raise SystemExit(0)
try: os.kill(int(pathlib.Path(os.environ['MOCK_PID']).read_text()), 0)
except (OSError, ValueError): raise SystemExit(7)
if os.environ.get('MOCK_UNHEALTHY'): raise SystemExit(22)
print('{"status":"ok"}')
''')
            executable(mock / 'sleep', 'import time\ntime.sleep(0.01)\n')
            executable(mock / 'pkill', "raise SystemExit('pkill must never be used')\n")
            executable(mock / 'python3', '''import os
if os.environ.get('MOCK_CLIENT_FAIL'): raise SystemExit(19)
print('mock fidelity result')
''')
            executable(metal_bin / 'llama-quantize', '''import json, os, pathlib, sys
pathlib.Path(os.environ['MOCK_ROOT'], 'quant-args.json').write_text(json.dumps(sys.argv[1:]))
pathlib.Path(sys.argv[-2]).write_bytes(b'fake model')
if os.environ.get('MOCK_QUANT_FAIL'): raise SystemExit(17)
''')
            source = ROOT / f'scripts/d1_{kind}.sh'
            target = scripts / source.name
            shutil.copyfile(source, target)
            if kind == 'run':
                args = ['bash', str(target), 'offline-test', 'none', '',
                        'exit 19' if overrides.get('MOCK_CLIENT_FAIL') else 'echo mock client result']
            else:
                args = ['bash', str(target), 'offline-test', 'Q4_K_M']
            try:
                yield root, env, args
            finally:
                if pidfile.exists():
                    pid = int(pidfile.read_text())
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        os.kill(pid, signal.SIGTERM)
                        self.fail('runner left its child alive')

    def test_success_and_owned_cleanup(self):
        for kind in ('run', 'qlab'):
            with self.subTest(kind=kind), self.fixture(kind) as (root, env, args):
                result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((root / 'child.pid').exists())
                if kind == 'run':
                    runs = list((root / 'home/lfm-fast/results').iterdir())
                    self.assertEqual(len(runs), 1)
                    for name in ('cmd.txt', 'log.txt', 'client.txt'):
                        self.assertTrue((runs[0] / name).exists())
                else:
                    self.assertIn('mock fidelity result', (root / 'results/d1_quant.txt').read_text())
                    quant = json.loads((root / 'quant-args.json').read_text())
                    self.assertEqual(quant[:2], ['--imatrix', 'data/d1/imatrix-d1.dat'])
                    self.assertEqual(quant[-1], 'Q4_K_M')

    def test_busy_port_start_and_client_failures(self):
        for kind in ('run', 'qlab'):
            for failure in ('MOCK_BUSY', 'MOCK_START_FAIL', 'MOCK_CLIENT_FAIL', 'MOCK_UNHEALTHY'):
                with self.subTest(kind=kind, failure=failure), self.fixture(kind, **{failure: '1'}) as (root, env, args):
                    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=15)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    if failure == 'MOCK_BUSY':
                        self.assertFalse((root / 'child.pid').exists())
                    if kind == 'qlab':
                        self.assertFalse((root / 'results/d1_quant.txt').exists())

    def test_new_imatrix_and_extra_quantize_args(self):
        with self.fixture('qlab', IMATRIX='data/d1/new.dat') as (root, env, args):
            result = subprocess.run(args + ['imat', '--leave-output-tensor'], env=env,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            quant = json.loads((root / 'quant-args.json').read_text())
            self.assertEqual(quant[:3], ['--imatrix', 'data/d1/new.dat', '--leave-output-tensor'])

    def test_failed_quantization_is_not_cached(self):
        with self.fixture('qlab', MOCK_QUANT_FAIL='1') as (root, env, args):
            result = subprocess.run(args, env=env, capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list((root / 'models/d1q').iterdir()), [])
            self.assertFalse((root / 'child.pid').exists())

    def test_signal_cleans_up_child(self):
        for kind in ('run', 'qlab'):
            with self.subTest(kind=kind), self.fixture(kind, MOCK_UNHEALTHY='1') as (root, env, args):
                with subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
                    for _ in range(200):
                        if (root / 'child.pid').exists() or process.poll() is not None:
                            break
                        time.sleep(0.01)
                    self.assertTrue((root / 'child.pid').exists())
                    process.terminate()
                    process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 143)


class EnvironmentTests(unittest.TestCase):
    def test_master_libraries_take_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            lfm = Path(tmp) / 'lfm-fast'
            lfm.mkdir()
            for name in ('env.sh', 'lenv-master.sh'):
                shutil.copyfile(ROOT / 'scripts' / name, lfm / name)
            result = subprocess.run(['bash', '-c', 'source "$HOME/lfm-fast/lenv-master.sh" && '
                                     'printf "%s\\n" "$L" "$LD_LIBRARY_PATH" "$ADSP_LIBRARY_PATH" "$D_Q8"'],
                                    env={**os.environ, 'HOME': tmp, 'D_Q8': '/custom/draft.gguf'},
                                    capture_output=True, text=True, check=True)
            lines = result.stdout.splitlines()
            self.assertEqual(lines[0], str(lfm / 'lcpp-master'))
            self.assertTrue(lines[1].startswith(str(lfm / 'lcpp-master/lib') + ':'))
            self.assertEqual(lines[2], str(lfm / 'lcpp-master/lib'))
            self.assertEqual(lines[3], '/custom/draft.gguf')


if __name__ == '__main__':
    unittest.main()
