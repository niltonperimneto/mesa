import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pso_compare as compare


class ComparisonTests(unittest.TestCase):
    def test_percentiles(self):
        self.assertEqual(compare.summarize([]), {'count': 0})
        report = compare.summarize([0, 1_000_000, 2_000_000, 3_000_000])
        self.assertEqual(report['median_ms'], 1.5)
        self.assertAlmostEqual(report['p95_ms'], 2.85)
        self.assertAlmostEqual(report['p99_ms'], 2.97)

    def row(self, build, status='pass', trial=0, output='result=PASS', count=2):
        return dict(mode='measurement', workload='probe', trial=trial, phase='cold',
                    build=build, status=status, correctness_lines=[output],
                    metrics=[dict(kind='draw_record', ns=1_000_000, **{'pass': n})
                             for n in range(count)])

    def test_only_successful_matched_pairs_contribute_timings(self):
        rows = [self.row('baseline'), self.row('candidate'),
                self.row('baseline', trial=1), self.row('candidate', 'fail', trial=1),
                self.row('candidate', trial=2), self.row('baseline', trial=3),
                self.row('candidate', trial=3, output='result=WRONG'),
                self.row('baseline', trial=4), self.row('candidate', trial=4, count=1)]
        result = compare.comparison(rows, {'probe': {'metric': 'draw_record'}})
        self.assertEqual(result['matched_pairs'], 1)
        self.assertEqual(result['failed_runs'], 1)
        self.assertEqual(len(result['correctness_mismatches']), 2)
        self.assertEqual(result['timings'][0]['baseline']['count'], 1)
        self.assertEqual(result['timings'][1]['candidate']['count'], 1)

    def test_measurement_environment_and_diagnostic_separation(self):
        workload = {'env': {'MESA_KK_PSO_WAIT_TIMEOUT_MS': '10',
                            'MESA_KK_CACHE_LOG': 'inherited.log', 'KK_PROBE_THREADS': '4'}}
        with patch.dict(os.environ, {'MESA_KK_METAL_CONCURRENCY_LOG': 'parent.log',
                                     'MESA_SHADER_CACHE_DISABLE': 'true',
                                     'KK_PROBE_SAMPLE_MASK': '5'}):
            env = compare.environment(workload, Path('/cache'), Path('/run'), False)
            self.assertFalse(any(key.endswith('_LOG') for key in env if key.startswith('MESA_KK_')))
            self.assertNotIn('MESA_SHADER_CACHE_DISABLE', env)
            self.assertNotIn('KK_PROBE_SAMPLE_MASK', env)
            self.assertEqual(env['MESA_KK_PSO_WAIT_TIMEOUT_MS'], '0')
            self.assertEqual(env['KK_PROBE_THREADS'], '4')
            self.assertEqual(env['MESA_SHADER_CACHE_DIR'], '/cache/mesa')
            diagnostic = compare.environment(workload, Path('/cache'), Path('/run'), True)
            self.assertEqual(diagnostic['MESA_KK_METAL_CONCURRENCY_LOG'], '/run/metal.jsonl')

    def test_stack_categories(self):
        self.assertEqual(compare.classify_stack('disk_cache_destroy util_queue_finish cnd_wait'),
                         ['disk_cache_teardown'])
        self.assertEqual(compare.classify_stack('kk_recipe_get kk_pso_cache_get cnd_wait'),
                         ['recipe_or_pso_wait'])

    def test_watchdog_terminates_only_its_process(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
        try:
            with tempfile.TemporaryDirectory() as temporary, patch.object(
                    compare, 'capture_stack', return_value={'status': 'test'}) as capture:
                directory = Path(temporary) / 'run'
                row = compare.execute([sys.executable, '-c', 'import time; time.sleep(10)'],
                                      os.environ.copy(), directory, .2, .05)
                self.assertTrue(row['timed_out'])
                self.assertNotEqual(row['returncode'], 0)
                self.assertEqual(capture.call_count, 2)
                self.assertIsNone(unrelated.poll())
                self.assertLess(row['elapsed_ns'], 3_000_000_000)
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=3)

    def test_launch_failure_and_invalid_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = compare.run({'command': ['/nonexistent-pso-probe'], 'metric': 'draw_record'},
                              root / 'driver', root / 'cache', root / 'missing', 2, 1, False)
            self.assertEqual(row['status'], 'fail')
            self.assertIn('launch_error', row)
            workload = {'command': [sys.executable, '-c',
                                    'print(\'METRIC {"kind":"draw_record","ns":-1}\')'],
                        'metric': 'draw_record'}
            row = compare.run(workload, root / 'driver', root / 'cache', root / 'bad', 2, 1, False)
            self.assertEqual(row['status'], 'fail')
            self.assertIn('invalid timing metric', row['reasons'])

    def test_truncated_trace_preserves_unfinished_operation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            start = dict(phase='begin', operation='render', operation_id=1,
                         pid=1, thread_id=1, time_ns=10, object='a')
            raw = json.dumps(start) + '\n{"phase":"end"'
            (directory / 'metal.jsonl').write_text(raw)
            result = compare.diagnostics(directory)
            self.assertTrue(result['metal']['truncated_last_record'])
            self.assertEqual(len(result['metal']['unfinished_operations']), 1)
            self.assertEqual((directory / 'metal.jsonl').read_text(), raw)

    def test_archive_miss_is_not_a_compilation_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'stdout.log').write_text('result=PASS\nMETRIC {"kind":"draw_record","ns":1}\n')
            begin = dict(phase='begin', operation='archive_load', operation_id=1,
                         pid=1, thread_id=1, time_ns=10, object='a')
            end = dict(phase='end', operation_id=1, pid=1, time_ns=20, success=False)
            (directory / 'metal.jsonl').write_text(json.dumps(begin) + '\n' + json.dumps(end) + '\n')
            row = compare.evaluate({'metric': 'draw_record'}, dict(timed_out=False, returncode=0), directory, True)
            self.assertEqual(row['status'], 'pass')
            self.assertEqual(row['diagnostics']['archive_load_misses'], 1)
            begin['operation'] = 'render'
            (directory / 'metal.jsonl').write_text(json.dumps(begin) + '\n' + json.dumps(end) + '\n')
            row = compare.evaluate({'metric': 'draw_record'}, dict(timed_out=False, returncode=0), directory, True)
            self.assertEqual(row['status'], 'fail')

    def invoke(self, root, worker):
        baseline, candidate = root / 'baseline', root / 'candidate'
        baseline.write_bytes(b'baseline')
        candidate.write_bytes(b'candidate')
        manifest = root / 'workloads.json'
        manifest.write_text(json.dumps({'workloads': [dict(name='probe', command=[
            sys.executable, '-c', worker, '{icd}', '{cache_dir}', '{out_dir}'],
            metric='draw_record', success_regex='result=PASS', inputs=[str(baseline)])]}))
        output = root / 'report'
        argv = ['pso_compare.py', '--baseline-icd', str(baseline), '--candidate-icd', str(candidate),
                '--workloads', str(manifest), '--out', str(output), '--repetitions', '2',
                '--timeout', '5', '--capture-after', '4']
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            code = compare.main()
        return code, json.loads((output / 'report.json').read_text())

    def test_full_suite_cache_isolation_and_provenance(self):
        worker = '\n'.join([
            'import json, os, pathlib, sys',
            'cache = pathlib.Path(sys.argv[2]) / "marker"',
            'warm = cache.exists()',
            'cache.write_text(sys.argv[1])',
            'print(json.dumps({"warm": warm, "logged": "MESA_KK_COMPILE_LOG" in os.environ}))',
            'print("METRIC " + json.dumps({"kind":"draw_record","ns":1000000,"pass":0}))',
            'print("METRIC " + json.dumps({"kind":"draw_record","ns":2000000,"pass":1}))',
            'print("result=PASS")',
        ])
        with tempfile.TemporaryDirectory() as temporary:
            code, report = self.invoke(Path(temporary), worker)
            self.assertEqual(code, 0)
            self.assertTrue(report['complete'])
            self.assertEqual(report['comparison']['matched_pairs'], 4)
            self.assertEqual(len(report['rows']), 12)
            self.assertNotEqual(report['drivers']['baseline']['sha256'], report['drivers']['candidate']['sha256'])
            self.assertEqual(len(report['inputs']), 1)
            for row in report['rows']:
                output = json.loads((Path(row['artifacts']) / 'stdout.log').read_text().splitlines()[0])
                self.assertEqual(output['warm'], row['phase'] == 'warm')
                self.assertEqual(output['logged'], row['mode'] == 'diagnostic')

    def test_failed_measurement_stops_sweep_and_runs_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            code, report = self.invoke(Path(temporary), 'raise SystemExit(7)')
            self.assertEqual(code, 1)
            self.assertFalse(report['complete'])
            self.assertEqual(report['stopped_after_measurement_failure'], 'probe')
            self.assertEqual(len(report['rows']), 5)
            self.assertEqual([r['mode'] for r in report['rows']], ['measurement'] + ['diagnostic'] * 4)


if __name__ == '__main__':
    unittest.main()
