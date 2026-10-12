import json
from pathlib import Path
import tempfile
import unittest
from metal_concurrency import analyze


def event(operation, start, end, thread, object='a', pid=1):
    return dict(operation=operation, start_ns=start, end_ns=end,
                thread_id=thread, object=object, pid=pid, success=True)


class TraceTests(unittest.TestCase):
    def report(self, events):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            path.write_text(''.join(json.dumps(e) + '\n' for e in events))
            return analyze(path)

    def test_overlap_and_endpoint(self):
        report = self.report([event('render', 0, 10, 1), event('compute', 5, 15, 2),
                              event('library', 15, 20, 3), event('archive_render', 0, 10, 1, 'b'),
                              event('archive_compute', 5, 15, 2, 'b')])
        self.assertEqual(report['maximum_simultaneous_threads'], {'compile': 2, 'archive': 2})
        self.assertEqual(report['pairwise_overlap_ns'], {'compile': 5, 'archive': 5})
        self.assertFalse(report['serialization_violations'])

    def test_serializer_exclusion(self):
        for operation in ['compute', 'render', 'serialize']:
            with self.subTest(operation=operation):
                report = self.report([event(operation, 0, 10, 1), event('serialize', 5, 15, 2)])
                self.assertEqual(len(report['serialization_violations']), 1)

    def test_library_serializer_overlap(self):
        report = self.report([event('library', 0, 10, 1), event('serialize', 5, 15, 2)])
        self.assertFalse(report['serialization_violations'])
        self.assertEqual(report['library_serialization_overlap_ns'], 5)

    def test_identities_and_threads(self):
        report = self.report([event('render', 0, 10, 1), event('compute', 0, 10, 1),
                              event('serialize', 0, 10, 2, 'b'), event('serialize', 0, 10, 2, 'a', 2)])
        self.assertEqual(report['maximum_simultaneous_threads']['compile'], 1)
        self.assertFalse(report['serialization_violations'])

    def test_uncaptured_serializer_overlap(self):
        begin = dict(phase='begin', operation='uncaptured_session', operation_id=1,
                     pid=1, thread_id=2, time_ns=0, object='b')
        end = dict(phase='end', operation_id=1, pid=1, time_ns=20, success=True)
        report = self.report([begin, event('render', 0, 10, 1, 'b'),
                              event('serialize', 5, 15, 2),
                              event('serialize', 0, 10, 2, 'a', 2), end])
        self.assertFalse(report['serialization_violations'])
        self.assertEqual(report['uncaptured_serialization_overlap_ns'], 5)

    def test_phases_and_legacy_records(self):
        begin = dict(phase='begin', operation='render', operation_id=1,
                     pid=1, thread_id=2, time_ns=10, object='a')
        end = dict(phase='end', operation_id=1, pid=1, time_ns=20, success=True)
        report = self.report([begin, event('render', 10, 20, 2), end])
        self.assertEqual(report['events'], 1)
        self.assertEqual(report['phase_events'], 2)
        self.assertEqual(report['phase_operations'], {'render': 1})
        self.assertFalse(report['unfinished_operations'])

    def test_unfinished_nested_phase(self):
        parent = dict(phase='begin', operation='recipe_prepare', operation_id=1,
                      pid=1, thread_id=2, time_ns=10, object='a',
                      origin='draw', key='abc')
        child = dict(phase='begin', operation='compiler_read_lock', operation_id=2,
                     parent_operation_id=1, pid=1, thread_id=2, time_ns=20,
                     object='b', origin='', key='')
        other_process = dict(phase='begin', operation='render', operation_id=1,
                             pid=2, thread_id=2, time_ns=30, object='a')
        report = self.report([parent, child, other_process])
        self.assertEqual(len(report['unfinished_operations']), 3)
        blocked = report['unfinished_operations'][1]
        self.assertEqual(blocked['context_key'], 'abc')
        self.assertEqual(blocked['context_origin'], 'draw')
        self.assertEqual(blocked['elapsed_ns'], 10)

    def test_completed_phase_duration_context(self):
        parent = dict(phase='begin', operation='recipe_prepare', operation_id=1,
                      pid=1, thread_id=2, time_ns=10, object='a', origin='draw', key='abc')
        child = dict(phase='begin', operation='render', operation_id=2,
                     parent_operation_id=1, pid=1, thread_id=2, time_ns=15,
                     object='a', origin='', key='')
        end = dict(phase='end', operation_id=2, pid=1, time_ns=35, success=True)
        report = self.report([parent, child, end, dict(end, operation_id=1, time_ns=40)])
        self.assertEqual(report['phase_durations']['render'],
                         dict(count=1, total_ns=20, max_ns=20, draw_count=1, draw_duration_ns=20))
        self.assertEqual(report['phase_durations']['recipe_prepare']['total_ns'], 30)
        self.assertFalse(report['unfinished_operations'])

    def test_invalid_phases(self):
        begin = dict(phase='begin', operation='render', operation_id=1,
                     pid=1, thread_id=2, time_ns=10, object='a')
        end = dict(phase='end', operation_id=1, pid=1, time_ns=20, success=False)
        self.assertEqual(self.report([begin, end])['phase_failures'], {'render': 1})
        for records in [[end], [begin, begin], [begin, end, end],
                        [begin, dict(end, time_ns=5)],
                        [dict(begin, operation='unknown')],
                        [dict(begin, phase='unknown')]]:
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.report(records)

    def test_invalid_and_empty(self):
        self.assertEqual(self.report([])['maximum_simultaneous_threads'], {'compile': 0, 'archive': 0})
        with self.assertRaises(ValueError):
            self.report([event('render', 10, 0, 1)])
        with self.assertRaises(ValueError):
            self.report([event('unknown', 0, 10, 1)])
        with self.assertRaises(KeyError):
            self.report([{}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            path.write_text('invalid json\n')
            with self.assertRaises(json.JSONDecodeError):
                analyze(path)


if __name__ == '__main__':
    unittest.main()
