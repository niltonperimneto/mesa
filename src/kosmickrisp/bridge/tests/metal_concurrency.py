import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess

COMPILATIONS = {'library', 'compute', 'render', 'specialize'}
CAPTURED_PIPELINES = {'compute', 'render', 'specialize'}
ARCHIVES = {'archive_compute', 'archive_render'}
PHASE_OPERATIONS = COMPILATIONS | ARCHIVES | {
    'serialize', 'compiler_read_lock', 'compiler_write_lock',
    'recipe_prepare', 'recipe_wait', 'fs_generate', 'fs_variant_wait',
    'pso_wait', 'native_pipeline', 'native_lookup', 'native_specialize',
    'archive_load', 'recapture', 'serialization_wait',
    'serialization_drain', 'serialization_bypass', 'uncaptured_session',
}


def analyze(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    events = []
    starts = {}
    completed = {}
    latest_ns = 0
    for record in records:
        if 'phase' not in record:
            events.append(record)
            latest_ns = max(latest_ns, record['end_ns'])
            continue
        identity = (record['pid'], record['operation_id'])
        latest_ns = max(latest_ns, record['time_ns'])
        if record['phase'] == 'begin':
            if record['operation'] not in PHASE_OPERATIONS:
                raise ValueError('unknown phase operation')
            if identity in starts:
                raise ValueError('duplicate operation ID')
            starts[identity] = record
        elif record['phase'] == 'end':
            if identity not in starts or identity in completed:
                raise ValueError('unmatched phase end')
            if record['time_ns'] < starts[identity]['time_ns']:
                raise ValueError('negative phase duration')
            completed[identity] = record
        else:
            raise ValueError('unknown phase')
    unfinished = []
    durations = {}
    for identity, start in starts.items():
        context = start
        parents = set()
        while not context.get('origin') and context.get('parent_operation_id'):
            parent = (context['pid'], context['parent_operation_id'])
            if parent in parents or parent not in starts:
                break
            parents.add(parent)
            context = starts[parent]
        if identity in completed:
            duration = completed[identity]['time_ns'] - start['time_ns']
            stats = durations.setdefault(start['operation'], {
                'count': 0, 'total_ns': 0, 'max_ns': 0,
                'draw_count': 0, 'draw_duration_ns': 0,
            })
            stats['count'] += 1
            stats['total_ns'] += duration
            stats['max_ns'] = max(stats['max_ns'], duration)
            if context.get('origin') == 'draw':
                stats['draw_count'] += 1
                stats['draw_duration_ns'] += duration
        else:
            unfinished.append(dict(start, elapsed_ns=latest_ns - start['time_ns'],
                                   context_key=context.get('key', ''),
                                   context_origin=context.get('origin', '')))
    groups = defaultdict(list)
    for event in events:
        if event['operation'] not in COMPILATIONS | ARCHIVES | {'serialize'}:
            raise ValueError('unknown operation')
        if event['end_ns'] < event['start_ns']:
            raise ValueError('negative operation duration')
        groups[(event['pid'], event['object'])].append(event)
    violations = []
    library_serialization_overlap_ns = 0
    parallel = {'compile': 0, 'archive': 0}
    overlap_ns = Counter()
    for identity, group in groups.items():
        active = []
        for event in sorted(group, key=lambda e: e['start_ns']):
            active = [a for a in active if a['end_ns'] > event['start_ns']]
            operation = event['operation']
            category = 'compile' if operation in COMPILATIONS else 'archive' if operation in ARCHIVES else None
            matching = [a for a in active if a['operation'] in (COMPILATIONS if category == 'compile' else ARCHIVES)] if category else []
            if category:
                threads = {a['thread_id'] for a in matching} | {event['thread_id']}
                parallel[category] = max(parallel[category], len(threads))
                for other in matching:
                    if other['thread_id'] != event['thread_id']:
                        overlap_ns[category] += min(other['end_ns'], event['end_ns']) - event['start_ns']
            for other in active:
                if {operation, other['operation']} == {'library', 'serialize'}:
                    library_serialization_overlap_ns += min(other['end_ns'], event['end_ns']) - event['start_ns']
                if ('serialize' in (operation, other['operation']) and
                    (operation in CAPTURED_PIPELINES or other['operation'] in CAPTURED_PIPELINES or operation == other['operation'])):
                    violations.append({'identity': identity, 'operations': [other, event]})
            active.append(event)
    uncaptured = {(e['pid'], e['object']) for e in starts.values()
                  if e['operation'] == 'uncaptured_session'}
    bypass_overlap_ns = 0
    serializers = [e for e in events if e['operation'] == 'serialize']
    for event in events:
        if event['operation'] not in CAPTURED_PIPELINES or (event['pid'], event['object']) not in uncaptured:
            continue
        for serializer in serializers:
            if serializer['pid'] != event['pid'] or serializer['object'] == event['object']:
                continue
            bypass_overlap_ns += max(0, min(serializer['end_ns'], event['end_ns']) -
                                     max(serializer['start_ns'], event['start_ns']))
    return {'events': len(events), 'operations': dict(Counter(e['operation'] for e in events)),
            'failures': dict(Counter(e['operation'] for e in events if not e['success'])),
            'maximum_simultaneous_threads': parallel, 'pairwise_overlap_ns': dict(overlap_ns),
            'library_serialization_overlap_ns': library_serialization_overlap_ns,
            'serialization_violations': violations,
            'uncaptured_serialization_overlap_ns': bypass_overlap_ns,
            'phase_events': len(records) - len(events),
            'phase_operations': dict(Counter(e['operation'] for e in starts.values())),
            'phase_failures': dict(Counter(starts[k]['operation'] for k, e in completed.items()
                                           if not e['success'])),
            'phase_durations': durations,
            'unfinished_operations': unfinished}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--arch', choices=['arm64', 'x86_64'])
    parser.add_argument('--require-overlap', action='store_true')
    parser.add_argument('--require-bypass-overlap', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.build:
        if not args.arch:
            parser.error('--build requires --arch')
        if args.trace.exists():
            parser.error('probe trace must not already exist')
        source = Path(__file__).resolve().parents[4]
        binary = args.out / 'metal-concurrency'
        command = ['clang', '-arch', args.arch, '-O2', '-Wno-unguarded-availability-new',
                   '-I' + str(source / 'src'), str(Path(__file__).with_suffix('.m')),
                   str(args.build / 'src/kosmickrisp/bridge/libmtl_bridge.a.p/mtl_compiler.m.o'),
                   '-framework', 'Foundation', '-framework', 'Metal', '-pthread', '-o', str(binary)]
        compilation = subprocess.run(command, capture_output=True, text=True)
        (args.out / "compile.log").write_text(compilation.stdout + compilation.stderr)
        if compilation.returncode:
            print(compilation.stderr)
            return compilation.returncode
        archive_file = args.out / 'pipeline-archive.bin'
        env = dict(os.environ, MESA_KK_METAL_CONCURRENCY_LOG=str(args.trace.resolve()),
                   KK_TEST_ARCHIVE_FILE=str(archive_file.resolve()))
        env.pop('MESA_KK_PSO_CACHE_DISABLE', None)
        env.pop('MESA_SHADER_CACHE_DISABLE', None)
        with (args.out / 'probe.log').open('w') as log:
            result = subprocess.run([str(binary)], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            print((args.out / 'probe.log').read_text())
            return result.returncode
        with (args.out / 'readback-restart.log').open('w') as log:
            restart = subprocess.run([str(binary), '--read-archive'], env=env,
                                     stdout=log, stderr=subprocess.STDOUT, timeout=180)
        if restart.returncode:
            print((args.out / 'readback-restart.log').read_text())
            return restart.returncode
    report = analyze(args.trace)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['events'] or report['serialization_violations'] or report['unfinished_operations']:
        return 1
    if any(report['failures'].get(op, 0) for op in COMPILATIONS | ARCHIVES):
        return 1
    if args.require_bypass_overlap and report['uncaptured_serialization_overlap_ns'] <= 0:
        return 1
    if args.require_overlap and any(report['maximum_simultaneous_threads'][c] < 2 for c in ['compile', 'archive']):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
