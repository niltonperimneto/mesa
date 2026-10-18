import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import subprocess
import time

from metal_concurrency import analyze


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    if not values:
        return {'count': 0}
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return (ordered[low] + (ordered[high] - ordered[low]) * (position - low)) / 1e6

    return {'count': len(values), 'median_ms': statistics.median(values) / 1e6,
            'p95_ms': percentile(.95), 'p99_ms': percentile(.99),
            'min_ms': ordered[0] / 1e6, 'max_ms': ordered[-1] / 1e6}


def classify_stack(text):
    checks = [
        ('disk_cache_teardown', ('disk_cache_destroy', 'util_queue_finish')),
        ('metal_compiler', ('newRenderPipelineState',)),
        ('metal_compiler', ('newLibraryWithDescriptor',)),
        ('serialization', ('mtl_compiler_serialize',)),
        ('compiler_lock', ('pthread_rwlock',)),
        ('fragment_variant', ('kk_compile_fs_variant',)),
        ('recipe_or_pso_wait', ('kk_recipe_get', 'cnd_wait')),
        ('recipe_or_pso_wait', ('kk_pso_cache_get', 'cnd_wait')),
        ('fragment_variant_wait', ('kk_shader_get_fs_variant', 'cnd_wait')),
    ]
    return list(dict.fromkeys(name for name, symbols in checks
                              if all(symbol in text for symbol in symbols)))


def capture_stack(pid, path):
    if platform.system() != 'Darwin' or not Path('/usr/bin/sample').exists():
        return {'status': 'unavailable', 'pid': pid}
    try:
        result = subprocess.run(['/usr/bin/sample', str(pid), '1', '1', '-file', str(path)],
                                capture_output=True, text=True, timeout=5)
        text = path.read_text(errors='replace') if path.exists() else ''
        return {'status': 'captured' if result.returncode == 0 and text else 'failed',
                'pid': pid, 'path': str(path), 'returncode': result.returncode,
                'stderr': result.stderr, 'possible_blockers': classify_stack(text)}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'status': 'failed', 'pid': pid, 'error': str(error)}


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)


def execute(command, env, directory, timeout, capture_after=None):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    captures = []
    started = time.monotonic_ns()
    timed_out = False
    with (directory / 'stdout.log').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
        try:
            process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
        except OSError as error:
            return {'returncode': None, 'timed_out': False, 'launch_error': str(error),
                    'elapsed_ns': time.monotonic_ns() - started,
                    'captures': [], 'artifacts': str(directory)}
        try:
            while process.poll() is None:
                elapsed = (time.monotonic_ns() - started) / 1e9
                if elapsed >= timeout:
                    captures.append(capture_stack(process.pid, directory / 'stack-timeout.txt'))
                    timed_out = True
                    stop_process(process)
                    break
                if capture_after is not None and elapsed >= capture_after and not captures:
                    captures.append(capture_stack(process.pid, directory / 'stack-slow.txt'))
                    continue
                try:
                    process.wait(timeout=min(.1, timeout - elapsed))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            stop_process(process)
    return {'returncode': process.returncode, 'timed_out': timed_out,
            'elapsed_ns': time.monotonic_ns() - started,
            'captures': captures, 'artifacts': str(directory)}


def complete_lines(path):
    if not path.exists():
        return [], False
    lines = path.read_text().splitlines(keepends=True)
    truncated = bool(lines and not lines[-1].endswith('\n'))
    return (lines[:-1] if truncated else lines), truncated


def diagnostics(directory):
    result = {}
    metal = directory / 'metal.jsonl'
    if metal.exists():
        lines, truncated = complete_lines(metal)
        complete = directory / 'metal-complete.jsonl'
        complete.write_text(''.join(lines))
        result['metal'] = analyze(complete)
        result['metal']['truncated_last_record'] = truncated
    else:
        result['metal'] = None
    lines, truncated = complete_lines(directory / 'compiler.csv')
    counts = Counter()
    durations = Counter()
    draw_counts = Counter()
    for row in csv.reader(lines):
        if len(row) != 4:
            raise ValueError('invalid compiler record')
        kind, origin, start, end = row
        duration = int(end) - int(start)
        if duration < 0:
            raise ValueError('negative compiler duration')
        counts[kind] += 1
        durations[kind] += duration
        if origin == 'draw':
            draw_counts[kind] += 1
    result['compiler'] = {'counts': dict(counts), 'duration_ns': dict(durations),
                          'draw_counts': dict(draw_counts), 'truncated_last_record': truncated}
    lines, truncated = complete_lines(directory / 'cache.jsonl')
    records = [json.loads(line) for line in lines if line.strip()]
    outcomes = Counter(record.get('outcome') for record in records
                       if record.get('event') == 'pso')
    result['cache'] = {'outcomes': dict(outcomes), 'truncated_last_record': truncated}
    result['persistence'] = persistence([r for r in records if r.get('event') == 'persistence'])
    stats = directory / 'variant-stats.jsonl'
    result['variant'] = variant_stats(stats) if stats.exists() else None
    return result


def persistence(records):
    compiles = [r for r in records if r.get('outcome') == 'native_compile']
    reason_ns = Counter()
    for record in compiles:
        if record.get('duration_ns', 0) < 0:
            raise ValueError('negative native compile duration')
        reason_ns[record['reason']] += record.get('duration_ns', 0)
    return {'outcomes': dict(Counter(r.get('outcome') for r in records)),
            'native_compile_reasons': dict(Counter(r['reason'] for r in compiles)),
            'native_compile_routes': dict(Counter(r['route'] for r in compiles)),
            'native_compile_duration_ns': dict(reason_ns),
            'native_compile_failures': sum(not r.get('success', True) for r in compiles),
            'dropped': dict(Counter(r['reason'] for r in records
                                    if r.get('outcome') == 'persist_dropped')),
            'summary': next(({k: v for k, v in r.items() if k not in ('event', 'outcome')}
                             for r in reversed(records) if r.get('outcome') == 'cache_summary'),
                            None)}


def variant_stats(path):
    lines, truncated = complete_lines(path)
    phases, queue, cache = {}, None, None
    for record in map(json.loads, (line for line in lines if line.strip())):
        if record.get('event') == 'variant_summary':
            phases[record['phase']] = {key: record[key] for key in
                                       ('count', 'duration_ns', 'draw_count', 'draw_duration_ns')}
        elif record.get('event') == 'queue_summary':
            queue = {key: value for key, value in record.items() if key != 'event'}
        elif record.get('event') == 'cache_summary':
            cache = {key: value for key, value in record.items() if key != 'event'}
    return {'phases': phases, 'queue': queue, 'cache': cache, 'truncated_last_record': truncated}


def environment(workload, cache, directory, diagnostic):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('MESA_KK_', 'KK_PROBE_', 'MESA_SHADER_CACHE'))}
    env.update({'KK_PROBE_METRICS': '1', 'KK_PROBE_REPEAT_DRAWS': '10'})
    env.update(workload.get('env', {}))
    for key in list(env):
        if key.startswith('MESA_KK_') and (key.endswith('_LOG') or key == 'MESA_KK_VARIANT_STATS'):
            del env[key]
    env['MESA_SHADER_CACHE_DIR'] = str(cache / 'mesa')
    env['KK_PROBE_CACHE'] = str(cache / 'pipeline.cache')
    if diagnostic:
        env.update({'MESA_KK_METAL_CONCURRENCY_LOG': str(directory / 'metal.jsonl'),
                    'MESA_KK_COMPILE_LOG': str(directory / 'compiler.csv'),
                    'MESA_KK_CACHE_LOG': str(directory / 'cache.jsonl'),
                    'MESA_KK_VARIANT_STATS': str(directory / 'variant-stats.jsonl')})
    return env


def run(workload, driver, cache, directory, timeout, capture_after, diagnostic):
    cache.mkdir(parents=True, exist_ok=True)
    replacements = {'{icd}': str(driver), '{cache_dir}': str(cache), '{out_dir}': str(directory)}
    command = []
    for argument in workload['command']:
        for token, value in replacements.items():
            argument = argument.replace(token, value)
        command.append(argument)
    env = environment(workload, cache, directory, diagnostic)
    row = execute(command, env, directory, timeout, capture_after if diagnostic else None)
    row['environment'] = {key: value for key, value in env.items()
                          if key.startswith(('MESA_KK_', 'KK_PROBE_', 'MESA_SHADER_CACHE'))
                          or key in workload.get('env', {})}
    return evaluate(workload, row, directory, diagnostic)


def evaluate(workload, row, directory, diagnostic):
    output = (directory / 'stdout.log').read_text(errors='replace')
    reasons = []
    if row['timed_out']:
        reasons.append('process timeout, including teardown')
    elif row['returncode'] != 0:
        reasons.append('nonzero process exit')
    if workload.get('success_regex') and not re.search(workload['success_regex'], output):
        reasons.append('missing correctness evidence')
    if re.search(workload.get('failure_regex', r'\bresult=FAIL\b'), output):
        reasons.append('correctness failure')
    metrics = []
    try:
        for line in output.splitlines():
            if line.startswith('METRIC '):
                metric = json.loads(line[7:])
                if type(metric.get('ns')) is not int or metric['ns'] < 0 or not isinstance(metric.get('kind'), str):
                    raise ValueError('invalid timing metric')
                if type(metric.get('pass', 0)) is not int or metric.get('pass', 0) < 0:
                    raise ValueError('invalid metric pass')
                metrics.append(metric)
        if workload.get('metric') and not any(m['kind'] == workload['metric'] for m in metrics):
            reasons.append('missing requested timing metric')
    except (ValueError, TypeError, AttributeError) as error:
        reasons.append(str(error))
    row.update({'metrics': metrics, 'correctness_lines': [line for line in output.splitlines()
                                                         if 'result=' in line],
                'status': 'fail' if reasons else 'pass', 'reasons': reasons})
    if diagnostic:
        try:
            row['diagnostics'] = diagnostics(directory)
            metal = row['diagnostics']['metal']
            if metal:
                row['diagnostics']['archive_load_misses'] = metal['phase_failures'].get('archive_load', 0)
            phase_errors = {key: count for key, count in (metal or {}).get('phase_failures', {}).items()
                            if key != 'archive_load'}
            if metal and (phase_errors or any(metal[key] for key in (
                    'failures', 'unfinished_operations', 'serialization_violations'))):
                row['status'] = 'fail'
                row['reasons'].append('failed, unfinished, or conflicting Metal operations')
        except (ValueError, KeyError, TypeError) as error:
            row['diagnostic_error'] = str(error)
            row['status'] = 'fail'
            row['reasons'].append('invalid diagnostic trace')
    return row


def comparison(rows, workloads):
    groups = defaultdict(list)
    pairs = defaultdict(dict)
    for row in rows:
        if row['mode'] != 'measurement':
            continue
        pairs[(row['workload'], row['trial'], row['phase'])][row['build']] = row
    mismatches = []
    matched_pairs = 0
    for identity, pair in pairs.items():
        if set(pair) != {'baseline', 'candidate'} or any(row['status'] != 'pass' for row in pair.values()):
            continue
        if pair['baseline']['correctness_lines'] != pair['candidate']['correctness_lines']:
            mismatches.append(list(identity))
            continue
        workload = workloads[identity[0]]
        metrics = {build: [m for m in row['metrics'] if m['kind'] == workload.get('metric')]
                   for build, row in pair.items()}
        if [m.get('pass', 0) for m in metrics['baseline']] != [m.get('pass', 0) for m in metrics['candidate']]:
            mismatches.append(list(identity))
            continue
        matched_pairs += 1
        for build, values in metrics.items():
            for metric in values:
                group = 'initial' if metric.get('pass', 0) < workload.get('initial_passes', 1) else 'resident'
                groups[(identity[0], identity[2], build, group)].append(metric['ns'])
    timings = []
    for workload in workloads:
        for phase in ('cold', 'warm'):
            for group in ('initial', 'resident'):
                baseline = summarize(groups[(workload, phase, 'baseline', group)])
                candidate = summarize(groups[(workload, phase, 'candidate', group)])
                changes = {key: (candidate[key] / baseline[key] - 1) * 100
                           for key in ('median_ms', 'p95_ms', 'p99_ms')
                           if baseline.get(key, 0) > 0 and key in candidate}
                timings.append({'workload': workload, 'phase': phase, 'draws': group,
                                'baseline': baseline, 'candidate': candidate,
                                'change_percent': changes})
    return {'timings': timings, 'correctness_mismatches': mismatches,
            'matched_pairs': matched_pairs,
            'failed_runs': sum(row['status'] != 'pass' for row in rows)}


def load_workloads(path):
    manifest = json.loads(path.read_text())
    workloads = manifest['workloads']
    if not isinstance(workloads, list) or not workloads:
        raise ValueError('workloads must be a nonempty list')
    names = set()
    for workload in workloads:
        name = workload['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in names:
            raise ValueError('invalid or duplicate workload name')
        names.add(name)
        if not isinstance(workload['command'], list) or not workload['command'] or not all(
                isinstance(argument, str) and argument for argument in workload['command']):
            raise ValueError('command must be a nonempty argument list')
        env = workload.get('env', {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ValueError('environment values must be strings')
        if type(workload.get('initial_passes', 1)) is not int or workload.get('initial_passes', 1) < 1:
            raise ValueError('initial_passes must be positive')
        if not isinstance(workload.get('metric'), str) or not workload['metric']:
            raise ValueError('a timing metric is required')
        if not isinstance(workload.get('inputs', []), list) or not all(
                isinstance(value, str) for value in workload.get('inputs', [])):
            raise ValueError('inputs must be a list of paths')
        for key in ('success_regex', 'failure_regex'):
            if key in workload:
                re.compile(workload[key])
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-icd', type=Path, required=True)
    parser.add_argument('--candidate-icd', type=Path, required=True)
    parser.add_argument('--baseline-revision')
    parser.add_argument('--workloads', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--capture-after', type=float, default=5)
    parser.add_argument('--historical-sample', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.repetitions < 1 or not math.isfinite(args.timeout) or not 0 < args.capture_after < args.timeout:
        parser.error('positive repetitions and 0 < capture-after < timeout are required')
    try:
        manifest = load_workloads(args.workloads)
        args.out = args.out.resolve()
        args.out.mkdir(parents=True, exist_ok=False)
        drivers = {}
        for name, source in [('baseline', args.baseline_icd), ('candidate', args.candidate_icd)]:
            source = source.resolve(strict=True)
            target = args.out / 'drivers' / name / source.name
            target.parent.mkdir(parents=True)
            shutil.copy2(source, target)
            drivers[name] = {'source': str(source), 'path': str(target), 'sha256': digest(target)}
        workloads = {w['name']: w for w in manifest['workloads']}
        inputs = {}
        for workload in workloads.values():
            for filename in workload.get('inputs', []):
                path = Path(filename).resolve(strict=True)
                inputs[str(path)] = digest(path)
        references = []
        for index, path in enumerate(args.historical_sample):
            copied = args.out / f'historical-stack-{index}.txt'
            shutil.copy2(path, copied)
            references.append({'source': str(path.resolve()), 'sha256': digest(copied),
                               'possible_blockers': classify_stack(copied.read_text(errors='replace'))})
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    report = {'drivers': drivers, 'baseline_revision': args.baseline_revision,
              'platform': platform.platform(), 'manifest': manifest, 'inputs': inputs,
              'historical_samples': references, 'repetitions': args.repetitions,
              'timeout_seconds': args.timeout, 'capture_after_seconds': args.capture_after,
              'cold_cache_scope': 'fresh Mesa and application caches; system compiler caches retained',
              'percentiles': 'descriptive sample percentiles, without a confidence interval',
              'measurement_logging': False, 'rows': [], 'complete': False}

    def save():
        report['comparison'] = comparison(report['rows'], workloads)
        temporary = args.out / 'report.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.out / 'report.json')

    save()
    failed_workload = None
    try:
        for mode, repetitions in [('measurement', args.repetitions), ('diagnostic', 1)]:
            for name, workload in workloads.items():
                if mode == 'diagnostic' and failed_workload and name != failed_workload:
                    continue
                for trial in range(repetitions):
                    order = ['baseline', 'candidate'] if trial % 2 == 0 else ['candidate', 'baseline']
                    root = args.out / mode / name / f'trial-{trial:03d}'
                    for phase in ('cold', 'warm'):
                        for build in order:
                            directory = root / build / phase
                            row = run(workload, Path(drivers[build]['path']), root / build / 'cache',
                                      directory, args.timeout, args.capture_after, mode == 'diagnostic')
                            row.update({'mode': mode, 'workload': name, 'trial': trial,
                                        'phase': phase, 'build': build})
                            report['rows'].append(row)
                            save()
                            print(mode, name, trial, phase, build, row['status'], flush=True)
                            if mode == 'measurement' and row['status'] != 'pass':
                                failed_workload = name
                                break
                        if mode == 'measurement' and failed_workload:
                            break
                    if mode == 'measurement' and failed_workload:
                        break
                if mode == 'measurement' and failed_workload:
                    break
        report['complete'] = failed_workload is None
        report['stopped_after_measurement_failure'] = failed_workload
    finally:
        save()
    print(str(args.out / 'report.json'))
    return 1 if report['comparison']['failed_runs'] or report['comparison']['correctness_mismatches'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
