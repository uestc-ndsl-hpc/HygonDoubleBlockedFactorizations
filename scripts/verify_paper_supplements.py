#!/usr/bin/env python3
"""Recheck public numerical, stage-spread and failure-boundary evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'results/fp32/paper_final/supplements_20261021'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics_for(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    return {'mean_ms': mean, 'sample_sd_ms': sd, 'rsd_percent': 100 * sd / mean,
            'min_ms': min(values), 'max_ms': max(values)}


def verify(directory, check_sources=True):
    entries = (directory / 'SHA256SUMS').read_text().splitlines()
    listed = set()
    for line in entries:
        expected, relative = line.split('  ', 1)
        path = (directory / relative).resolve()
        require(path.is_relative_to(directory.resolve()), 'Unsafe checksum path')
        require(digest(path) == expected, 'Checksum mismatch: ' + relative)
        require(relative not in listed, 'Duplicate checksum entry')
        listed.add(relative)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
              if p.is_file() and p.name != 'SHA256SUMS'}
    require(listed == actual, 'Checksum manifest does not cover the exact file set')
    for entry in json.loads((directory / 'export_provenance.json').read_text()):
        require(digest(directory / entry['public_file']) == entry['public_sha256'],
                'Export provenance mismatch')

    folder = directory / 'fp64_residual'
    records = json.loads((folder / 'results.json').read_text())
    require(len(records) == 45, 'Expected 45 method/order/seed records')
    expected = set()
    methods = {'getrf': {'rocSOLVER GETRF', 'DoubleBlock-LU+BatchedPivot+MGEMM'},
               'potrf': {'rocSOLVER', 'DoubleBlock+KBSYRK', 'Recursive'}}
    seeds = {'getrf': 1234567890, 'potrf': 20260716}
    for kind in methods:
        for n in (8192, 16384, 24576, 32768, 40960):
            offsets = (0, 1, 2) if n in (8192, 24576) else (0,)
            for delta in offsets:
                for method in methods[kind]:
                    expected.add((kind, n, seeds[kind] + delta, method))
    require({(r['kind'], r['n'], r['seed'], r['method']) for r in records} == expected,
            'Incomplete/duplicate coverage')
    require(sum(r['cohort'] == 'paper_seed' for r in records) == 25, 'Paper-seed coverage')
    require(sum(r['cohort'] == 'extra_seed' for r in records) == 20, 'Extra-seed coverage')
    csv_records = rows(folder / 'results.csv')
    require(len(csv_records) == len(records), 'Summary CSV count mismatch')
    for r, c in zip(records, csv_records):
        require(all(str(r[k]) == c[k] for k in r), 'Summary JSON/CSV mismatch')
        require(r['info'] == 0 and math.isfinite(r['residual']) and r['residual'] >= 0,
                'Invalid info or residual')
        raw = [row for row in rows(folder / r['raw_csv']) if row['method'] == r['method']]
        require(len(raw) == 1, 'Duplicate/missing raw method')
        row = raw[0]
        require(row['residual_metric'] == 'cpu_fp64_full_relative_residual', 'Wrong metric')
        require(float(row['residual_value']) == r['residual'], 'Raw residual mismatch')
        require(row['parameters'] == r['parameters'] and row['iteration'] == '0', 'Raw parameters/sample count')
        n = r['n']
        if r['method'].startswith('DoubleBlock-LU'):
            parameters = f'{min(n//2,20480)}/{min(n//4,4096)}/iterative/2048/2048/8192'
        elif r['method'] == 'DoubleBlock+KBSYRK':
            parameters = '8192/1024/8192'
        elif r['method'] == 'Recursive':
            parameters = '2048'
        else:
            parameters = 'N/A'
        require(r['parameters'] == parameters, 'Unexpected parameter policy')
    jobs = json.loads((folder / 'jobs.json').read_text())
    require(len(jobs) == 18, 'Expected 18 benchmark invocations')
    for job in jobs:
        content = (folder / job['log']).read_text()
        count = len(methods[job['kind']])
        require(job['exit_code'] == 0, 'Unsuccessful invocation')
        require(content.count('CPU FP64 validation: info=0') == count, 'Missing info checks')
        require('CPU CBLAS DGEMM; tile=512; seed=' + str(job['seed']) in content,
                'Backend/tile/seed mismatch')
        require('validation excluded from GPU timing' in content and 'RecTiledTRSM' not in content,
                'Unexpected validation or recursive path')
        require(len(rows(folder / job['csv'])) == count, 'Unexpected raw CSV count')
    if check_sources:
        source = json.loads((folder / 'tested_source_sha256.json').read_text())
        for name, expected_hash in source['files'].items():
            data = (ROOT / name).read_bytes().replace(b'\r\n', b'\n')
            require(hashlib.sha256(data).hexdigest() == expected_hash,
                    'Source has changed since collection: ' + name)

    profiles = {}
    for file in (directory / 'stage_spread/raw').glob('*_N40960.log'):
        content = file.read_text()
        require('exit_code=0' in content, 'Unsuccessful stage profile')
        chunks = re.split(r'STAGE_SAMPLE (?:GETRF|POTRF) iteration=(\d+)', content)
        samples = []
        for i in range(1, len(chunks), 2):
            require(int(chunks[i]) == len(samples) + 1, 'Stage iteration order')
            values = {}
            for line in chunks[i+1].splitlines():
                fields = [part.strip() for part in line.split('|')]
                if len(fields) == 6 and re.fullmatch(r'\d+\.\d+', fields[1]):
                    values[fields[0]] = float(fields[1])
            require(bool(values), 'Missing stage rows')
            samples.append(values)
        require(len(samples) == 10, 'Expected ten measured stage samples')
        profiles[file.name.removesuffix('_N40960.log')] = samples
    require(len(profiles) == 6, 'Expected six stage cases')
    stages = {
        ('Chol. K-blocked SYRK', 'reference'): ['Outer SYRK'],
        ('Chol. K-blocked SYRK', 'optimized'): ['Outer SYRK'],
        ('LU batched pivots', 'reference'): ['Pivot propagation'],
        ('LU batched pivots', 'optimized'): ['Inner pivot batch', 'Outer pivot batch'],
        ('LU M-tiled GEMM', 'reference'): ['Inner GEMM'],
        ('LU M-tiled GEMM', 'optimized'): ['Inner GEMM'],
        ('LU iterative TRSM', 'reference'): ['Outer TRSM direct'],
        ('LU iterative TRSM', 'optimized'): ['Outer TRSM block', 'Outer TRSM GEMM']}
    spread_rows = rows(directory / 'stage_spread/stage_spread_statistics.csv')
    require(len(spread_rows) == 8, 'Stage comparison count')
    stage_rsd = []
    for row in spread_rows:
        components = stages[(row['mechanism'], row['variant'])]
        values = [sum(sample[s] for s in components) for sample in profiles[row['case']]]
        calculated = statistics_for(values)
        for key, value in calculated.items():
            require(abs(float(row[key]) - value) <= 0.000051, 'Stage statistic mismatch: ' + key)
        stage_rsd.append(calculated['rsd_percent'])
    total_rsd = [statistics_for([sum(s.values()) for s in samples])['rsd_percent']
                 for samples in profiles.values()]

    boundary = rows(directory / 'rocsolver_boundary/outcomes.csv')
    require(len(boundary) == 6, 'Expected six boundary probes')
    require({(r['routine'], int(r['n'])) for r in boundary}
            == {(k,n) for k in ('getrf','potrf') for n in (46340,46341,49152)}, 'Boundary coverage')
    for row in boundary:
        file = directory / 'rocsolver_boundary' / row['log']
        content = file.read_text()
        code = int(file.with_suffix('.exit').read_text())
        require(int(row['exit_code']) == code and int(row['n_squared']) == int(row['n'])**2,
                'Boundary outcome mismatch')
        fault = 'invalid address' in content.lower()
        require(str(fault) == row['invalid_address_fault'], 'Fault flag mismatch')
        require(';'.join(re.findall(r'Error code:\s*(\d+)', content)) == row['error_codes'], 'Fault code mismatch')
        require((code == 0 and not fault) if int(row['n']) == 46340 else (code != 0 and fault),
                'Unexpected success/failure boundary')
    return {'files_checked': len(entries), 'residual_records': 45, 'benchmark_jobs': 18,
            'stage_cases': len(profiles), 'maximum_stage_rsd_percent': max(stage_rsd),
            'maximum_total_rsd_percent': max(total_rsd), 'boundary_cases': 6,
            'tested_sources_checked': check_sources}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT)
    parser.add_argument('--skip-source-check', action='store_true', help='For later source revisions; data checks remain enabled')
    args = parser.parse_args()
    print(json.dumps(verify(args.data_dir.resolve(), not args.skip_source_check), indent=2))
