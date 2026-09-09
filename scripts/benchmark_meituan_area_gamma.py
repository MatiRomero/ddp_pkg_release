"""Reproduce one full citywide timing probe before launching historical fitting.

The area probe changes area 5 from a fixed 0.5 initialization to 0.1. It measures
cost only and is explicitly not a fitted table or a held-out performance result.
"""
import argparse
import json
from pathlib import Path
import platform
import resource
import time

from ddp.area_gamma import atomic_json, file_identity, identity, source_identity
from ddp.pb_experiment import PreparedPB
from ddp.scripts.csv_loader import load_jobs_from_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', type=Path, default=Path('data/meituan_area_gamma_v1/dataset.json'))
    parser.add_argument('--day', default='0')
    parser.add_argument('--mode', choices=['scalar', 'area'], default='scalar')
    parser.add_argument('--dispatch', choices=['rbatch', 'rbatch2'], default='rbatch')
    parser.add_argument('--window', type=float, default=60)
    parser.add_argument('--tau-s', type=float, default=30)
    parser.add_argument('--gamma', type=float, default=.5)
    parser.add_argument('--area', default='5')
    parser.add_argument('--candidate-gamma', type=float, default=.1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new benchmark output path')
    data = json.loads(args.dataset_manifest.read_text())
    meta = data['days'][args.day]
    path = args.dataset_manifest.parent / meta['path']
    if file_identity(path) != meta['sha256']:
        raise ValueError('Job file identity mismatch')
    start = time.perf_counter()
    code_id = source_identity()
    jobs = load_jobs_from_csv(path)
    prepared = PreparedPB(jobs, args.window, dispatch=args.dispatch, tau_s=args.tau_s)
    benchmark = {'purpose': 'unfitted runtime benchmark', 'mode': args.mode,
                 'initial_gamma': args.gamma, 'area': args.area, 'candidate_gamma': args.candidate_gamma}
    table = None
    if args.mode == 'scalar':
        row, _ = prepared.run(gamma=args.gamma)
    else:
        target = next(day for day in data['days'] if day != args.day)
        if args.area not in data['areas']:
            parser.error('Unknown area')
        vector = dict.fromkeys(data['areas'], args.gamma); vector[args.area] = args.candidate_gamma
        table = {'schema_version': 1, 'grouping': 'da_id', 'matching_scope': 'citywide',
                 'shadow': 'pb', 'dispatch': args.dispatch, 'tau': 0, 'window_seconds': args.window,
                 **({'tau_s': args.tau_s} if args.dispatch == 'rbatch2' else {}),
                 'target_day': target, 'historical_days': [day for day in data['days'] if day != target],
                 'dataset_days': list(data['days']), 'dataset_id': data['dataset_id'],
                 'training_manifest_id': identity(benchmark), 'coefficients': vector,
                 'fallback_gamma': args.gamma, 'missing_area_policy': 'historical_global', 'benchmark': benchmark}
        row, _ = prepared.run(gamma_table=table)
    if source_identity() != code_id:
        raise RuntimeError('Source changed during benchmark; choose a new output and rerun')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    row.update(benchmark=benchmark, table=table, code_id=code_id, jobs_sha256=meta['sha256'],
               dataset_id=data['dataset_id'], preparation_s=prepared.preparation_s,
               wall_s=time.perf_counter() - start,
               process_peak_rss_mb=rss / (1024 * 1024 if platform.system() == 'Darwin' else 1024))
    atomic_json(args.output, row)
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
