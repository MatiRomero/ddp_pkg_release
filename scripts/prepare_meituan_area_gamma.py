"""Recover labels with the validated join and prepare portable GRID inputs.

Reads the original research checkout; never writes into it. Reuses the completed
opportunity aggregates without repeating the opportunity calculation.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import os
import tempfile

import pandas as pd

from analyze_meituan_zones import load_labeled_snapshots, KEYS
from ddp.area_gamma import atomic_json, file_identity, identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('data/meituan_area_gamma_v1'))
    a = p.parse_args()
    output = a.output.resolve()
    if output == a.reference_root.resolve() or a.reference_root.resolve() in output.parents:
        p.error('Output must be separate from the original research checkout')
    final_output = output
    if output.exists():
        saved = json.loads((output / 'dataset.json').read_text())
        for day, meta in saved['days'].items():
            source = a.reference_root / 'data' / f'meituan_city_lunchtime_plat10301330_day{day}.csv'
            if file_identity(source) != meta['source_sha256'] or file_identity(output / meta['path']) != meta['sha256']:
                p.error('Existing dataset differs from source; use a new output directory')
        if saved['raw_source_sha256'] != file_identity(a.raw):
            p.error('Raw source changed; use a new output directory')
        for name, digest in saved['diagnostic_files'].items():
            original = a.reference_root / 'results/area_density_diagnostic' / ('provenance.json' if name == 'opportunities_provenance.json' else name)
            if file_identity(output / name) != digest or file_identity(original) != digest:
                p.error('Historical diagnostics changed; use a new output directory')
        print(f'Reusing verified dataset at {output}')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=output.name + '.preparing-', dir=output.parent))
    frame, audit, raw_rows, accepted_rows = load_labeled_snapshots(a.reference_root / 'data', a.raw)
    sources = {str(day): file_identity(a.reference_root / 'data' / f'meituan_city_lunchtime_plat10301330_day{day}.csv') for day in range(8)}
    dataset_id = identity({'snapshots': sources, 'areas_in_source_order': frame.da_id.tolist(), 'join_version': 1})
    output.mkdir(parents=True, exist_ok=True)
    days = {}
    for day, part in frame.groupby('day', sort=True):
        # Retain the snapshot's exact coordinate text and order; use only labels from the join.
        source = a.reference_root / 'data' / f'meituan_city_lunchtime_plat10301330_day{day}.csv'
        records = list(csv.DictReader(source.open()))
        if len(records) != len(part):
            raise ValueError('Recovered labels do not align to snapshot rows')
        seen = {}
        for row, area in zip(records, part.da_id):
            key = identity({key: row[key] for key in KEYS})
            occurrence = seen.get(key, 0)
            seen[key] = occurrence + 1
            row.update(da_id=str(area), day=str(day), dataset_id=dataset_id,
                       job_id=f'{day}:{key}:{occurrence}')
        path = output / f'day{day}.csv'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        days[str(day)] = {'path': path.name, 'sha256': file_identity(path), 'jobs': len(records),
                          'exposure_seconds': 10800, 'observation_start': '10:30:00',
                          'observation_end': '13:30:00', 'date': str(part.dt.iloc[0]),
                          'source_sha256': sources[str(day)]}
    diagnostic = a.reference_root / 'results/area_density_diagnostic'
    diagnostic_provenance = json.loads((diagnostic / 'provenance.json').read_text())
    if diagnostic_provenance['window_seconds'] != 60:
        raise ValueError('Expected the completed 60-second diagnostics')
    for name in ['area_daily_arrivals.csv', 'area_day_summary.csv', 'provenance.json']:
        shutil.copy2(diagnostic / name, output / ('opportunities_' + name if name == 'provenance.json' else name))
    # Validate reuse against this exact recovered population.
    saved_counts = pd.read_csv(output / 'area_daily_arrivals.csv', index_col=0)
    counts = pd.crosstab(frame.da_id, frame.day)
    if not (saved_counts.to_numpy() == counts.to_numpy()).all():
        raise ValueError('Diagnostic counts disagree with recovered snapshot population')
    manifest = {'schema_version': 1, 'dataset_id': dataset_id, 'days': days,
                'areas': [str(v) for v in sorted(frame.da_id.unique())],
                'raw_source_sha256': file_identity(a.raw), 'raw_rows': raw_rows,
                'accepted_raw_rows': accepted_rows, 'join': 'accepted records, microdegree OD + dt + both timestamps',
                'join_audit': audit.to_dict(orient='records'),
                'diagnostic_files': {name: file_identity(output / name) for name in ['area_daily_arrivals.csv', 'area_day_summary.csv', 'opportunities_provenance.json']}}
    atomic_json(output / 'dataset.json', manifest)
    os.replace(output, final_output)
    print(f'Prepared {len(frame):,} jobs, {len(days)} days, {len(manifest["areas"])} areas at {final_output}')


if __name__ == '__main__':
    main()
