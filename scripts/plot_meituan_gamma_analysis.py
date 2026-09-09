"""Standalone scientific figures from validated saved-result analysis tables."""
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ddp-gamma-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/meituan_gamma_analysis_20260909'
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
A = json.loads((OUT / 'analysis.json').read_text())
OLD = 'meituan_area_gamma_60s_pilot_v1'
FULL = 'meituan_rbatch2_pb_120s_g6_v1'
PILOT = 'meituan_area5_rbatch2_pb_120s_g6_v1'
with (OUT / 'candidate_results.csv').open() as f:
    CANDIDATES = list(csv.DictReader(f))
with (OUT / 'complete_conditional_curves.csv').open() as f:
    CURVES = list(csv.DictReader(f))
NAVY, TEAL, ORANGE, GREY = '#18344c', '#227e86', '#bc5f39', '#9ba7b1'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': NAVY, 'text.color': NAVY, 'axes.titleweight': 'bold',
                     'axes.edgecolor': '#8c969e', 'grid.color': '#dce2e6', 'grid.linewidth': .6,
                     'legend.frameon': False, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})


def finish(fig, name):
    fig.savefig(FIG / (name + '.png'), dpi=210, bbox_inches='tight', pad_inches=.12)
    fig.savefig(FIG / (name + '.svg'), bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


def scalar():
    x = A[OLD]['global']
    gamma = [r['gamma'] for r in x]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.0), constrained_layout=True)
    for day in map(str, range(8)):
        rows = sorted([r for r in CANDIDATES if r['experiment'] == OLD and r['day'] == day], key=lambda r: float(r['candidate_gamma']))
        axes[0].plot(gamma, [100 * float(r['savings_fraction']) for r in rows], color=GREY, alpha=.5, lw=.85)
    axes[0].plot(gamma, [r['savings_pct'] for r in x], color=NAVY, marker='o', ms=3, lw=2, label='All days, weighted')
    axes[0].axvline(.3, color=TEAL, ls=':', lw=1)
    axes[0].set(title='Distance saved', ylabel='Saved / total direct distance (%)', xlabel=r'$\gamma$')
    axes[0].legend(loc='lower left', fontsize=8)
    axes[0].grid(axis='y')
    axes[1].plot(gamma, [r['pooled_pct'] for r in x], color=TEAL, marker='o', ms=3, label='Jobs pooled')
    axes[1].set(title='Pooling and waiting', ylabel='Jobs pooled (%)', xlabel=r'$\gamma$')
    twin = axes[1].twinx()
    twin.spines['right'].set_visible(True)
    twin.plot(gamma, [r['mean_wait_seconds'] for r in x], color=ORANGE, marker='s', ms=3, label='Mean wait')
    twin.set_ylabel('Mean wait (seconds)', color=ORANGE)
    twin.tick_params(axis='y', labelcolor=ORANGE)
    axes[1].tick_params(axis='y', labelcolor=TEAL)
    axes[1].legend(loc='lower left', fontsize=8)
    twin.legend(loc='upper right', fontsize=8)
    for ax in axes:
        ax.set_xticks([0, .2, .4, .6, .8, 1])
    finish(fig, '60s_global_sensitivity')
    fig, ax = plt.subplots(figsize=(7.25, 2.75), constrained_layout=True)
    gamma3 = [next(r for r in CANDIDATES if r['experiment'] == OLD and r['day'] == str(d) and float(r['candidate_gamma']) == .3) for d in range(8)]
    xs = np.arange(8)
    for shift, ref_gamma, color in [(-.18, 0, TEAL), (.18, .5, ORANGE)]:
        refs = [next(r for r in CANDIDATES if r['experiment'] == OLD and r['day'] == str(d) and float(r['candidate_gamma']) == ref_gamma) for d in range(8)]
        gains = [100 * (float(r['savings']) - float(b['savings'])) / float(b['savings']) for r, b in zip(gamma3, refs)]
        ax.bar(xs + shift, gains, .34, color=color, label=r'Compared with $\gamma=' + str(ref_gamma) + '$')
    ax.set_xticks(xs, ['Oct ' + str(17 + d) for d in range(8)])
    ax.set_ylabel('Increase in distance saved (%)')
    ax.set_title('60s: held-out global selection improves every day')
    ax.set_ylim(0, 2.5)
    ax.legend(ncol=2, loc='upper right', fontsize=8)
    ax.grid(axis='y'); ax.set_axisbelow(True)
    finish(fig, '60s_heldout_global')


def conditional(name, areas, filename, color):
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.8), constrained_layout=True)
    for ax, area in zip(axes, areas):
        ys = []
        for fold in map(str, range(8)):
            rows = sorted([r for r in CURVES if r['experiment'] == name and r['area'] == area and r['fold'] == fold], key=lambda r: float(r['gamma']))
            x = [float(r['gamma']) for r in rows]
            y = [float(r['gain_vs_current_relative_pct']) for r in rows]
            ys.append(y)
            ax.plot(x, y, color=color, alpha=.32, lw=.8)
        ax.plot(x, np.mean(ys, axis=0), color=NAVY, lw=1.6, marker='o', ms=3)
        ax.axhline(0, color=GREY, ls=':', lw=.9)
        ax.axvline(.4, color=GREY, ls=':', lw=.9)
        ax.set(title='Area ' + area, xlabel=r'$\gamma$')
        ax.set_xticks([0, .2, .4, .6, .8, 1])
        ax.grid(axis='y')
    axes[0].set_ylabel('Training change in distance saved (%)')
    finish(fig, filename)


def heldout():
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.0), constrained_layout=True, sharey=True)
    for ax, name, title in zip(axes, [PILOT, FULL], ['Five-area pilot: 22, 6, 12', 'Full run: 0, 1, 2']):
        rows = sorted([r for r in A[name]['reconstructed_heldout_prefix'] if r['completed_coordinates'] == 3], key=lambda r: int(r['fold']))
        y = [r['heldout_gain_relative_pct'] for r in rows]
        ax.bar(range(8), y, color=[TEAL if v >= 0 else ORANGE for v in y], width=.72)
        ax.axhline(0, color=NAVY, lw=.7)
        ax.set_xticks(range(8), list(map(str, range(17, 25))))
        ax.set(title=title, xlabel='October 2022: held-out date')
        ax.grid(axis='y'); ax.set_axisbelow(True)
    axes[0].set_ylabel('Change in held-out distance saved (%)')
    finish(fig, '120s_heldout_completed_prefix')


def descriptors():
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65), constrained_layout=True, sharey=True)
    fields = [('historical_arrivals', 'Arrivals on seven training days', True),
              ('nominal_density', r'Nominal density $\lambda\,120$', True),
              ('mean_poolable_all', 'Poolable future jobs (60s)', False)]
    for ax, (field, label, log) in zip(axes, fields):
        for area, color, marker in [('22', ORANGE, 's'), ('6', TEAL, 'o'), ('12', NAVY, '^')]:
            rows = [r for r in A[PILOT]['selected'] if r['area'] == area]
            x = [r[field] for r in rows]
            y = [r['selected_gamma'] for r in rows]
            ax.scatter(x, y, c=color, marker=marker, s=26, alpha=.7, label='Area ' + area)
        if log:
            ax.set_xscale('log')
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylim(-.055, .52)
        ax.set_yticks([0, .2, .4])
        ax.grid(axis='y')
    axes[0].set_ylabel(r'Selected $\gamma$')
    axes[1].legend(ncol=1, fontsize=8, loc='lower right')
    finish(fig, '120s_pilot_gamma_descriptors')


scalar()
conditional(PILOT, ['22', '6', '12'], '120s_pilot_conditional_curves', TEAL)
conditional(FULL, ['0', '1', '2'], '120s_full_conditional_curves', ORANGE)
heldout()
descriptors()
print('Wrote six PNG + SVG scientific figures to', FIG)
