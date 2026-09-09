"""Build a scientific PDF and a readable Markdown report from validated outputs."""
import csv
import json
from pathlib import Path
import statistics
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/meituan_gamma_analysis_20260909'
PDF = ROOT / 'output/pdf/meituan_gamma_results_20260909.pdf'
PDF.parent.mkdir(parents=True, exist_ok=True)
A = json.loads((OUT / 'analysis.json').read_text())
OLD, FULL, PILOT = ('meituan_area_gamma_60s_pilot_v1', 'meituan_rbatch2_pb_120s_g6_v1', 'meituan_area5_rbatch2_pb_120s_g6_v1')
with (OUT / 'coverage.csv').open() as f:
    COVERAGE = list(csv.DictReader(f))
with (OUT / 'candidate_results.csv').open() as f:
    CANDIDATES = list(csv.DictReader(f))
NAVY, TEAL, ORANGE = colors.HexColor('#18344c'), colors.HexColor('#227e86'), colors.HexColor('#bc5f39')
GREY, LIGHT = colors.HexColor('#536776'), colors.HexColor('#edf2f5')
WIDTH = 512
styles = {
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=23, leading=27, textColor=NAVY, spaceAfter=12),
    'h1': ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=17, leading=21, textColor=NAVY, spaceAfter=12),
    'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=11.5, leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=6),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=10, leading=14.3, textColor=NAVY, spaceAfter=9),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=8.3, leading=11.4, textColor=GREY, spaceAfter=7),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=9.2, leading=12.2, textColor=NAVY),
    'head': ParagraphStyle('head', fontName='Helvetica-Bold', fontSize=9.1, leading=11.5, textColor=colors.white),
    'callout': ParagraphStyle('callout', fontName='Helvetica-Bold', fontSize=11.5, leading=16, textColor=TEAL, spaceAfter=12),
}
story = []


def p(text, style='body'):
    story.append(Paragraph(text, styles[style]))


def heading(title):
    p(title, 'h1')


def tab(headers, rows, widths):
    cells = [[Paragraph(escape(str(c)), styles['head']) for c in headers]]
    cells += [[Paragraph(str(c), styles['cell']) for c in row] for row in rows]
    obj = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
    obj.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor('#c9d3da')),
    ]))
    story.append(obj)
    story.append(Spacer(1, 9))


def fig(name, width=WIDTH):
    path = OUT / 'figures' / (name + '.png')
    w, h = ImageReader(str(path)).getSize()
    story.append(Image(str(path), width=width, height=width * h / w))
    story.append(Spacer(1, 6))


def page():
    story.append(PageBreak())


def scalar(name, gamma):
    return next(r for r in A[name]['global'] if r['gamma'] == gamma)


def prefix(name, count=3):
    return next(r for r in A[name]['heldout_prefix_summary'] if r['completed_coordinates'] == count)


def area(name, id):
    return [r for r in A[name]['selected'] if r['area'] == id]


def freq(rows):
    vals = sorted({r['selected_gamma'] for r in rows})
    return '; '.join(f"{v:g} ({sum(r['selected_gamma'] == v for r in rows)}/8)" for v in vals)


old0, old3, old5, old1 = [scalar(OLD, g) for g in [0., .3, .5, 1.]]
pilot3, full3 = prefix(PILOT), prefix(FULL)
old_plain_gain = 100 * (old3['savings'] / old0['savings'] - 1)
old_default_gain = 100 * (old3['savings'] / old5['savings'] - 1)
pilot_training = statistics.mean(r['cumulative_training_gain_relative_pct'] for r in area(PILOT, '12'))

p('Meituan area-gamma experiments', 'title')
p('Complete 60s RBAT benchmark and preliminary 120s periodic results<br/>Downloaded snapshot analyzed 9 September 2026', 'small')
p('The global coefficient is stable. The completed area updates show very small gains, and the five-area pilot loses a little on held-out days.', 'callout')
p(f"At 60 seconds, <b>gamma 0.3</b> is best on every day and in every leave-one-day-out global fit. It saves <b>{old3['savings_pct']:.3f}%</b> of total direct distance. At 120 seconds, <b>gamma 0.4</b> is best on every day and fold in the completed six-value global sweep.")
p(f"The five-area pilot has completed areas <b>22, 6 and 12</b> for all eight folds. Their combined held-out change is <b>{pilot3['heldout_gain_relative_pct']:+.4f}% in distance saved</b> relative to gamma 0.4. The full experiment has completed areas <b>0, 1 and 2</b>, with a change of <b>{full3['heldout_gain_relative_pct']:+.4f}%</b>. These are tiny changes in the savings objective, not percentage-point changes in the saving rate.")
tab(['Experiment', 'Complete and usable', 'Still incomplete'], [
    ['60s PB-RBAT', '88/88 scalar runs: 8 days x 11 gammas', 'No area-fitting results in this download'],
    ['120s periodic, full run', '48/48 scalar runs; areas 0, 1, 2: 1,008/1,008 fitting rows', 'Area 3: 112/336 rows; later areas absent'],
    ['120s periodic, five-area pilot', '8/8 reference runs; areas 22, 6, 12: 1,008/1,008 fitting rows', 'Area 5: 193/336 rows; area 9 absent'],
], [120, 218, 174])
p('A useful distinction', 'h2')
p('There are no final area-policy evaluation files yet. However, exact cached simulations cover all eight held-out days for the first one, two and three completed updates in both periodic experiments. This report reconstructs those intermediate evaluations after validating their full coefficient vectors and data identities. No simulations were rerun.')
p('All runs retain all citywide jobs and use PB only. The 60s policy is event-driven RBAT (rbatch); the 120s policy is periodic RBAT (rbatch2), matching every 30 seconds. The saved metadata confirms 120 seconds, rather than 12 seconds. Gamma maps to a full shadow coefficient of gamma / 2.', 'small')

page(); heading('1  |  The complete 60s global benchmark')
p('Eight three-hour lunch snapshots cover 190,875 jobs in 23 area IDs. Each of the 11 gamma values is applied uniformly to all areas. The saving rate below divides aggregate savings by the actual aggregate direct distance; it is not an OPT ratio.')
fig('60s_global_sensitivity')
p('Thin grey curves show the eight individual days; the dark curve uses aggregate savings / aggregate direct distance. Pooling is weighted by jobs, and mean wait by jobs. Both panels use the same complete sample.', 'small')
tab(['Global gamma', 'Distance saved', 'Jobs pooled', 'Mean wait', 'Cross-area pairs'], [
    [f'{r["gamma"]:g}', f'{r["savings_pct"]:.3f}%', f'{r["pooled_pct"]:.2f}%', f'{r["mean_wait_seconds"]:.2f}s', f'{r["cross_area_pairs_pct"]:.2f}%']
    for r in [old0, scalar(OLD, .2), old3, scalar(OLD, .4), old5, old1]
], [82, 110, 104, 102, 114])
p(f"Gamma 0.3 improves distance saved by <b>{old_plain_gain:.3f}%</b> over gamma 0 (plain RBAT), or <b>{old3['savings_pct'] - old0['savings_pct']:.3f} percentage points</b> of direct distance. Relative to gamma 0.5, the gain is <b>{old_default_gain:.3f}%</b> in savings, or <b>{old3['savings_pct'] - old5['savings_pct']:.3f} percentage points</b>.")
p('Larger gamma values pool fewer jobs and eventually reduce savings. Maximizing the pooled fraction would choose a different policy from maximizing distance saved. The substantial cross-area pair share also supports retaining citywide matching during area fitting.')

page(); heading('2  |  Global tuning generalizes across these days')
p('For each held-out day, choose gamma by summing citywide savings on the other seven days, then evaluate that gamma on the omitted day. All eight fits select gamma 0.3. The same-day oracle also picks 0.3 in all eight cases; it is only an oracle diagnostic, not the selection rule.')
fig('60s_heldout_global')
tab(['Held-out date', 'Fitted gamma', 'Distance saved', 'Gain vs gamma 0', 'Gain vs gamma 0.5'], [
    ['Oct ' + str(17 + int(r['day'])), f"{r['historical_global_gamma']:g}", f"{r['heldout_savings_pct']:.3f}%",
     f"{100 * r['gain_vs_plain'] / next(float(c['savings']) for c in CANDIDATES if c['experiment'] == OLD and c['day'] == r['day'] and float(c['candidate_gamma']) == 0):+.3f}%",
     f"{100 * r['gain_vs_fixed_reference'] / next(float(c['savings']) for c in CANDIDATES if c['experiment'] == OLD and c['day'] == r['day'] and float(c['candidate_gamma']) == .5):+.3f}%"]
    for r in A[OLD]['heldout_global']
], [102, 83, 108, 108, 111])
p('What this tells us about area fitting', 'h2')
p('There is a stable global starting point and a modest benefit to tuning it. Area fitting must improve on that tuned baseline. A different selected gamma in an area is not itself evidence of better performance; the citywide held-out savings are the relevant test.')
p('The planned 60s area-specific curves, gamma-versus-density plot and held-out area-policy comparison cannot be produced from uniform-gamma rows. Their missing status is explicit here. The corresponding preliminary area analysis is available for 120s on the following pages.', 'small')

page(); heading('3  |  What the completed 120s area sweeps show')
p('The five-area pilot starts at gamma 0.4 everywhere, then updates areas 22, 6, 12, 5 and 9 in that order. At each update, all six gamma values are tested on the same seven training days while the other 22 area coefficients remain fixed. Earlier selected updates stay in place.')
fig('120s_pilot_conditional_curves')
p('Each thin curve is one seven-day training fold; the dark curve is their descriptive mean. The vertical scale is the relative change in total citywide savings from the current coefficient. Panels have different vertical scales. Curves are conditional on preceding updates, not independent isolated-area experiments.', 'small')
tab(['Pilot area', 'Selected gamma (folds)', 'Mean training gain', 'Interpretation'], [
    ['22', freq(area(PILOT, '22')), f"{statistics.mean(r['training_gain_relative_pct'] for r in area(PILOT, '22')):+.4f}%", 'Sparse area; lower gamma wins, with tiny citywide effect'],
    ['6', freq(area(PILOT, '6')), '+0.0000%', 'Retains the global reference in every fold'],
    ['12', freq(area(PILOT, '12')), f"{statistics.mean(r['training_gain_relative_pct'] for r in area(PILOT, '12')):+.4f}%", 'Choice varies across folds; very small advantage'],
], [60, 140, 100, 212])
p(f"After these three updates, mean cumulative training improvement is <b>{pilot_training:.4f}%</b> of savings across folds (range 0.0032% to 0.0070%). All maximizers are unique under the saved 1e-6 absolute near-optimal tolerance; that numerical uniqueness does not make the gains practically large.")
p('The separate full experiment', 'h2')
tab(['Completed area', 'Selected gamma', 'Mean training gain'], [
    ['0', '0.6 in 8/8 folds', '+0.0124%'], ['1', '0.4 in 8/8 folds', '+0.0000%'], ['2', '0.4 in 8/8 folds', '+0.0000%'],
], [150, 190, 172])
p('The full run starts from historical global selection: gamma 0.4 in every fold. Its different area order makes the later curves conditional on a different coefficient vector.', 'small')

page(); heading('4  |  Preliminary held-out area-policy performance')
p('The first three completed updates can already be evaluated on all eight held-out days using exact saved simulations. The comparator is uniform gamma 0.4 on the same day. That is also the historically tuned global gamma in every fold of the full experiment.')
fig('120s_heldout_completed_prefix')
tab(['Completed policy prefix', 'Days', 'Change in savings', 'Saving-rate change', 'Daily signs + / - / ='], [
    ['Pilot: area 22', '8/8', f"{prefix(PILOT, 1)['heldout_gain_relative_pct']:+.4f}%", f"{prefix(PILOT, 1)['heldout_gain_savings_pp']:+.5f} pp", '4 / 0 / 4'],
    ['Pilot: areas 22, 6', '8/8', f"{prefix(PILOT, 2)['heldout_gain_relative_pct']:+.4f}%", f"{prefix(PILOT, 2)['heldout_gain_savings_pp']:+.5f} pp", '4 / 0 / 4'],
    ['Pilot: areas 22, 6, 12', '8/8', f"{pilot3['heldout_gain_relative_pct']:+.4f}%", f"{pilot3['heldout_gain_savings_pp']:+.5f} pp", '3 / 3 / 2'],
    ['Full: areas 0, 1, 2', '8/8', f"{full3['heldout_gain_relative_pct']:+.4f}%", f"{full3['heldout_gain_savings_pp']:+.5f} pp", '5 / 3 / 0'],
], [160, 43, 101, 104, 104])
p('The pilot is slightly worse after area 12', 'h2')
p('Area 12 chooses gamma 0.2 when holding out October 17, 18 and 24. On each of those three held-out days, that change loses savings. In the other five folds it retains gamma 0.4. This is a concrete example of why a training improvement needs a held-out check.')
p('Why the cache reconstruction is valid', 'h2')
p('Each chosen vector comes from a verified selection that excludes the target day. A cached simulation is reused only when the entire vector, job-file hash, day, code, dataset, window, period, policy, seed and numerical runtime match. The cache payload and supporting candidate result are checked. A simulation produced while serving another fold is still the same day/vector policy evaluation; its outcome is never used to choose the held-out fold\'s vector.')
p('These are intermediate policies after a completed prefix, not the final five-area or 23-area fitted policies. The eight training sets overlap, and there is only one observed week. The figures are descriptive paired results, with no independence-based confidence interval or significance claim.', 'small')

page(); heading('5  |  Gamma, arrivals and pooling opportunities')
p('Each point represents one completed pilot area in one held-out fold. Descriptors use the other seven days only. Points may overlap; they are not 24 independent area observations.')
fig('120s_pilot_gamma_descriptors')
tab(['Area', 'Historical jobs: range', 'Nominal density: range', '60s opportunities: range', 'Gamma choices'], [
    [id,
     f"{min(r['historical_arrivals'] for r in area(PILOT, id)):,}-{max(r['historical_arrivals'] for r in area(PILOT, id)):,}",
     f"{min(r['nominal_density'] for r in area(PILOT, id)):.2f}-{max(r['nominal_density'] for r in area(PILOT, id)):.2f}",
     f"{min(r['mean_poolable_all'] for r in area(PILOT, id)):.2f}-{max(r['mean_poolable_all'] for r in area(PILOT, id)):.2f}",
     ', '.join(f'{g:g}' for g in sorted({r['selected_gamma'] for r in area(PILOT, id)}))]
    for id in ['22', '6', '12']
], [42, 127, 123, 127, 93])
p('How to read these quantities', 'h2')
p('Nominal density is historical arrivals / historical exposure seconds x 120: expected same-area arrivals during the waiting window, rather than spatial density or realized active jobs.')
p('Opportunities count positive-reward future partners from <b>all</b> areas. Existing diagnostics use <b>60 seconds</b> and exclude truncated windows. They provide structural context here; a 120-second opportunity count would need recomputation.')
p('What is supported so far', 'h2')
p('Sparse area 22 prefers a lower gamma than area 6. Area 12 has more all-area opportunities despite fewer arrivals, and its coefficient is less stable. Three areas do not establish a density law or a monotone relationship.')
p('A small citywide effect could conceal a larger local effect, but these aggregate results cannot attribute savings to an area or job. That would require match-level output.')
p('Complete areas 5 and 9 before extending to more windows. Retain gamma 0.4 as the reference and assess paired held-out gains after the five-area pass.', 'callout')

page(); heading('6  |  Completeness, validation and next outputs')
p('Every completed coordinate requires 6 gammas x 7 training days = 42 rows per fold, or 336 across all eight folds. Partial fourth-coordinate totals are unbalanced and are not used to choose or compare gammas.')
tab(['Held-out day', 'Pilot area 5: rows / 42', 'Full-run area 3: rows / 42'], [
    ['Oct ' + str(17 + fold),
     next(r['found'] for r in COVERAGE if r['experiment'] == PILOT and r['fold'] == str(fold) and r['step'] == '3') + ' / 42',
     next(r['found'] for r in COVERAGE if r['experiment'] == FULL and r['fold'] == str(fold) and r['step'] == '3') + ' / 42']
    for fold in range(8)
], [112, 200, 200])
p('What remains in the downloads', 'h2')
p('The five-area pilot is missing <b>143</b> area-5 candidate rows and all <b>336</b> area-9 rows, followed by the final held-out area-policy outputs. The full run is missing <b>224</b> area-3 rows and all later-area rows. Download completeness is not a live GRID queue status; no jobs were submitted, cancelled or rerun for this report.')
p('Validation and aggregation', 'h2')
p('All <b>2,465</b> available candidate JSON results passed validation against their own archived source and manifest. Checks cover input and diagnostic file hashes, result and cache content hashes, job accounting, direct distances recalculated from all input jobs, normalized metrics, numerical runtime, coefficient vectors, all 48 saved coordinate selections and their historical dependency chains. The two periodic gamma-0.4 baselines reproduce identical savings, pair counts, waits and decision hashes on every day.')
p('Aggregate saving rate = 100 x sum(savings) / sum(direct distances). Relative improvement = 100 x (new savings - reference savings) / reference savings. A percentage-point change uses direct distance as denominator. Distances use the simulator\'s Euclidean coordinate units, not kilometers or road-network distance. LP and OPT were not computed. Leave-one-day-out evaluation tests cross-day transfer; it is not a chronological future-day split.', 'small')
p('Reproducibility', 'h2')
p('The analysis folder contains the immutable source inventory with SHA256 hashes, validation audit, candidate table, complete conditional curves, selected coefficients, missing-result paths and reconstructed held-out cache references. The report builder reads those validated tables. Frozen experiment source and original downloads were not changed.', 'small')
p('Result/manifest IDs: meituan_area_gamma_60s_pilot_v1; meituan_rbatch2_pb_120s_g6_v1; meituan_area5_rbatch2_pb_120s_g6_v1. Sources include their upload archives and data/meituan_area_gamma_v1/dataset.json. Design: docs/hexagon_gamma_plan.md. Paper context: Dynamic_Delivery_Pooling.pdf, Figure 21 and Appendix E.3; its gamma-0.4 reference agrees with this 120s sweep, without claiming replication of its exact sample.', 'small')


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#d5dee4'))
    canvas.line(50, 40, 562, 40)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(GREY)
    canvas.drawString(50, 27, 'Meituan gamma analysis | saved snapshot, 9 September 2026')
    canvas.drawRightString(562, 27, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(str(PDF), pagesize=(612, 792), leftMargin=50, rightMargin=50,
                        topMargin=43, bottomMargin=52, title='Meituan gamma experiments: complete 60s and preliminary 120s results',
                        author='Research analysis', pageCompression=1)
doc.build(story, onFirstPage=footer, onLaterPages=footer)

md = f'''# Meituan gamma analysis — downloaded snapshot, 9 September 2026

The complete 60s global benchmark is stable: **gamma 0.3 wins on every day and in every leave-one-day-out fit**. The 120s global sweep similarly selects **gamma 0.4** throughout. Area fitting has produced tiny effects so far: the first three updates in the five-area pilot change held-out distance saved by **{pilot3['heldout_gain_relative_pct']:+.4f}%**, while the first three areas in the full experiment change it by **{full3['heldout_gain_relative_pct']:+.4f}%**.

These percentages are relative changes in the savings objective. The corresponding changes in savings / direct distance are {pilot3['heldout_gain_savings_pp']:+.5f} and {full3['heldout_gain_savings_pp']:+.5f} percentage points.

## What was tested and what is complete

- All runs use PB only, with citywide matching over all 190,875 jobs from eight lunch snapshots, October 17–24, 2022. Area labels are job-level `da_id`; no isolated-area simulation is used.
- The complete 60s run uses event-driven RBAT (`rbatch`): 8 days x 11 global gammas = **88/88 runs**. There are no 60s area-fitting outputs in the downloaded folder.
- The new runs use **120s**, periodic RBAT (`rbatch2`) with a **30s matching period**, and gamma 0, 0.2, 0.4, 0.6, 0.8, 1. The full shadow coefficient is gamma / 2.
- Full experiment: **48/48** global runs; areas **0, 1, 2** complete for all folds; area 3 has **112/336** rows.
- Five-area pilot: **8/8** gamma-0.4 reference runs; areas **22, 6, 12** complete for all folds; area 5 has **193/336** rows; area 9 is absent.

No final area-evaluation files are present. Nevertheless, exact cached day/vector simulations permit valid held-out evaluation of each completed one-, two- and three-coordinate prefix on **all eight days**. No new simulation was run.

## Complete 60s report

| Gamma | Saved / direct distance | Jobs pooled | Mean wait | Cross-area share of pairs |
|---|---:|---:|---:|---:|
'''
for g in [0., .2, .3, .4, .5, 1.]:
    r = scalar(OLD, g)
    md += f"| {g:g} | {r['savings_pct']:.3f}% | {r['pooled_pct']:.2f}% | {r['mean_wait_seconds']:.2f}s | {r['cross_area_pairs_pct']:.2f}% |\n"
md += f'''
Gamma 0.3 improves saved distance by **{old_plain_gain:.3f}% versus plain RBAT**, and **{old_default_gain:.3f}% versus gamma 0.5**. These equal {old3['savings_pct'] - old0['savings_pct']:.3f} and {old3['savings_pct'] - old5['savings_pct']:.3f} percentage points of direct distance. All eight held-out days improve. The same-day oracle also selects 0.3 on every day, but is reported only as an oracle diagnostic.

Higher gamma eventually reduces both pooling and savings. More pooled jobs is not equivalent to greater distance saved. The plotted near-optimal global region is approximately 0.2–0.4; this is descriptive, not a formal equivalence range.

![60s sensitivity]({OUT / 'figures/60s_global_sensitivity.png'})

![60s held-out global comparison]({OUT / 'figures/60s_heldout_global.png'})

The planned 60s area-specific curves, gamma-versus-density analysis and held-out area-policy comparisons cannot be inferred from these uniform-gamma runs. The 120s results provide the first area evidence.

## Preliminary 120s area fits

All conditional candidates use the same seven training days and a complete citywide instance. Select one area's gamma using total historical citywide savings, retain earlier updates, then move to the next area. The pilot starts at fixed gamma 0.4; the full experiment starts from historical global selection (also 0.4 in every fold). Their area order differs.

| Experiment / area | Selected gamma across eight folds | Mean incremental training gain in savings |
|---|---|---:|
'''
for name, ids, label in [(PILOT, ['22', '6', '12'], 'Pilot'), (FULL, ['0', '1', '2'], 'Full')]:
    for id in ids:
        rows = area(name, id)
        md += f"| {label} / {id} | {freq(rows)} | {statistics.mean(r['training_gain_relative_pct'] for r in rows):+.4f}% |\n"
md += f'''
The pilot's mean cumulative training gain after areas 22, 6 and 12 is **{pilot_training:.4f}%** of savings. Unique numerical maximizers under the stored 1e-6 absolute tolerance should not be mistaken for economically large differences.

![Pilot conditional curves]({OUT / 'figures/120s_pilot_conditional_curves.png'})

![Full-run conditional curves]({OUT / 'figures/120s_full_conditional_curves.png'})

## Held-out results reconstructed from exact cached simulations

| Completed policy | Days | Relative change in savings | Saving-rate change | Positive / negative / unchanged days |
|---|---:|---:|---:|---|
'''
for name, step, label in [(PILOT, 1, 'Pilot: 22'), (PILOT, 2, 'Pilot: 22, 6'), (PILOT, 3, 'Pilot: 22, 6, 12'), (FULL, 3, 'Full: 0, 1, 2')]:
    r = prefix(name, step)
    md += f"| {label} | {r['days']}/8 | {r['heldout_gain_relative_pct']:+.4f}% | {r['heldout_gain_savings_pp']:+.5f} pp | {r['positive_days']} / {r['negative_days']} / {r['unchanged_days']} |\n"
md += f'''
**Area 12 illustrates the value of held-out evaluation.** It selects gamma 0.2 in folds holding out October 17, 18 and 24. That change loses savings on each of those held-out days. It retains 0.4 in the other five folds.

![Held-out changes after three updates]({OUT / 'figures/120s_heldout_completed_prefix.png'})

Reconstruction is valid because selection excludes the target day and the cache is keyed by the exact full coefficient vector, data, day, job-file hash, code, policy, window, matching period, seed and numerical runtime. Cache payloads and supporting candidate results were validated. A simulation cached while another fold is training remains the same day/vector evaluation; its outcome is not used to fit this held-out fold's vector. The supporting selection IDs and cache paths are retained in the reconstructed held-out CSVs.

These are intermediate policies after completed prefixes. They are not the final five-area or 23-area fitted policies. The eight training sets overlap; the results are descriptive paired comparisons over one observed week, without independence-based confidence intervals or significance claims.

## Gamma versus historical descriptors

![Pilot gamma descriptors]({OUT / 'figures/120s_pilot_gamma_descriptors.png'})

All descriptors exclude the target day. Nominal density is historical arrivals / exposure seconds x 120, not spatial density or realized active jobs. The opportunity count considers positive-reward future partners from all areas, but the existing diagnostic uses **60 seconds**, not 120 seconds. It is a structural descriptor here; a window-matched opportunity analysis requires recomputation.

Area 22 is sparse and prefers a lower gamma. Area 12 has more cross-area-inclusive opportunities than area 6 despite fewer arrivals, and its gamma is less stable. With only three distinct pilot areas and overlapping folds, this does not establish a density law or monotone coefficient rule. The aggregate outcomes do not attribute savings to individual areas or jobs.

## Missing outputs and next step

The pilot still needs 143 area-5 candidate rows, all 336 area-9 candidate rows, their selections and the final held-out area-policy outputs. The full run is missing 224 area-3 rows and all later-area rows. No incomplete fourth-coordinate curve is used for a gamma comparison because candidate-day totals are unbalanced. Local download completeness does not establish live GRID job status.

**Complete the five-area pilot before extending to more waiting windows.** Keep gamma 0.4 as the reference and assess paired held-out gains after the full pass. So far, coefficient heterogeneity has not delivered a material observed benefit.

## Validation and definitions

All 2,465 available candidate JSON files passed archived-source, manifest, dataset, result-content, accounting, metric, runtime and cache checks. All 48 saved coordinate selections were reconstructed and their dependency chains validated. Direct distances were recomputed from every input job. The two periodic gamma-0.4 baseline sets have identical savings, pair counts, waits and decision hashes for all eight days.

- Saving rate: 100 x sum(savings) / sum(direct distance).
- Relative improvement: 100 x (new savings - reference savings) / reference savings.
- Saving-rate percentage-point change: 100 x savings difference / direct distance.
- Pooled fraction and mean wait are weighted by jobs; cross-area share uses selected pairs.
- Distance is in simulator Euclidean coordinate units, not kilometers or road-network distance. OPT/LP ratios are unavailable because those benchmarks were disabled.
- Leave-one-day-out is a cross-day transfer check, not a chronological future-day split.
- Comparing the 60s and 120s levels changes both waiting window and dispatch policy; this is not an isolated estimate of either effect.
- Cached runtime must be counted once per unique simulation; summing repeated candidate `time_s` values overstates cost. Saved unique simulation cost is about {A[PILOT]['audit']['unique_cached_simulation_hours']:.2f} hours in the pilot and {A[FULL]['audit']['unique_cached_simulation_hours']:.2f} hours in the full run, excluding preparation and queue/wait overhead. Those are cumulative simulation times, not wall-clock completion estimates.

## Sources and reproducibility

The report uses the three result folders named above, their matching upload archives, experiment manifests in `configs/`, and `data/meituan_area_gamma_v1/`. Paper context: *Dynamic Delivery Pooling*, Figure 21 and Appendix E.3; the 120s gamma-0.4 reference is consistent with the current global sweep, without claiming replication of its exact sample. Planned outputs are documented in `docs/hexagon_gamma_plan.md`.

In this analysis folder: `source_inventory.json`, `validation.json`, `candidate_results.csv`, `complete_conditional_curves.csv`, `selected_areas.csv`, `coverage.csv`, `missing_coordinate_results.csv` and each experiment's `_reconstructed_heldout_prefix.csv` preserve the audit and numerical inputs. Scientific figures have PNG and editable SVG versions. The source downloads and frozen experiment code were not modified.

Reproduce analysis with the bundled Python runtime and `scripts/analyze_meituan_gamma_downloads.py --source-root <research-checkout>`, then generate figures with `scripts/plot_meituan_gamma_analysis.py` (Matplotlib) and the PDF with `scripts/report_meituan_gamma_analysis.py` (ReportLab).
'''
(OUT / 'report.md').write_text(md)
print('PDF:', PDF)
print('Markdown:', OUT / 'report.md')
