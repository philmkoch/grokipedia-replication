"""Party test on the second snapshot of the US two-party politicians, November 2026.

    python rescrape.py DATA_DIR FEATURES_DIR NOV_DATA_DIR NOV_FEATURES_DIR OUT_DIR

DATA_DIR and FEATURES_DIR are the February corpus and its features, as for
results.py. NOV_DATA_DIR holds the November pairs_clean.parquet, NOV_FEATURES_DIR the
features.py output on it. The politicians of the February corpus form the
panel, the others are new. Writes OUT_DIR/results.json and the report's figure
rescrape_party as PDF.
"""
# %% Setup
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SIDES = ['wp', 'gp']
FEB_SCRAPE = pd.Timestamp('2026-02-13', tz='UTC')  # the February GP files date from Feb 11 and 12
TEXTWIDTH = 418.26 / 72.27  # text width of the report in inches, so figure text prints at 9 pt
WP, GP = '#004488', '#DDAA33'  # Tol high-contrast blue and yellow, apart in greyscale too
DIFF, REF = '#777777', 'black'  # GP vs WP quantities, reference lines


def add_lean(frame):
    """Lean is P(supports conservatives) minus P(supports liberals), shift is GP lean minus WP lean."""
    for side in SIDES:
        frame[f'{side}_lean'] = frame[f'{side}_stance_con'] - frame[f'{side}_stance_lib']
    frame['lean_shift'] = frame.gp_lean - frame.wp_lean
    return frame


data_dir, features_dir, nov_data_dir, nov_features_dir, out_dir = map(Path, sys.argv[1:6])
out_dir.mkdir(parents=True, exist_ok=True)
results = {}


# %% November pairs, the panel and its February features
nov = pd.read_parquet(nov_data_dir / 'pairs_clean.parquet',
                      columns=['slug', 'us_party', 'panel', 'gp_date_modified', 'gp_text'])
nov = add_lean(nov.merge(pd.read_parquet(nov_features_dir / 'features.parquet'), on='slug'))
panel = nov[nov.panel]
added = nov[~nov.panel]

feb = pd.read_parquet(features_dir / 'features.parquet', filters=[('slug', 'in', list(panel.slug))])
feb = add_lean(feb.merge(panel[['slug', 'us_party']], on='slug'))
results['n_pairs'] = len(nov)
results['n_panel'] = len(panel)
results['n_new'] = len(added)


# %% Lean shift per party, the panel in February and November and the new politicians
for cohort, frame in [('feb_panel', feb), ('nov_panel', panel), ('nov_new', added)]:
    for short, party in [('dem', 'Democratic'), ('rep', 'Republican')]:
        results[f'{cohort}_{short}_lean_shift'] = frame.loc[frame.us_party == party, 'lean_shift'].mean()

# how many panel GP articles Grokipedia changed between the two snapshots, by date and by text
feb_texts = pd.read_parquet(data_dir / 'pairs_clean.parquet', columns=['slug', 'gp_text'],
                            filters=[('slug', 'in', list(panel.slug))])
both = panel.merge(feb_texts, on='slug', suffixes=('', '_feb'))
results['panel_pct_gp_edited'] = (pd.to_datetime(both.gp_date_modified, utc=True) > FEB_SCRAPE).mean() * 100
same_text = both.gp_text.str.split().str.join(' ') == both.gp_text_feb.str.split().str.join(' ')
results['panel_pct_gp_same_text'] = same_text.mean() * 100

# numpy scalars become plain JSON numbers
results_json = json.dumps(results, indent=2, default=lambda value: value.item())
(out_dir / 'results.json').write_text(results_json + '\n')


# %% Figure in the report's style, serif 9 pt at the report's text width
mpl.use('agg')  # no GUI backend, which would round figure sizes to whole pixels
sns.set_theme(style='whitegrid', context='paper', rc={
    'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.constrained_layout.use': True, 'pdf.fonttype': 42, 'grid.color': '0.9'})

# (a) mean lean per party and cohort with 95% CI, (b) share of November articles above each shift
figure, (mean_axis, shift_axis) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 0.38 * TEXTWIDTH),
                                               width_ratios=[1.1, 1])
rows = [(f'{party[:3]}. {label}', frame[frame.us_party == party])
        for party in ['Democratic', 'Republican']
        for label, frame in [('Feb', feb), ('Nov', panel), ('new', added)]]
for y, (_, group) in enumerate(rows):
    mean_axis.hlines(y, group.wp_lean.mean(), group.gp_lean.mean(), color=DIFF, lw=2, zorder=1)
    # WP hollow on top of GP filled, so both stay visible where a cohort does not move
    for side, color in [('wp', WP), ('gp', GP)]:
        values = group[f'{side}_lean']
        mean_axis.errorbar(values.mean(), y, xerr=1.96 * values.sem(), fmt='o', color=color,
                           mfc='none' if side == 'wp' else color, mew=1.5, ms=6, lw=1,
                           zorder=3 if side == 'wp' else 2)
# direct labels above the widest row, the new Republicans
for side, color, label in [('wp', WP, 'Wikipedia'), ('gp', GP, 'Grokipedia')]:
    mean_axis.text(rows[-1][1][f'{side}_lean'].mean(), len(rows) - 1.25, label, color=color,
                   ha='center', va='bottom')
mean_axis.set(yticks=range(len(rows)), yticklabels=[label for label, _ in rows],
              ylim=(len(rows) - 0.4, -0.6), xlim=(-0.19, 0.37),
              xlabel='Mean lean score (right = conservative)')
mean_axis.grid(axis='y', visible=False)
mean_axis.set_title('(a)', loc='left')

# share at each shift counts the article at that shift itself, so the curve ends at 1/n
for party, style in [('Republican', '-'), ('Democratic', '--')]:
    shifts = np.sort(nov.loc[nov.us_party == party, 'lean_shift'])
    shift_axis.step(shifts, np.linspace(100, 0, len(shifts), endpoint=False), where='post',
                    color=DIFF, ls=style, lw=1.2, label=party)
shift_axis.set(xlabel='Lean shift, GP $-$ WP', ylabel='Share of articles (%)', xlim=(-0.4, 0.8),
               ylim=(0, 100), xticks=np.arange(-0.25, 0.8, 0.25), yticks=range(0, 101, 25))
shift_axis.legend(loc='upper right', frameon=False)
shift_axis.set_title('(b)', loc='left')
for axis in [mean_axis, shift_axis]:
    axis.axvline(0, color=REF, lw=0.8)
figure.savefig(out_dir / 'rescrape_party.pdf')
