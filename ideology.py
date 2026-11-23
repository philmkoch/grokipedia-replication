"""Does Grokipedia's lean follow the politician's ideology, beyond the party label?

    python ideology.py FEATURES_DIR NOV_DATA_DIR NOV_FEATURES_DIR EXT_DIR OUT_DIR

FEATURES_DIR, NOV_DATA_DIR and NOV_FEATURES_DIR are as for rescrape.py. EXT_DIR receives
Voteview's members of all Congresses and the bioguide ids of the politicians on Wikidata, and
anything already there is reused. Ideology is the first DW-NOMINATE dimension (left negative,
right positive), comparable across Congresses. Because it comes from roll-call votes and not
from either text, regressing both leans on it compares how strongly each encyclopedia writes
a politician in the voice of their own side. Writes OUT_DIR/results.json and the figure
ideology as PDF.
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
from scipy import stats

from scraping import get

SIDES = ['wp', 'gp']
TEXTWIDTH = 418.26 / 72.27  # text width of the report in inches, so figure text prints at 9 pt
WP, GP = '#004488', '#DDAA33'  # Tol high-contrast blue and yellow, apart in greyscale too
DIFF, REF = '#777777', 'black'  # GP vs WP quantities, reference lines
ERA_BOUNDS = [1789, 1900, 1965, 2000, 2031]  # by the last year in Congress
MODERN = 2000  # served in Congress in 2000 or later
BIN = 0.1  # width of the ideology bins in the figure
MIN_BIN = 10  # politicians a bin needs to be drawn
N_BOOT = 2000
SEED = 0
VOTEVIEW_URL = 'https://voteview.com/static/data/out/members/HSall_members.csv'
SPARQL = 'https://query.wikidata.org/sparql'


def add_lean(frame):
    """Lean is P(supports conservatives) minus P(supports liberals), shift is GP lean minus WP lean."""
    for side in SIDES:
        frame[f'{side}_lean'] = frame[f'{side}_stance_con'] - frame[f'{side}_stance_lib']
    frame['lean_shift'] = frame.gp_lean - frame.wp_lean
    return frame


def ols(frame, outcome, regressors):
    """OLS with heteroskedasticity-robust (HC1) standard errors: coefficient, SE and p per term."""
    frame = frame.dropna(subset=[outcome, *regressors])
    X = np.column_stack([np.ones(len(frame))] + [frame[r].astype(float) for r in regressors])
    y = frame[outcome].to_numpy(float)
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residual = y - X @ beta
    covariance = xtx_inv @ (X.T * residual ** 2) @ X @ xtx_inv * len(y) / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(covariance))
    p = 2 * stats.norm.sf(np.abs(beta / se))
    fit = {'n': len(y)}
    for name, b, s, q in zip(['const', *regressors], beta, se, p):
        fit.update({f'{name}_coef': b, f'{name}_se': s, f'{name}_p': q})
    return fit


def slope(x, y):
    return np.cov(x, y)[0, 1] / np.var(x, ddof=1)


features_dir, nov_data_dir, nov_features_dir, ext_dir, out_dir = map(Path, sys.argv[1:6])
ext_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
results = {}
rng = np.random.default_rng(SEED)


# %% Downloads: Voteview members, bioguide ids from Wikidata
people = pd.read_parquet(nov_data_dir / 'pairs_clean.parquet',
                         columns=['slug', 'wd_id', 'name', 'us_party', 'panel', 'wp_word_count', 'gp_word_count'])
people = add_lean(people.merge(pd.read_parquet(nov_features_dir / 'features.parquet'), on='slug'))

voteview_file = ext_dir / 'HSall_members.csv'
if not voteview_file.exists():
    voteview_file.write_bytes(get(VOTEVIEW_URL).content)
wikidata_file = ext_dir / 'wikidata_ids.csv'  # shared with musk_support.py, which adds FEC ids
if not wikidata_file.exists():
    rows, ids = [], people.wd_id.dropna().unique().tolist()
    for start in range(0, len(ids), 300):
        values = ' '.join(f'wd:{person}' for person in ids[start:start + 300])
        query = (f'SELECT ?p ?fec ?bio WHERE {{ VALUES ?p {{{values}}} '
                 'OPTIONAL {?p wdt:P1839 ?fec} OPTIONAL {?p wdt:P1157 ?bio} }')
        for binding in get(SPARQL, {'query': query, 'format': 'json'}).json()['results']['bindings']:
            rows.append({'wd_id': binding['p']['value'].rsplit('/', 1)[1],
                         'fec_id': binding.get('fec', {}).get('value'),
                         'bioguide': binding.get('bio', {}).get('value')})
    pd.DataFrame(rows).drop_duplicates().to_csv(wikidata_file, index=False)


# %% Politicians with a score: career mean of the first dimension, last year in Congress
members = pd.read_csv(voteview_file, usecols=['congress', 'chamber', 'bioguide_id', 'nominate_dim1'])
members = members[members.chamber != 'President'].dropna(subset=['bioguide_id', 'nominate_dim1'])
careers = members.groupby('bioguide_id').agg(ideology=('nominate_dim1', 'mean'), last=('congress', 'max'))
careers['last_year'] = 1789 + 2 * (careers['last'] - 1)
bioguide = pd.read_csv(wikidata_file).dropna(subset=['bioguide']).drop_duplicates('wd_id')
people = people.merge(bioguide[['wd_id', 'bioguide']], on='wd_id', how='left')
people = people.merge(careers, left_on='bioguide', right_index=True, how='left')
scored = people.dropna(subset=['ideology']).copy()
scored['republican'] = scored.us_party == 'Republican'
scored['log_length_ratio'] = np.log10(scored.gp_word_count / scored.wp_word_count)
scored['era'] = pd.cut(scored.last_year, ERA_BOUNDS, right=False, labels=False)
eras = [f'era_{era}' for era in range(1, len(ERA_BOUNDS) - 1)]  # the first era is the baseline
for era in range(1, len(ERA_BOUNDS) - 1):
    scored[f'era_{era}'] = scored.era == era
modern = scored[scored.last_year >= MODERN]
results['n_scored'] = len(scored)
results['n_scored_modern'] = len(modern)
for short, party in [('dem', 'Democratic'), ('rep', 'Republican')]:
    results[f'n_scored_{short}'] = int((scored.us_party == party).sum())
    results[f'ideology_mean_{short}'] = scored.loc[scored.us_party == party, 'ideology'].mean()


# %% How steeply each lean rises with ideology, pooled and within each party
for sample_name, sample in [('all', scored), ('modern', modern)]:
    for scope, frame in [('pooled', sample), ('dem', sample[~sample.republican]),
                         ('rep', sample[sample.republican])]:
        key = f'{sample_name}_{scope}'
        x, wp, gp = frame.ideology.to_numpy(), frame.wp_lean.to_numpy(), frame.gp_lean.to_numpy()
        results[f'slope_wp_{key}'] = slope(x, wp)
        results[f'slope_gp_{key}'] = slope(x, gp)
        results[f'slope_ratio_{key}'] = slope(x, gp) / slope(x, wp)
        draws = rng.integers(0, len(x), (N_BOOT, len(x)))
        ratios = [slope(x[d], gp[d]) / slope(x[d], wp[d]) for d in draws]
        results[f'slope_ratio_ci_low_{key}'], results[f'slope_ratio_ci_high_{key}'] = np.percentile(ratios, [2.5, 97.5])
        # the shift on ideology: its slope is the GP minus WP slope, its constant the shift at the centre
        fit = ols(frame, 'lean_shift', ['ideology'])
        for term in ['const', 'ideology']:
            for stat in ['coef', 'se', 'p']:
                results[f'shift_{term}_{stat}_{key}'] = fit[f'{term}_{stat}']

# the party gap split into the shift at the centre and the stronger slope, from the pooled fit
centre, steepness = results['shift_const_coef_all_pooled'], results['shift_ideology_coef_all_pooled']
for short in ['dem', 'rep']:
    results[f'predicted_shift_{short}'] = centre + steepness * results[f'ideology_mean_{short}']


# %% Party beyond ideology: the Republican label once ideology, era and article length are held fixed
for label, regressors in [('party', ['ideology', 'republican']),
                          ('party_era', ['ideology', 'republican', *eras]),
                          ('party_era_length', ['ideology', 'republican', *eras, 'log_length_ratio'])]:
    fit = ols(scored, 'lean_shift', regressors)
    results[f'ols_{label}_n'] = fit['n']
    for term in ['ideology', 'republican', 'log_length_ratio']:
        if f'{term}_coef' in fit:
            for stat in ['coef', 'se', 'p']:
                results[f'ols_{label}_{term}_{stat}'] = fit[f'{term}_{stat}']

# members whose votes sit on the other party's side: conservative Democrats, liberal Republicans
cross = scored[scored.republican != (scored.ideology > 0)]
conservative_democrats = cross[~cross.republican]
results['n_conservative_democrats'] = len(conservative_democrats)
results['n_liberal_republicans'] = int(cross.republican.sum())
results['lean_shift_conservative_democrats'] = conservative_democrats.lean_shift.mean()
results['pct_moved_right_conservative_democrats'] = (conservative_democrats.lean_shift > 0).mean() * 100
# compared with Democrats and Republicans of the same range of votes
low, high = conservative_democrats.ideology.min(), conservative_democrats.ideology.max()
near = scored[scored.ideology.between(low - 0.2, high) & ~scored.index.isin(cross.index)]
for short, frame in [('dem', near[~near.republican]), ('rep', near[near.republican])]:
    results[f'lean_shift_near_conservative_democrats_{short}'] = frame.lean_shift.mean()
    results[f'n_near_conservative_democrats_{short}'] = len(frame)
    results[f'mannwhitney_p_conservative_democrats_vs_{short}'] = stats.mannwhitneyu(
        conservative_democrats.lean_shift, frame.lean_shift).pvalue


# %% Robustness: the stronger slope is not the longer text, it holds where GP is barely longer
scored['length_tercile'] = pd.qcut(scored.log_length_ratio, 3, labels=False)
for tercile, frame in scored.groupby('length_tercile'):
    results[f'length_ratio_median_tercile_{tercile}'] = 10 ** frame.log_length_ratio.median()
    results[f'slope_ratio_tercile_{tercile}'] = (slope(frame.ideology, frame.gp_lean)
                                                 / slope(frame.ideology, frame.wp_lean))


# %% Robustness: the February snapshot of the panel
feb = add_lean(pd.read_parquet(features_dir / 'features.parquet',
                               filters=[('slug', 'in', list(scored.slug[scored.panel]))]))
feb = feb.merge(scored[['slug', 'ideology', 'republican']], on='slug')
results['n_scored_feb'] = len(feb)
for scope, frame in [('pooled', feb), ('dem', feb[~feb.republican]), ('rep', feb[feb.republican])]:
    results[f'feb_slope_ratio_{scope}'] = slope(frame.ideology, frame.gp_lean) / slope(frame.ideology, frame.wp_lean)
    fit = ols(frame, 'lean_shift', ['ideology'])
    for stat in ['coef', 'se', 'p']:
        results[f'feb_shift_ideology_{stat}_{scope}'] = fit[f'ideology_{stat}']

results_json = json.dumps(results, indent=2, default=lambda value: value.item())
(out_dir / 'results.json').write_text(results_json + '\n')


# %% Figure in the report's style
mpl.use('agg')  # no GUI backend, which would round figure sizes to whole pixels
sns.set_theme(style='whitegrid', context='paper', rc={
    'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.constrained_layout.use': True, 'pdf.fonttype': 42, 'grid.color': '0.9'})

# (a) mean WP and GP lean per ideology bin, (b) mean shift per bin and party, 95% CI each
scored['bin'] = (np.floor(scored.ideology / BIN) + 0.5) * BIN
figure, (lean_axis, shift_axis) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 0.4 * TEXTWIDTH), sharex=True)
bins = scored.groupby('bin').filter(lambda group: len(group) >= MIN_BIN).groupby('bin')
for side, color, label in [('wp', WP, 'Wikipedia'), ('gp', GP, 'Grokipedia')]:
    means, cis = bins[f'{side}_lean'].mean(), 1.96 * bins[f'{side}_lean'].sem()
    lean_axis.errorbar(means.index, means, yerr=cis, fmt='o', color=color, ms=4, lw=1,
                       mfc='none' if side == 'wp' else color, mew=1.2, label=label)
lean_axis.set(xlabel='Ideology (DW-NOMINATE)', ylabel='Mean lean score')
lean_axis.legend(loc='upper left', frameon=False)
lean_axis.set_title('(a)', loc='left')
for party, marker, fill in [('Democratic', 's', 'none'), ('Republican', 'o', DIFF)]:
    frame = scored[scored.us_party == party]
    groups = frame.groupby('bin').filter(lambda group: len(group) >= MIN_BIN).groupby('bin').lean_shift
    shift_axis.errorbar(groups.mean().index, groups.mean(), yerr=1.96 * groups.sem(), fmt=marker,
                        color=DIFF, mfc=fill, ms=4, lw=1, mew=1.2, label=party)
shift_axis.set(xlabel='Ideology (DW-NOMINATE)', ylabel='Lean shift, GP $-$ WP')
shift_axis.legend(loc='lower right', frameon=False)
shift_axis.set_title('(b)', loc='left')
for axis in [lean_axis, shift_axis]:
    axis.axhline(0, color=REF, lw=0.8)
    axis.axvline(0, color=REF, lw=0.4, ls=':')
figure.savefig(out_dir / 'ideology.pdf')
