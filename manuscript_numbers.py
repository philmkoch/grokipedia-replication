"""Checks every number of the manuscript against what the analysis code gives.

    python manuscript_numbers.py [ROOT]

Run it after reproduce.py has filled out/. Part 1 lists each number in the manuscript's text, figure legends
and figures with the value the code gives, rounded to the digits the manuscript prints. Most values come
from the results.json files of the analysis scripts. The few that no script writes are computed here: the
reference group of all 98,013 biographies, the current Congress, the America PAC totals, the composition of
the sample and the dates of the November scrape. Part 2 compares the two figures and every value in out/
with the copies shipped in expected/ and results/. Writes out/manuscript_check.md and .csv, and exits with
status 1 if a number does not reproduce.
"""
# %% Setup
import ast
import csv
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent).resolve()
OUT = ROOT / 'out'
AMERICA_PAC = 'C00879510'  # FEC id of Musk's super PAC, as in musk_support.py
OFFICES = {'house', 'senate', 'governor', 'cabinet', 'mayor', 'president'}  # lists of 2026 office holders


def load(name):
    return json.loads((OUT / name).read_text())


main, november, ideology, musk, paper = (load(name) for name in [
    'results.json', 'november/results.json', 'ideology/results.json', 'musk/results.json', 'paper/results.json'])


def constant(script, name):
    """A module-level constant of one of the scripts, so the manuscript's parameters are checked against the code."""
    for node in ast.parse((ROOT / script).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                for target in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f'{name} not found in {script}')


def add_lean(frame):
    for side in ['wp', 'gp']:
        frame[f'{side}_lean'] = frame[f'{side}_stance_con'] - frame[f'{side}_stance_lib']
    frame['lean_shift'] = frame.gp_lean - frame.wp_lean
    return frame


def slope(x, y):
    return np.cov(x, y)[0, 1] / np.var(x, ddof=1)


# %% Numbers no script writes
extra = {}
everyone = add_lean(pd.read_parquet(ROOT / 'features/features.parquet'))
extra['n_pairs_features'] = len(everyone)
extra['all_lean_shift'] = everyone.lean_shift.mean()
extra['all_con_shift'] = (everyone.gp_stance_con - everyone.wp_stance_con).mean()
extra['all_lib_shift'] = (everyone.gp_stance_lib - everyone.wp_stance_lib).mean()
extra['dem_con_times_all'] = main['stance_con_shift_dem'] / extra['all_con_shift']
extra['dem_lib_times_all'] = main['stance_lib_shift_dem'] / extra['all_lib_shift']

pairs = pd.read_parquet(ROOT / 'data/november/pairs_clean.parquet',
                        columns=['slug', 'wd_id', 'us_party', 'sources', 'wp_fetched_at', 'gp_fetched_at'])
tags = pairs.sources.str.split('|')
holds_office = tags.map(lambda found: bool(OFFICES & set(found)))
extra['n_no_office'] = int((pairs.sources == 'panel').sum())  # on no 2026 list: the historical panel
extra['n_office'] = int(holds_office.sum())
extra['n_nominee_only'] = int((~holds_office & (pairs.sources != 'panel')).sum())
fetched = pd.to_datetime(pd.concat([pairs.wp_fetched_at, pairs.gp_fetched_at]), utc=True).dt.date.unique()
extra['november_fetch_dates'] = ' '.join(sorted(map(str, fetched)))
extra['days_election_to_scrape'] = (date.fromisoformat(extra['november_fetch_dates'].split()[-1])
                                    - date(2026, 11, 3)).days

# the 119th Congress: the scored politicians of ideology.py who sat in it
people = add_lean(pairs.merge(pd.read_parquet(ROOT / 'features/november/features.parquet'), on='slug'))
members = pd.read_csv(ROOT / 'data/external/HSall_members.csv',
                      usecols=['congress', 'chamber', 'bioguide_id', 'nominate_dim1'])
members = members[members.chamber != 'President'].dropna(subset=['bioguide_id', 'nominate_dim1'])
careers = members.groupby('bioguide_id').agg(ideology=('nominate_dim1', 'mean'))
bioguide = pd.read_csv(ROOT / 'data/external/wikidata_ids.csv').dropna(subset=['bioguide']).drop_duplicates('wd_id')
people = people.merge(bioguide[['wd_id', 'bioguide']], on='wd_id', how='left')
scored = people.merge(careers, left_on='bioguide', right_index=True, how='left').dropna(subset=['ideology'])
current = pd.read_csv(ROOT / 'data/external/HS119_members.csv', usecols=['chamber', 'bioguide_id'])
current = scored[scored.bioguide.isin(current[current.chamber != 'President'].bioguide_id)]
extra['n_scored'] = len(scored)
extra['n_119th'] = len(current)
extra['slope_ratio_119th'] = slope(current.ideology, current.gp_lean) / slope(current.ideology, current.wp_lean)

# America PAC, 2024 and 2026 cycles, cleaned as in musk_support.py (junk filings out, duplicates dropped)
max_item = constant('musk_support.py', 'MAX_ITEM')
for cycle in [2024, 2026]:
    spending = pd.read_csv(ROOT / f'data/external/independent_expenditure_{cycle}.csv', dtype=str)
    spending['amount'] = pd.to_numeric(spending.exp_amo, errors='coerce')
    spending = spending[(spending.spe_id == AMERICA_PAC) & spending.amount.between(0, max_item)]
    extra[f'america_pac_{cycle}'] = spending.drop_duplicates(['spe_id', 'tran_id', 'file_num']).amount.sum() / 1e6
extra['america_pac_total'] = extra['america_pac_2024'] + extra['america_pac_2026']


# %% Part 1: the claims of the manuscript
claims = []


def show(value, digits):
    return f'{value:.{digits + 3}f}' if isinstance(value, (float, np.floating)) and digits is not None else str(value)


def equal(where, statement, printed, value, digits=None):
    """printed is the number as the manuscript prints it, value what the code gives, rounded to `digits`."""
    ok = round(float(value), digits) == printed if digits is not None else value == printed
    claims.append((where, statement, printed, show(value, digits), ok))


def below(where, statement, bound, value):
    claims.append((where, statement, f'< {bound:.3g}', f'{value:.3g}', bool(value < bound)))


def above(where, statement, bound, value):
    claims.append((where, statement, f'> {bound:.3g}', f'{value:.3g}', bool(value > bound)))


def same(where, statement, first, second):
    """The figure script and the analysis script compute one number twice, so the two must agree."""
    claims.append((where, statement, f'{first:.6f}', f'{second:.6f}', bool(np.isclose(first, second, rtol=1e-6))))


def within(where, statement, low, high, values, digits=2):
    """All of `values` lie in [low, high] and the extremes are the printed ends of the range."""
    ok = round(min(values), digits) == low and round(max(values), digits) == high
    claims.append((where, statement, f'{low} to {high}', f'{min(values):.{digits + 3}f} to {max(values):.{digits + 3}f}', ok))


def ci(where, statement, low, high, centre, half, digits):
    equal(where, f'{statement}, lower end', low, centre - half, digits)
    equal(where, f'{statement}, upper end', high, centre + half, digits)


# Abstract and Introduction
equal('Abstract', "biographies of 1,614 politicians", 1614, november['n_pairs'])
equal('Abstract', "Grokipedia moves 89% of Republican biographies to the right", 89, main['pct_moved_right_rep'], 0)
equal('Abstract', "1,022 politicians with roll call voting records", 1022, ideology['n_scored'])
equal('Abstract', "lean rises about twice as steeply (ratio 2.08)", 2.0, ideology['slope_ratio_all_pooled'], 0)
equal('Introduction', "DeWine sentence, Wikipedia, P(conservatives)", 0.000005, paper['dewine_p_con_wp'], 6)
equal('Introduction', "DeWine sentence, Grokipedia, P(conservatives)", 0.9999996, paper['dewine_p_con_gp'], 7)
equal('Introduction', "959 of the 1,614 held no office (on no 2026 list)", 959, extra['n_no_office'])
equal('Introduction', "1,614 politicians in February and November", 1614, november['n_pairs'])
equal('Introduction', "1,022 politicians linked to DW-NOMINATE", 1022, extra['n_scored'])
equal('Introduction', "November snapshot five days after the midterms", 5, extra['days_election_to_scrape'])

# Results, party averages
equal('Results', "572 Republican biographies", 572, main['n_republicans'])
equal('Results', "89% of them shift right", 89, main['pct_moved_right_rep'], 0)
equal('Results', "mean shift of Republicans", 0.126, main['lean_shift_rep'], 3)
ci('Results', "Republicans' 95% CI", 0.117, 0.135, main['lean_shift_rep'], main['lean_shift_ci_rep'], 3)
equal('Results', "DeWine lean on Wikipedia", 0.056, main['dewine_lean_wp'], 3)
equal('Results', "DeWine lean on Grokipedia", 0.182, main['dewine_lean_gp'], 3)
equal('Results', "average Republican, Wikipedia lean", 0.053, main['lean_mean_wp_rep'], 3)
equal('Results', "average Republican, Grokipedia lean", 0.180, main['lean_mean_gp_rep'], 3)
equal('Results', "mean shift of Democrats", 0.001, main['lean_shift_dem'], 3)
ci('Results', "Democrats' 95% CI", -0.006, 0.007, main['lean_shift_dem'], main['lean_shift_ci_dem'], 3)
equal('Results', "590 Democratic biographies", 590, main['n_democrats'])
equal('Results', "53% of them move right", 53, main['pct_moved_right_dem'], 0)
equal('Results', "all 98,013 biographies", 98013, extra['n_pairs_features'])
equal('Results', "mean shift over all 98,013 biographies", 0.002, extra['all_lean_shift'], 3)
equal('Results', "Republicans, P(conservatives) up", 0.129, main['stance_con_shift_rep'], 3)
equal('Results', "Republicans, P(liberals) up", 0.002, main['stance_lib_shift_rep'], 3)
equal('Results', "Democrats, P(conservatives) up", 0.044, main['stance_con_shift_dem'], 3)
equal('Results', "Democrats, P(liberals) up", 0.043, main['stance_lib_shift_dem'], 3)
equal('Results', "Democrats' rise in P(conservatives), times that of the average biography", 7, extra['dem_con_times_all'], 0)
equal('Results', "Democrats' rise in P(liberals), times that of the average biography", 10, extra['dem_lib_times_all'], 0)
below('Results', "dictionary check, Republicans (Wilcoxon P)", 1e-14, main['gst_wilcoxon_p_rep'])
equal('Results', "dictionary check, Democrats (Wilcoxon P)", 0.52, main['gst_wilcoxon_p_dem'], 2)
first, stop = re.search(r'range\((\d+), (\d+)\)', (ROOT / 'results.py').read_text()).groups()
equal('Methods', "partisan phrases of the 104th to 114th Congresses", '104-114', f'{first}-{int(stop) - 1}')
equal('Results', "November, same Republicans", 0.125, november['nov_panel_rep_lean_shift'], 3)
equal('Results', "November, same Democrats", 0.0, november['nov_panel_dem_lean_shift'], 3)

# Results, ideology
equal('Results', "Grokipedia slope on DW-NOMINATE", 0.437, ideology['slope_gp_all_pooled'], 3)
equal('Results', "Wikipedia slope on DW-NOMINATE", 0.210, ideology['slope_wp_all_pooled'], 3)
equal('Results', "slope ratio", 2.08, ideology['slope_ratio_all_pooled'], 2)
equal('Results', "slope ratio, lower bootstrap bound", 1.99, ideology['slope_ratio_ci_low_all_pooled'], 2)
equal('Results', "slope ratio, upper bootstrap bound", 2.17, ideology['slope_ratio_ci_high_all_pooled'], 2)
equal('Results', "shift offset to the right", 0.067, ideology['shift_const_coef_all_pooled'], 3)
equal('Results', "shift per unit of ideology", 0.227, ideology['shift_ideology_coef_all_pooled'], 3)
equal('Results', "488 members of the current Congress", 488, extra['n_119th'])
equal('Results', "slope ratio in the current Congress", 2.02, extra['slope_ratio_119th'], 2)
robust = [ideology['slope_ratio_all_dem'], ideology['slope_ratio_all_rep'], ideology['feb_slope_ratio_pooled'],
          ideology['slope_ratio_tercile_0']]
within('Results', "ratio within each party, in February alone, and where the lengths are closest", 1.94, 2.33, robust)
equal('Fig. 2', "1,022 politicians with a DW-NOMINATE score", 1022, paper['n_scored'])
equal('Fig. 2', "12 conservative Democrats", 12, paper['n_conservative_democrats'])
equal('Fig. 2', "at least 10 politicians per bin", 10, constant('paper_figures.py', 'MIN_BIN'))
equal('Fig. 2', "bins of 0.1", 0.1, constant('paper_figures.py', 'BIN'), 1)
same('Fig. 2', "slope printed in panel a, Grokipedia: figure script equals ideology.py", ideology['slope_gp_all_pooled'],
     paper['slope_gp_all_pooled'])
same('Fig. 2', "slope printed in panel a, Wikipedia: figure script equals ideology.py", ideology['slope_wp_all_pooled'],
     paper['slope_wp_all_pooled'])
same('Fig. 2', "offset printed in panel b: figure script equals ideology.py", ideology['shift_const_coef_all_pooled'],
     paper['shift_const_coef_all_pooled'])
same('Fig. 2', "slope printed in panel b: figure script equals ideology.py", ideology['shift_ideology_coef_all_pooled'],
     paper['shift_ideology_coef_all_pooled'])

# Results, Musk
equal('Results', "27 Republicans in America PAC races", 27, musk['n_musk_rep'])
equal('Methods', "America PAC race: at least $100,000", 100000, constant('musk_support.py', 'MUSK_MIN'))
equal('Methods', "contested race: at least $1 million", 1000000, constant('musk_support.py', 'CONTESTED'))
equal('Results', "their mean shift", 0.172, musk['nov_lean_shift_musk_rep'], 3)
equal('Results', "31 Republicans in contested races without America PAC money", 31, musk['n_contested_rep'])
equal('Results', "their mean shift", 0.156, musk['nov_lean_shift_contested_rep'], 3)
equal('Results', "difference", 0.016, musk['nov_musk_minus_contested_rep'], 3)
ci('Results', "difference's 95% CI", -0.039, 0.072, musk['nov_musk_minus_contested_rep'],
   musk['nov_musk_minus_contested_ci_rep'], 3)
equal('Results', "other Republicans' shift", 0.144, musk['nov_lean_shift_rest_rep'], 3)
below('Results', "upper CI bound is below half of what other Republicans receive", musk['nov_lean_shift_rest_rep'] / 2,
      musk['nov_musk_minus_contested_rep'] + musk['nov_musk_minus_contested_ci_rep'])
equal('Results', "Musk-race Republicans' articles edited in the campaign (%)", 22, musk['pct_edited_campaign_musk_rep'], 0)
equal('Results', "contested-race Republicans' articles edited in the campaign (%)", 26,
      musk['pct_edited_campaign_contested_rep'], 0)

# Discussion, Methods, figure legends
equal('Discussion', "592 of the 1,614 politicians have no score", 592, november['n_pairs'] - ideology['n_scored'])
equal('Discussion', "Wikipedia's own slope", 0.210, ideology['slope_wp_all_pooled'], 3)
equal('Methods', "98,013 pairs", 98013, main['n_pairs'])
equal('Methods', "1,162 US politicians with one major party (590 + 572)", 1162,
      paper['n_feb_panel_dem'] + paper['n_feb_panel_rep'])
equal('Methods', "the same 1,162 are the November panel", 1162, november['n_panel'])
equal('Methods', "November snapshot on 8 November 2026", '2026-11-08', extra['november_fetch_dates'])
above('Methods', "America PAC reported more than $270 million in 2024 and 2026 (USD million)", 270,
      extra['america_pac_total'])
equal('Methods', "bootstrap resamples", 2000, constant('ideology.py', 'N_BOOT'))
equal('Methods', "confidence intervals of means are 1.96 standard errors", True,
      all('1.96 *' in (ROOT / script).read_text() for script in ['results.py', 'paper_figures.py', 'musk_support.py']))
equal('Methods', "party differences use Mann-Whitney tests", True,
      all('mannwhitneyu' in (ROOT / script).read_text() for script in ['results.py', 'musk_support.py', 'ideology.py']))
equal('Methods', "sentences of at least five words", 5, constant('features.py', 'MIN_SENTENCE_WORDS'))
source = (ROOT / 'features.py').read_text()
equal('Methods', "hypotheses 'The author of this text supports conservatives/liberals.'", True,
      "supports {camp}." in source and "('con', 'conservatives'), ('lib', 'liberals')" in source)
equal('Methods', "Political DEBATE pinned to a fixed model version", True,
      bool(re.search(r"Political_DEBATE_large_v1\.0', '[0-9a-f]{40}'", source)))
same('Fig. 1', "share moving right, Republicans (label in panel b): figure script equals results.py",
     main['pct_moved_right_rep'], paper['pct_moved_right_rep'])
same('Fig. 1', "share moving right, Democrats (label in panel b): figure script equals results.py",
     main['pct_moved_right_dem'], paper['pct_moved_right_dem'])
same('Fig. 1', "net shift of Republicans (bracket in panel c): figure script equals results.py",
     main['lean_shift_rep'], paper['lean_shift_rep'])
same('Fig. 1', "net shift of Democrats (bracket in panel c): figure script equals results.py",
     main['lean_shift_dem'], paper['lean_shift_dem'])

# %% Part 2: the figures and every value against the shipped reference
FIGURES = [('fig1', 'Fig. 1'), ('fig2', 'Fig. 2')]
figure_rows = []
for name, label in FIGURES:
    new, old = OUT / 'figures' / f'{name}.png', ROOT / 'expected' / f'{name}.png'
    if hashlib.sha256(new.read_bytes()).digest() == hashlib.sha256(old.read_bytes()).digest():
        figure_rows.append((label, 'identical to the figure in the manuscript (same bytes)', True))
        continue
    a, b = mpimg.imread(new), mpimg.imread(old)
    # a renderer or font version can move a few pixels, a different figure moves many
    gap = float(np.abs(a - b).mean()) if a.shape == b.shape else np.inf
    figure_rows.append((label, f'differs from the manuscript figure, mean pixel gap {gap:.5f}', gap < 0.005))


def differences(want, got, path=''):
    """Paths of the values in `got` that differ from `want` (relative tolerance 1e-4, NaN equals NaN)."""
    if isinstance(want, dict):
        found = [f'{path}/{key}: missing' for key in want if key not in got]
        for key in want:
            if key in got:
                found += differences(want[key], got[key], f'{path}/{key}')
        return found
    if isinstance(want, str) or isinstance(got, str):
        return [] if want == got else [f'{path}: {want!r} vs {got!r}']
    return [] if np.isclose(want, got, rtol=1e-4, atol=1e-9, equal_nan=True) else [f'{path}: {want} vs {got}']


def n_values(item):
    return sum(map(n_values, item.values())) if isinstance(item, dict) else 1


reference_rows = []
for out_name, reference_name in [('results.json', 'results/results.json'),
                                 ('november/results.json', 'results/november/results.json'),
                                 ('ideology/results.json', 'results/ideology/results.json'),
                                 ('musk/results.json', 'results/musk/results.json'),
                                 ('paper/results.json', 'results/paper/results.json')]:
    want = json.loads((ROOT / reference_name).read_text())
    found = differences(want, load(out_name))
    reference_rows.append((f'out/{out_name}', reference_name, n_values(want), found))

# %% Report
lines = ['# Check of every number in the manuscript', '', '## Part 1. Numbers in the text, legends and figures', '',
         '| # | Where | Manuscript says | Printed | Reproduced | Status |', '|--:|---|---|---|---|---|']
for number, (where, statement, printed, shown, ok) in enumerate(claims, 1):
    lines.append(f'| {number} | {where} | {statement} | {printed} | {shown} | {"ok" if ok else "**FAIL**"} |')
lines += ['', '## Not in the manuscript, for context', '',
          'Slope ratios in the February snapshot alone, within each party: '
          f'Democrats {ideology["feb_slope_ratio_dem"]:.2f}, Republicans {ideology["feb_slope_ratio_rep"]:.2f} '
          f'(pooled {ideology["feb_slope_ratio_pooled"]:.2f}, {ideology["n_scored_feb"]} politicians).']
lines += ['', '## Sample composition behind "959 held no office"', '',
          f'On no 2026 list (historical panel): {extra["n_no_office"]}. On a list of office holders (Congress, governors, '
          f'cabinet, mayors, president): {extra["n_office"]}. Nominees for 2026 without such an office: '
          f'{extra["n_nominee_only"]}. Sum {extra["n_no_office"] + extra["n_office"] + extra["n_nominee_only"]}.',
          '', f'America PAC independent expenditures: {extra["america_pac_2024"]:.1f} million USD in the 2024 cycle, '
          f'{extra["america_pac_2026"]:.1f} million in 2026, {extra["america_pac_total"]:.1f} million in total.',
          '', '## Part 2. Figures and output files against the shipped reference', '', '| Item | Result | Status |',
          '|---|---|---|']
for label, text, ok in figure_rows:
    lines.append(f'| {label} | {text} | {"ok" if ok else "**FAIL**"} |')
for out_name, reference_name, count, found in reference_rows:
    lines.append(f'| {out_name} | {count - len(found)} of {count} values match {reference_name} | '
                 f'{"ok" if not found else "**FAIL**"} |')
failures = [claim for claim in claims if not claim[-1]]
lines += ['', f'**{len(claims) - len(failures)} of {len(claims)} claims reproduce.**']
for out_name, reference_name, count, found in reference_rows:
    for item in found[:10]:
        lines.append(f'- {out_name} {item}')
(OUT / 'manuscript_check.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
with open(OUT / 'manuscript_check.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['where', 'manuscript_says', 'printed', 'reproduced', 'ok'])
    writer.writerows(claims)

print('\n'.join(lines))
bad = failures or [row for row in figure_rows if not row[-1]] or [row for row in reference_rows if row[-1]]
sys.exit(1 if bad else 0)
