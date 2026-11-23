"""The two figures of the short paper, with every plotted number.

    python paper_figures.py DATA_DIR FEATURES_DIR NOV_DATA_DIR NOV_FEATURES_DIR EXT_DIR OUT_DIR

Directories are as for rescrape.py and ideology.py, and EXT_DIR holds Voteview's
HSall_members.csv and wikidata_ids.csv (nothing is downloaded here). Fig. 1: (a) the DeWine
sentence pair, (b) the February share-above-shift curves of the panel, (c) the two layers of
the February shift. Fig. 2: (a) lean and (b) shift by ideology. Writes fig1 and fig2 as
PDF and 300-dpi PNG, and results.json with the plotted values, to OUT_DIR.
"""
# %% Setup
import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns

SIDES = ['wp', 'gp']
TEXTWIDTH = 418.26 / 72.27  # text width of the report in inches, so figure text prints at 9 pt
WP, GP = '#004488', '#DDAA33'  # Tol high-contrast blue and yellow, apart in greyscale too
DIFF, REF = '#777777', 'black'  # GP vs WP quantities, reference lines
BIN = 0.1  # width of the ideology bins, as in ideology.py
MIN_BIN = 10  # politicians a bin needs to be drawn


def add_lean(frame):
    """Lean is P(supports conservatives) minus P(supports liberals), shift is GP lean minus WP lean."""
    for side in SIDES:
        frame[f'{side}_lean'] = frame[f'{side}_stance_con'] - frame[f'{side}_stance_lib']
    frame['lean_shift'] = frame.gp_lean - frame.wp_lean
    return frame


def ols(x, y):
    """Intercept and slope of the least-squares line."""
    slope = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    return y.mean() - slope * x.mean(), slope


data_dir, features_dir, nov_data_dir, nov_features_dir, ext_dir, out_dir = map(Path, sys.argv[1:7])
out_dir.mkdir(parents=True, exist_ok=True)
results = {}

mpl.use('agg')  # no GUI backend, which would round figure sizes to whole pixels
sns.set_theme(style='whitegrid', context='paper', rc={
    'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.constrained_layout.use': True, 'pdf.fonttype': 42, 'grid.color': '0.9'})


# %% Data: November pairs with features, the February panel, the DeWine sentences
nov = pd.read_parquet(nov_data_dir / 'pairs_clean.parquet',
                      columns=['slug', 'wd_id', 'name', 'us_party', 'panel'])
nov = add_lean(nov.merge(pd.read_parquet(nov_features_dir / 'features.parquet'), on='slug'))
panel = nov[nov.panel]
# the party of a panel politician comes from the November pairs, as in rescrape.py
feb = pd.read_parquet(features_dir / 'features.parquet', filters=[('slug', 'in', list(panel.slug))])
feb = add_lean(feb.merge(panel[['slug', 'us_party']], on='slug'))
feb['con_shift'] = feb.gp_stance_con - feb.wp_stance_con
feb['lib_shift'] = feb.gp_stance_lib - feb.wp_stance_lib
results['n_nov'] = len(nov)
results['n_feb_panel_dem'] = int((feb.us_party == 'Democratic').sum())
results['n_feb_panel_rep'] = int((feb.us_party == 'Republican').sum())

sentences = pd.read_parquet(data_dir / 'dewine_sentences.parquet')
dewine = {side: sentences[(sentences.side == side) & (sentences['index'] == i)].iloc[0]
          for side, i in [('wp', 76), ('gp', 2)]}
results['dewine_p_con_wp'] = dewine['wp'].p_con
results['dewine_p_con_gp'] = dewine['gp'].p_con


# %% Figure 1 numbers: the two layers in February, the November curves
PARTIES = [('rep', 'Republican'), ('dem', 'Democratic')]
layers = {}
for short, party in PARTIES:
    group = feb[feb.us_party == party]
    for name, column in [('con', 'con_shift'), ('lib', 'lib_shift')]:
        layers[short, name] = (group[column].mean(), 1.96 * group[column].sem())
        results[f'stance_{name}_shift_{short}'] = group[column].mean()
        results[f'stance_{name}_shift_ci_{short}'] = 1.96 * group[column].sem()
    results[f'lean_shift_{short}'] = group.lean_shift.mean()
    results[f'pct_moved_right_{short}'] = (group.lean_shift > 0).mean() * 100
    results[f'nov_pct_moved_right_{short}'] = (nov.loc[nov.us_party == party, 'lean_shift'] > 0).mean() * 100
    results[f'nov_n_{short}'] = int((nov.us_party == party).sum())


# %% Figure 1
def layout_words(figure, axis, words, x0, x1, bold_phrase):
    """Greedy line wrap by measured width, so a bold phrase can sit inside running text.

    Returns lines of (word, bold, x offset) with x in axes fraction from x0.
    """
    renderer = figure.canvas.get_renderer()
    width = axis.bbox.width

    def measure(text, bold):
        probe = axis.text(0, 0, text, fontfamily='cmb10' if bold else 'cmr10', fontsize=9)
        size = probe.get_window_extent(renderer).width / width
        probe.remove()
        return size

    space = 0.33 * 9 / 72 / (width / figure.dpi)  # a third of an em, in axes fraction
    # cmr10 has no curly quotes, but renders TeX's `` and '' as such
    words = re.sub(r'"(?=\w)', '``', words).replace('"', "''")
    bold_words = set(bold_phrase.split())
    tokens, seen = [], 0
    for word in words.split():
        tokens.append((word, word.strip(',.') in bold_words and seen < len(bold_words)))
        seen += tokens[-1][1]
    lines, line, cursor = [], [], x0
    for word, bold in tokens:
        size = measure(word, bold)
        if line and cursor + size > x1:
            lines.append(line)
            line, cursor = [], x0
        line.append((word, bold, cursor))
        cursor += size + space
    lines.append(line)
    return lines


# dpi 300 while drawing, because text widths measured at the screen dpi differ from the saved ones
figure = plt.figure(figsize=(TEXTWIDTH, 0.8 * TEXTWIDTH), dpi=300)
top, bottom = figure.subfigures(2, 1, height_ratios=[1, 2.35], hspace=0)  # so (a) spans the full width
text_axis = top.add_subplot()
curve_axis, layer_axis = bottom.subplots(1, 2, width_ratios=[1, 1.2])

# (a) the sentence pair as text, probabilities aligned right
text_axis.set_axis_off()
text_axis.set_title('(a)', loc='left')
figure.canvas.draw()  # fixes the axes size, which the wrapping needs
TEXT_START, TEXT_END = 0.14, 0.845
LINE = 1.5 * 9 / 72 / (text_axis.bbox.height / figure.dpi)  # line height in axes fraction
y = 0.97
text_axis.text(1, y, 'P(author supports conservatives)', ha='right', va='top', color=DIFF)
y -= 1.35 * LINE
for side, color, label in [('wp', WP, 'Wikipedia'), ('gp', GP, 'Grokipedia')]:
    row = dewine[side]
    text_axis.text(0, y, label, color=color, ha='left', va='top')
    lines = layout_words(figure, text_axis, row.sentence, TEXT_START, TEXT_END, 'A longstanding pro-life advocate')
    for number, line in enumerate(lines):
        for word, bold, x in line:
            text_axis.text(x, y - number * LINE, word, va='top', ha='left',
                           fontfamily='cmb10' if bold else 'cmr10')
    text_axis.text(1, y, f'{row.p_con:.6f}' if side == 'wp' else f'{row.p_con:.7f}', ha='right', va='top',
                   color=color)
    y -= (len(lines) + 0.45) * LINE

# (c) change in the share of sentences read as pro-conservative and pro-liberal per party. The 95%
# CIs (half-width < 0.01) are narrower than the markers, so no whiskers; the bracket is the net shift
ROWS = {'rep': 0, 'dem': 2}  # each party's pair sits 0.7 apart, the parties 2 apart
for short, y in ROWS.items():
    con, lib = layers[short, 'con'][0], layers[short, 'lib'][0]
    for name, mean, row in [('con', con, y), ('lib', lib, y + 0.7)]:
        layer_axis.plot(mean, row, 'o', color=DIFF, mfc=DIFF if name == 'con' else 'white', mew=1.4, ms=6)
        word = 'pro-conservative' if name == 'con' else 'pro-liberal'
        right_most = mean == max(con, lib) and short == 'rep'  # its label goes left, inside the axes
        layer_axis.annotate(word, (mean, row), xytext=(-9 if right_most else 9, 0), textcoords='offset points',
                            ha='right' if right_most else 'left', va='center', color=DIFF)
    middle = y + 0.35
    layer_axis.annotate('', (con, middle), (lib, middle), arrowprops=dict(arrowstyle='|-|', color=REF,
                                                                         lw=0.8, shrinkA=0, shrinkB=0,
                                                                         mutation_scale=3))
    layer_axis.annotate(f'net {results[f"lean_shift_{short}"]:+.3f}', (max(con, lib), middle),
                        xytext=(6, 0), textcoords='offset points', ha='left', va='center', color=REF)
layer_axis.set(yticks=[0.35, 2.35], yticklabels=['Republicans', 'Democrats'], ylim=(3.3, -0.6),
               xlim=(-0.015, 0.19),
               xlabel='Change in mean probability,\nGrokipedia $-$ Wikipedia')
layer_axis.set_xticks([0, 0.05, 0.1, 0.15])
layer_axis.tick_params(axis='y', length=0)
layer_axis.grid(axis='y', visible=False)
layer_axis.axvline(0, color=REF, lw=0.8)
layer_axis.set_title('(c)', loc='left')

# (b) share of the February panel's biographies with a shift at least x, counting the article at x itself
curve_labels = {}
for short, party in PARTIES:
    shifts = np.sort(feb.loc[feb.us_party == party, 'lean_shift'])
    share = np.linspace(100, 0, len(shifts), endpoint=False)
    curve_axis.step(shifts, share, where='post', color=DIFF, ls='-' if short == 'rep' else '--', lw=1.2)
    value = results[f'pct_moved_right_{short}']
    curve_axis.plot(0, value, 'o', color=DIFF, ms=4)
    # Republicans' label sits right of the dot, above their curve, Democrats' left of theirs, below it
    curve_axis.annotate(f'{value:.0f}%', (0, value), xytext=(5, 3) if short == 'rep' else (-5, -4),
                        textcoords='offset points', ha='left' if short == 'rep' else 'right', va='center',
                        color=DIFF)
curve_axis.text(0.15, 80, 'Republicans', color=DIFF, ha='left', va='center')
curve_axis.text(-0.02, 10, 'Democrats', color=DIFF, ha='right', va='center')
curve_axis.set(xlabel='Lean shift, Grokipedia $-$ Wikipedia', ylabel='Share of biographies (%)', xlim=(-0.35, 0.5),
               ylim=(0, 100), xticks=[-0.25, 0, 0.25, 0.5], yticks=range(0, 101, 25))
curve_axis.axvline(0, color=REF, lw=0.8)
curve_axis.set_title('(b)', loc='left')
figure.savefig(out_dir / 'fig1.pdf')
figure.savefig(out_dir / 'fig1.png', dpi=300)
plt.close(figure)


# %% Ideology: DW-NOMINATE of the career, as in ideology.py
members = pd.read_csv(ext_dir / 'HSall_members.csv', usecols=['congress', 'chamber', 'bioguide_id', 'nominate_dim1'])
members = members[members.chamber != 'President'].dropna(subset=['bioguide_id', 'nominate_dim1'])
careers = members.groupby('bioguide_id').agg(ideology=('nominate_dim1', 'mean'))
bioguide = pd.read_csv(ext_dir / 'wikidata_ids.csv').dropna(subset=['bioguide']).drop_duplicates('wd_id')
people = nov.merge(bioguide[['wd_id', 'bioguide']], on='wd_id', how='left')
people = people.merge(careers, left_on='bioguide', right_index=True, how='left')
scored = people.dropna(subset=['ideology']).copy()
scored['republican'] = scored.us_party == 'Republican'
scored['bin'] = (np.floor(scored.ideology / BIN) + 0.5) * BIN
results['n_scored'] = len(scored)

x = scored.ideology.to_numpy()
fits = {}
for side in SIDES:
    fits[side] = ols(x, scored[f'{side}_lean'].to_numpy())
    results[f'slope_{side}_all_pooled'] = fits[side][1]
    results[f'intercept_{side}_all_pooled'] = fits[side][0]
shift_fit = ols(x, scored.lean_shift.to_numpy())
results['shift_const_coef_all_pooled'], results['shift_ideology_coef_all_pooled'] = shift_fit
cross = scored[~scored.republican & (scored.ideology > 0)]  # Democrats on the right of the scale
results['n_conservative_democrats'] = len(cross)
results['lean_shift_conservative_democrats'] = cross.lean_shift.mean()


# %% Figure 2
figure, (lean_axis, shift_axis) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 0.42 * TEXTWIDTH), sharex=True)
bins = scored.groupby('bin').filter(lambda group: len(group) >= MIN_BIN).groupby('bin')
span = np.array([x.min(), x.max()])

# (a) bin means with the line fitted to the individual politicians in each colour
for side, color, label in [('wp', WP, 'Wikipedia'), ('gp', GP, 'Grokipedia')]:
    means, cis = bins[f'{side}_lean'].mean(), 1.96 * bins[f'{side}_lean'].sem()
    lean_axis.errorbar(means.index, means, yerr=cis, fmt='o', color=color, ms=4, lw=1,
                       mfc='none' if side == 'wp' else color, mew=1.2)
    intercept, slope = fits[side]
    lean_axis.plot(span, intercept + slope * span, color=color, lw=1.2, zorder=1)
    results[f'fig2a_{side}_bins'] = {f'{b:.2f}': float(m) for b, m in means.items()}
lean_axis.text(-0.78, 0.34, f'Grokipedia\nslope {fits["gp"][1]:.2f}', color=GP, ha='left', va='center')
lean_axis.text(0.5, -0.12, f'Wikipedia\nslope {fits["wp"][1]:.2f}', color=WP, ha='left', va='center')
lean_axis.set(xlabel='Ideology (DW-NOMINATE)', ylabel='Mean lean')
lean_axis.set_title('(a)', loc='left')

# (b) bin means per party, the pooled linear fit, and the conservative Democrats one by one
for party, marker, fill in [('Democratic', 's', 'none'), ('Republican', 'o', DIFF)]:
    groups = scored[scored.us_party == party].groupby('bin').filter(lambda g: len(g) >= MIN_BIN).groupby('bin').lean_shift
    shift_axis.errorbar(groups.mean().index, groups.mean(), yerr=1.96 * groups.sem(), fmt=marker,
                        color=DIFF, mfc=fill, ms=4, lw=1, mew=1.2, zorder=3)
    results[f'fig2b_{party[:3].lower()}_bins'] = {f'{b:.2f}': float(m) for b, m in groups.mean().items()}
constant, steepness = shift_fit
shift_axis.plot(span, constant + steepness * span, color=DIFF, ls='--', lw=1.2, zorder=1)
shift_axis.plot(cross.ideology, cross.lean_shift, 's', color=DIFF, mfc='none', ms=2.8, mew=0.7, alpha=0.8,
                zorder=2)
NAMED = {'Zell Miller': (-16, 9, 'right'), 'Huey Long': (5, -3, 'left'), 'Richard Russell Jr.': (-6, 0, 'right')}
for _, row in cross[cross.name.isin(NAMED)].iterrows():
    dx, dy, align = NAMED[row['name']]
    shift_axis.annotate(row['name'].replace(' Jr.', ''), (row.ideology, row.lean_shift), xytext=(dx, dy),
                        textcoords='offset points', ha=align, va='center', fontsize=7.5, color=DIFF,
                        # two squares sit side by side, so a leader line says which is his
                        arrowprops={'arrowstyle': '-', 'color': DIFF, 'lw': 0.5, 'shrinkB': 2}
                        if row['name'] == 'Zell Miller' else None)
shift_axis.annotate(f'offset {constant:.3f}', (0, constant), xytext=(-0.78, 0.045), color=DIFF, ha='left',
                    va='center', arrowprops={'arrowstyle': '-', 'color': DIFF, 'lw': 0.7, 'shrinkB': 3})
shift_axis.text(0.6, 0.06, f'+{steepness:.3f}\nper unit', color=DIFF, ha='center', va='center')
shift_axis.plot(0, constant, 'o', color=DIFF, ms=3, zorder=4)
shift_axis.set(xlabel='Ideology (DW-NOMINATE)', ylabel='Lean shift,\nGrokipedia $-$ Wikipedia')
shift_axis.set_title('(b)', loc='left')
for axis in [lean_axis, shift_axis]:
    axis.axhline(0, color=REF, lw=0.8)
    axis.axvline(0, color=REF, lw=0.4, ls=':')
    # direction of the scale under the tick labels: x in data, y a fixed distance below the axes
    below = mtransforms.blended_transform_factory(
        axis.transData, mtransforms.offset_copy(axis.transAxes, fig=figure, y=-23, units='points'))
    for word, start, end, align in [('liberal', -0.08, -0.75, 'right'), ('conservative', 0.08, 0.95, 'left')]:
        axis.annotate(word, (end, 0), (start, 0), xycoords=below, textcoords=below, ha=align, va='center',
                      color=DIFF, annotation_clip=False,
                      arrowprops={'arrowstyle': '-|>', 'color': DIFF, 'lw': 0.7, 'mutation_scale': 6,
                                  'shrinkA': 2, 'shrinkB': 0})
    axis.xaxis.labelpad = 17  # room for the arrows
figure.savefig(out_dir / 'fig2.pdf')
figure.savefig(out_dir / 'fig2.png', dpi=300)
plt.close(figure)


# %% Checks against the reference results, then write
reference = json.loads((features_dir.parent / 'results' / 'results.json').read_text())
ideology_reference = json.loads((features_dir.parent / 'out' / 'ideology' / 'results.json').read_text())
expected = {key: (reference[key], results[key]) for key in [
    'stance_con_shift_rep', 'stance_lib_shift_rep', 'stance_con_shift_dem', 'stance_lib_shift_dem',
    'lean_shift_rep', 'lean_shift_dem', 'pct_moved_right_rep', 'pct_moved_right_dem']}
for key in ['slope_wp_all_pooled', 'slope_gp_all_pooled', 'shift_const_coef_all_pooled',
            'shift_ideology_coef_all_pooled', 'n_conservative_democrats', 'lean_shift_conservative_democrats']:
    expected[key] = (ideology_reference[key], results[key])
for key, (want, got) in expected.items():
    flag = 'ok' if abs(want - got) < 0.001 else 'MISMATCH'
    print(f'{key:40s} reference {want:10.4f} plotted {got:10.4f} {flag}')
print('November share moving right: Rep {:.1f}, Dem {:.1f}'.format(results['nov_pct_moved_right_rep'],
                                                                  results['nov_pct_moved_right_dem']))
(out_dir / 'results.json').write_text(json.dumps(results, indent=2, default=lambda value: value.item()) + '\n')
