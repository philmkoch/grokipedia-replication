"""Fig. 1 (a, b, c) for v7, with the full population of 98,013 biographies as a reference group.

    python fig1_reference.py DATA_DIR FEATURES_DIR NOV_DATA_DIR NOV_FEATURES_DIR OUT_DIR

Same selection and definitions as paper_figures.py (February panel of US politicians, lean =
con - lib, shift = GP - WP). Writes fig1_v7 (pdf, png); the symmetric-log x axis of the earlier
variant b is dropped. Also prints the
numbers for all 98K and for 98K without the US politicians in party.parquet.
"""
# %% Setup
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SIDES = ['wp', 'gp']
TEXTWIDTH = 418.26 / 72.27
WP, GP = '#004488', '#DDAA33'
DIFF, REF = '#777777', 'black'
NEUTRAL = 1e-6


def add_lean(frame):
    for side in SIDES:
        frame[f'{side}_lean'] = frame[f'{side}_stance_con'] - frame[f'{side}_stance_lib']
    frame['lean_shift'] = frame.gp_lean - frame.wp_lean
    frame['con_shift'] = frame.gp_stance_con - frame.wp_stance_con
    frame['lib_shift'] = frame.gp_stance_lib - frame.wp_stance_lib
    return frame


data_dir, features_dir, nov_data_dir, nov_features_dir, out_dir = map(Path, sys.argv[1:6])
mpl.use('agg')
sns.set_theme(style='whitegrid', context='paper', rc={
    'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.constrained_layout.use': True, 'pdf.fonttype': 42, 'grid.color': '0.9'})

# %% Data: panel as in paper_figures.py, plus every pair
nov = pd.read_parquet(nov_data_dir / 'pairs_clean.parquet', columns=['slug', 'us_party', 'panel'])
panel = nov[nov.panel]
everyone = pd.read_parquet(features_dir / 'features.parquet')
everyone = add_lean(everyone.merge(pd.read_parquet(data_dir / 'party.parquet', columns=['slug', 'us_party']),
                                   on='slug', how='left'))
feb = everyone[everyone.slug.isin(panel.slug)].drop(columns='us_party').merge(
    panel[['slug', 'us_party']], on='slug')
nonus = everyone[everyone.us_party.isna()]
print('n panel', len(feb), 'n all', len(everyone), 'n non-US-politician', len(nonus))

GROUPS = {'rep': feb[feb.us_party == 'Republican'], 'dem': feb[feb.us_party == 'Democratic'],
          'all': everyone, 'nonus': nonus}
stats = {}
for key, group in GROUPS.items():
    s = {'n': len(group), 'right': (group.lean_shift > 0).mean() * 100,
         'left': (group.lean_shift < 0).mean() * 100,
         'neutral': (group.lean_shift.abs() < NEUTRAL).mean() * 100,
         'exact0': (group.lean_shift == 0).mean() * 100,
         'net': group.lean_shift.mean(), 'net_ci': 1.96 * group.lean_shift.sem(),
         'median': group.lean_shift.median()}
    for name in ['con', 'lib']:
        s[name] = group[f'{name}_shift'].mean()
        s[f'{name}_ci'] = 1.96 * group[f'{name}_shift'].sem()
    stats[key] = s
    print(key, {k: round(v, 5) if isinstance(v, float) else v for k, v in s.items()})


# %% Figure
def layout_words(figure, axis, words, x0, x1, bold_phrase):
    """Greedy line wrap by measured width (copied from paper_figures.py)."""
    renderer = figure.canvas.get_renderer()
    width = axis.bbox.width

    def measure(text, bold):
        probe = axis.text(0, 0, text, fontfamily='cmb10' if bold else 'cmr10', fontsize=9)
        size = probe.get_window_extent(renderer).width / width
        probe.remove()
        return size

    space = 0.33 * 9 / 72 / (width / figure.dpi)
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


sentences = pd.read_parquet(data_dir / 'dewine_sentences.parquet')
dewine = {side: sentences[(sentences.side == side) & (sentences['index'] == i)].iloc[0]
          for side, i in [('wp', 76), ('gp', 2)]}

def draw(variant):
    figure = plt.figure(figsize=(TEXTWIDTH, 0.8 * TEXTWIDTH), dpi=300)
    top, bottom = figure.subfigures(2, 1, height_ratios=[1, 2.35], hspace=0)
    text_axis = top.add_subplot()
    curve_axis, layer_axis = bottom.subplots(1, 2, width_ratios=[1, 1.2])

    # (a) the sentence pair as text, probabilities aligned right
    text_axis.set_axis_off()
    text_axis.set_title('(a)', loc='left')
    figure.canvas.draw()
    TEXT_START, TEXT_END = 0.14, 0.845
    LINE = 1.5 * 9 / 72 / (text_axis.bbox.height / figure.dpi)
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

    # (c) change in mean P(con) filled and P(lib) hollow, net shift as bracket
    ROWS = {'rep': 0, 'dem': 2, 'all': 4}
    for short, y in ROWS.items():
        con, lib = stats[short]['con'], stats[short]['lib']
        for name, mean, row in [('con', con, y), ('lib', lib, y + 0.7)]:
            layer_axis.plot(mean, row, 'o', color=DIFF, mfc=DIFF if name == 'con' else 'white', mew=1.4, ms=6)
            word = 'pro-conservative' if name == 'con' else 'pro-liberal'
            left = mean == max(con, lib) and short == 'rep'
            layer_axis.annotate(word, (mean, row), xytext=(-9 if left else 9, 0), textcoords='offset points',
                                ha='right' if left else 'left', va='center', color=DIFF)
        middle = y + 0.35
        layer_axis.annotate('', (con, middle), (lib, middle), arrowprops=dict(
            arrowstyle='|-|', color=REF, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=3))
        layer_axis.annotate(f'net {stats[short]["net"]:+.3f}', (max(con, lib), middle), xytext=(6, 0),
                            textcoords='offset points', ha='left', va='center', color=REF)
    layer_axis.axhline(3.35, color='0.8', lw=0.8, ls=':')
    layer_axis.set(yticks=[0.35, 2.35, 4.35], yticklabels=['Republicans', 'Democrats', 'All\nbiographies'],
                   ylim=(5.3, -0.6), xlim=(-0.015, 0.19),
                   xlabel='Change in mean probability,\nGrokipedia $-$ Wikipedia')
    layer_axis.set_xticks([0, 0.05, 0.1, 0.15])
    layer_axis.tick_params(axis='y', length=0)
    layer_axis.grid(axis='y', visible=False)
    layer_axis.axvline(0, color=REF, lw=0.8)
    layer_axis.set_title('(c)', loc='left')

    # (b) share of biographies with a shift at least x
    for short in ['rep', 'dem', 'all']:
        shifts = np.sort(GROUPS[short].lean_shift.to_numpy())
        share = np.linspace(100, 0, len(shifts), endpoint=False)
        style = {'rep': ('-', 1.2, DIFF), 'dem': ('--', 1.2, DIFF), 'all': (':', 1.2, REF)}[short]
        curve_axis.step(shifts, share, where='post', color=style[2], ls=style[0], lw=style[1])
        if short == 'all':
            continue  # the reference curve gets no marker
        value = stats[short]['right']
        curve_axis.plot(0, value, 'o', color=style[2], ms=4)
        offset, ha = {'rep': ((5, 3), 'left'), 'dem': ((-5, -4), 'right'), 'all': ((5, -5), 'left')}[short]
        curve_axis.annotate(f'{value:.0f}%', (0, value), xytext=offset, textcoords='offset points',
                            ha=ha, va='center', color=style[2])
    if variant == 'a':
        curve_axis.text(0.15, 80, 'Republicans', color=DIFF, ha='left', va='center')
        curve_axis.text(-0.02, 10, 'Democrats', color=DIFF, ha='right', va='center')
        curve_axis.text(0.02, 20, 'All', color=REF, ha='left', va='center')
        curve_axis.set(xlim=(-0.35, 0.5), xticks=[-0.25, 0, 0.25, 0.5])
    else:
        curve_axis.set_xscale('symlog', linthresh=1e-3, linscale=0.5)
        curve_axis.set(xlim=(-0.5, 0.5), xticks=[-0.1, -0.01, 0, 0.01, 0.1],
                       xticklabels=['$-$0.1', '$-$0.01', '0', '0.01', '0.1'])
        curve_axis.text(0.03, 85, 'Republicans', color=DIFF, ha='left', va='center')
        curve_axis.text(-0.003, 12, 'Democrats', color=DIFF, ha='right', va='center')
        curve_axis.text(0.012, 12, 'All', color=REF, ha='left', va='center')
    curve_axis.set(xlabel='Lean shift, Grokipedia $-$ Wikipedia', ylabel='Share of biographies (%)',
                   ylim=(0, 100), yticks=range(0, 101, 25))
    curve_axis.axvline(0, color=REF, lw=0.8)
    curve_axis.set_title('(b)', loc='left')
    name = 'fig1_v7'
    figure.savefig(out_dir / f'{name}.pdf')
    figure.savefig(out_dir / f'{name}.png', dpi=300)
    plt.close(figure)


out_dir.mkdir(parents=True, exist_ok=True)
for variant in ['a']:
    draw(variant)
