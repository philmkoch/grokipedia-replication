"""The numbers and figures of the report on the February corpus, from the per-pair features.

    python results.py DATA_DIR FEATURES_DIR OUT_DIR

Reads DATA_DIR (pairs, ethnicity, party, ethnicity validation, the sentence scores of the
Mike DeWine pair, the Gentzkow-Shapiro-Taddy partisan phrase lists) and FEATURES_DIR/features.parquet from features.py. Writes
OUT_DIR/results.json and the report's figures rq1_structural, rq3_party and
rq4_bias_by_group as PDFs.
"""
# %% Setup
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import country_converter as coco
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from nltk.stem.snowball import SnowballStemmer
from scipy import stats

SIDES = ['wp', 'gp']
MIN_GROUP = 30  # smallest subgroup that enters a Kruskal-Wallis test or the group figure
ERAS = ['Pre-1500', '1500-1799', '1800-1899', '1900-1949', '1950-1979', '1980+']
ERA_STARTS = [-np.inf, 1500, 1800, 1900, 1950, 1980, np.inf]
NEUTRAL = 1e-6  # |lean| below this counts as no lean
PARTISAN = 0.1  # |lean| above this counts as clearly partisan
TEXTWIDTH = 418.26 / 72.27  # text width of the report in inches, so figure text prints at 9 pt
WP, GP = '#004488', '#DDAA33'  # Tol high-contrast blue and yellow, apart in greyscale too
DIFF, REF = '#777777', 'black'  # GP vs WP quantities, reference lines


def rank_biserial(difference):
    """Matched-pairs rank-biserial correlation, positive when GP is larger. Zeros are dropped."""
    difference = difference[difference != 0]
    return np.average(np.sign(difference), weights=difference.abs().rank())


def eta(frame, group, metric):
    """Kruskal-Wallis eta squared of `metric` across the groups of `group`. The denominator is
    n - 1 where Tomczak & Tomczak (2014) have n - k, below 1e-4 apart at this n."""
    samples = [values for _, values in frame.groupby(group)[metric] if len(values) >= MIN_GROUP]
    h = stats.kruskal(*samples).statistic
    n = sum(len(sample) for sample in samples)
    return (h - len(samples) + 1) / (n - 1)


def percent(condition):
    return condition.mean() * 100


def key_name(label):
    return re.sub(r'[ &]+', '_', label.lower())


data_dir, features_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
results = {}


# %% Join metadata and features, derive subgroups and GP minus WP differences
columns = ['slug', 'domain', 'gender', 'birthyear', 'bplace_country',
           'wp_word_count', 'gp_word_count', 'wp_section_count']
pairs = pd.read_parquet(data_dir / 'pairs_clean.parquet', columns=columns)
pairs = pairs.merge(pd.read_parquet(features_dir / 'features.parquet'), on='slug', how='left')
ethnicity = pd.read_parquet(data_dir / 'ethnicity.parquet').drop_duplicates('slug')
pairs = pairs.merge(ethnicity, on='slug', how='left')

continents = coco.CountryConverter().pandas_convert(pairs.bplace_country, to='Continent_7')
pairs['continent'] = continents.replace('not found', np.nan)
eras = pd.cut(pairs.birthyear, ERA_STARTS, labels=ERAS, right=False)
pairs['birth_era'] = eras.astype(object).fillna('Unknown')
for metric in ['bias_score', 'bias_frac', 'bias_score_lead', 'stance_con', 'stance_lib']:
    pairs[f'{metric}_diff'] = pairs[f'gp_{metric}'] - pairs[f'wp_{metric}']
# lean is positive for conservative, negative for liberal
for side in SIDES:
    pairs[f'{side}_lean'] = pairs[f'{side}_stance_con'] - pairs[f'{side}_stance_lib']
pairs['lean_shift'] = pairs.gp_lean - pairs.wp_lean
results['n_pairs'] = len(pairs)


# %% Corpus composition and the ethnicity label validation
for group in ['domain', 'ethnicity']:
    for label, count in pairs[group].value_counts().items():
        results[f'n_{group}_{key_name(label)}'] = count
        if label != 'Other':
            results[f'pct_{group}_{key_name(label)}'] = count / len(pairs) * 100

# GPT-5 mini labels against GPT-5 labels, plus a hand-checked sample
validation = pd.read_parquet(data_dir / 'ethnicity_cross_validation.parquet')
validation = validation[validation.source != 'hand20'].copy()
validation['agree'] = validation.gpt5mini_label == validation['gpt-5_label']
agreement = validation.groupby('gpt5mini_label').agree.mean() * 100
hand = pd.read_csv(data_dir / 'ethnicity_validation.csv')
results['eth_val_n'] = len(validation)
results['eth_val_agree_pct'] = percent(validation.agree)
chance = (validation.gpt5mini_label.value_counts(normalize=True)
          * validation['gpt-5_label'].value_counts(normalize=True)).sum()
results['eth_val_kappa'] = (validation.agree.mean() - chance) / (1 - chance)
results['eth_val_hand_n'] = len(hand)
results['eth_val_hand_correct'] = hand.correct.sum()
results['eth_val_agree_min_white_asian_black_hispanic_pct'] = (
    agreement[['White', 'Asian', 'Black', 'Hispanic']].min())
results['eth_val_agree_mena_pct'] = agreement['MENA']
results['eth_val_agree_indigenous_pct'] = agreement['Indigenous']


# %% RQ1: article length and sections
for side in SIDES:
    results[f'median_words_{side}'] = pairs[f'{side}_word_count'].median()
    results[f'median_sections_{side}'] = pairs[f'{side}_section_count'].median()
results['median_length_ratio'] = results['median_words_gp'] / results['median_words_wp']
results['pct_gp_longer'] = percent(pairs.gp_word_count > pairs.wp_word_count)
results['words_rank_biserial'] = rank_biserial(pairs.gp_word_count - pairs.wp_word_count)
results['sections_rank_biserial'] = rank_biserial(pairs.gp_section_count - pairs.wp_section_count)


# %% RQ2: verbatim overlap
for name, column in [('jaccard3', 'jaccard_3gram'), ('overlap3', 'overlap_3gram')]:
    results[f'{name}_mean'] = pairs[column].mean()
    results[f'{name}_median'] = pairs[column].median()
    results[f'{name}_p90'] = pairs[column].quantile(0.9)
results['n_overlap3_gt_05'] = (pairs.overlap_3gram > 0.5).sum()
results['pct_overlap3_gt_05'] = percent(pairs.overlap_3gram > 0.5)
results['n_jaccard3_gt_01'] = (pairs.jaccard_3gram > 0.1).sum()
top = pairs.loc[pairs.jaccard_3gram.idxmax()]
results['top_jaccard3_slug'] = top.slug
results['top_jaccard3'] = top.jaccard_3gram


# %% RQ3: bias scores
for side in SIDES:
    results[f'bias_mean_{side}'] = pairs[f'{side}_bias_score'].mean()
    results[f'bias_frac_mean_{side}_pct'] = pairs[f'{side}_bias_frac'].mean() * 100
for suffix, metric in [('', 'bias_score'), ('_lead', 'bias_score_lead')]:
    difference = pairs[f'{metric}_diff']
    results[f'bias{suffix}_mean_diff'] = difference.mean()
    results[f'bias{suffix}_rank_biserial'] = rank_biserial(difference)
    results[f'bias{suffix}_wilcoxon_p'] = stats.wilcoxon(difference).pvalue
    results[f'pct_gp_more_biased{suffix}'] = percent(difference > 0)


# %% RQ3: stance and lean across all articles
for side in SIDES:
    lean = pairs[f'{side}_lean']
    results[f'lean_mean_{side}'] = lean.mean()
    results[f'abs_lean_mean_{side}'] = lean.abs().mean()
    results[f'pct_zero_lean_{side}'] = percent(lean.abs() < NEUTRAL)
    results[f'n_partisan_left_{side}'] = (lean < -PARTISAN).sum()
    results[f'n_partisan_right_{side}'] = (lean > PARTISAN).sum()
for direction in ['left', 'right']:
    results[f'partisan_{direction}_growth'] = (results[f'n_partisan_{direction}_gp']
                                               / results[f'n_partisan_{direction}_wp'])
stance_means = pairs[['wp_stance_con', 'wp_stance_lib', 'gp_stance_con', 'gp_stance_lib']].mean()
results['stance_prob_means_min'] = stance_means.min()
results['stance_prob_means_max'] = stance_means.max()
results['pct_con_up'] = percent(pairs.stance_con_diff > 0)
results['pct_lib_up'] = percent(pairs.stance_lib_diff > 0)

# what happens to WP articles without a lean, and to those with one
wp_neutral = pairs.wp_lean.abs() < NEUTRAL
gp_left = pairs.gp_lean < -NEUTRAL
gp_right = pairs.gp_lean > NEUTRAL
gp_leaning = gp_left | gp_right
same_side = pairs.wp_lean * pairs.gp_lean > 0
results['pct_neutral_gain_lean'] = percent(gp_leaning[wp_neutral])
results['pct_neutral_gain_left'] = percent(gp_left[wp_neutral])
results['pct_neutral_gain_right'] = percent(gp_right[wp_neutral])
results['pct_leaning_keep'] = percent((same_side & gp_leaning)[~wp_neutral])
results['pct_leaning_flip'] = percent((~same_side & gp_leaning)[~wp_neutral])
results['pct_leaning_neutral'] = percent((~gp_leaning)[~wp_neutral])
wp_leaning = pairs[~wp_neutral]
results['keep_lean_ratio'] = wp_leaning.gp_lean.abs().mean() / wp_leaning.wp_lean.abs().mean()


# %% RQ3: lean of US politicians by party
party = pd.read_parquet(data_dir / 'party.parquet')
politicians = party[['slug', 'us_party']].dropna().merge(pairs, on='slug')
democrats = politicians[politicians.us_party == 'Democratic']
republicans = politicians[politicians.us_party == 'Republican']
results['n_politicians'] = party.slug.isin(pairs.slug).sum()
results['n_us_politicians'] = len(politicians)
results['n_democrats'] = len(democrats)
results['n_republicans'] = len(republicans)
results['lean_mean_gp_rep'] = republicans.gp_lean.mean()

# sign makes a lean towards the party's own side positive
for short, group, sign, direction in [('dem', democrats, -1, 'left'), ('rep', republicans, 1, 'right')]:
    results[f'pct_own_side_wp_{short}'] = percent(group.wp_lean * sign > 0)
    results[f'lean_mean_wp_{short}'] = group.wp_lean.mean()
    results[f'lean_shift_{short}'] = group.lean_shift.mean()
    results[f'lean_shift_ci_{short}'] = 1.96 * group.lean_shift.sem()
    results[f'pct_moved_right_{short}'] = percent(group.lean_shift > 0)
    # the shift split into its two parts, support for conservatives and support for liberals
    for camp in ['con', 'lib']:
        results[f'stance_{camp}_shift_{short}'] = group[f'stance_{camp}_diff'].mean()
    for side in SIDES:
        results[f'pct_clear_{direction}_{side}_{short}'] = percent(group[f'{side}_lean'] * sign > PARTISAN)
results['party_shift_mannwhitney_p'] = stats.mannwhitneyu(democrats.lean_shift,
                                                          republicans.lean_shift).pvalue


# %% RQ3: the running example, from the sentence scores of the Mike DeWine pair
sentences = pd.read_parquet(data_dir / 'dewine_sentences.parquet')
dewine = politicians[politicians.slug == 'Mike_DeWine'].iloc[0]
# position of sentence (1a) in the Wikipedia article and of (1b) in the Grokipedia article
for side, example in [('wp', 76), ('gp', 2)]:
    article = sentences[sentences.side == side].set_index('index')
    results[f'dewine_sentences_{side}'] = len(article)
    results[f'dewine_con_sentences_{side}'] = (article.p_con > 0.5).sum()
    results[f'dewine_example_con_{side}'] = article.p_con[example]
    results[f'dewine_lean_{side}'] = dewine[f'{side}_lean']


# %% RQ3: partisan phrases of Gentzkow, Shapiro & Taddy (2019) in the US politicians' articles
# Their lists hold the most partisan stemmed bigrams per Congress, positive = Republican. The main
# list averages Congresses 104 to 114 (1995-2016), the robustness list is the 114th alone.
phrase_dir = data_dir / 'gentzkow_shapiro'
congresses = [pd.read_csv(phrase_dir / f'partisan_phrases_{number}.txt', sep='|') for number in range(104, 115)]
phrase_lists = {'gst': pd.concat(congresses).groupby('phrase').partisanship.mean(),
                'gst114': congresses[-1].set_index('phrase').partisanship}
stopwords = set((phrase_dir / 'snowball_stopwords.txt').read_text().split())
stem = lru_cache(maxsize=None)(SnowballStemmer('english').stem)


def bigrams(text):
    """Bigram counts preprocessed as in the GST codebook, never across a comma or sentence end."""
    text = re.sub("['\u2019\\-\u2010\u2011]", '', text.lower())
    counts = Counter()
    for segment in re.split(r'[.,;:!?\n]', text):
        tokens = [stem(token) for token in re.findall('[a-z0-9]+', segment) if token not in stopwords]
        counts.update(f'{first} {second}' for first, second in zip(tokens, tokens[1:]))
    return counts


def republican_share(counts, phrases):
    """Share of matched phrase uses that are Republican minus the share that are Democratic."""
    matched = {gram: count for gram, count in counts.items() if gram in phrases.index}
    total = sum(matched.values())
    return sum(np.sign(phrases[gram]) * count for gram, count in matched.items()) / total if total else np.nan


texts = pd.read_parquet(data_dir / 'pairs_clean.parquet', columns=['slug', 'wp_text', 'gp_text'],
                        filters=[('slug', 'in', list(politicians.slug))])
politicians = politicians.merge(texts, on='slug')
for side in SIDES:
    politicians[f'{side}_bigrams'] = politicians[f'{side}_text'].map(bigrams)
    for name, phrases in phrase_lists.items():
        politicians[f'{side}_{name}'] = politicians[f'{side}_bigrams'].map(
            lambda counts: republican_share(counts, phrases))

for short, party_name in [('dem', 'Democratic'), ('rep', 'Republican')]:
    group = politicians[politicians.us_party == party_name]
    for name in phrase_lists:
        shift = (group[f'gp_{name}'] - group[f'wp_{name}']).dropna()
        results[f'{name}_shift_{short}'] = shift.mean()
        results[f'{name}_pct_right_{short}'] = percent(shift > 0)
        results[f'{name}_wilcoxon_p_{short}'] = stats.wilcoxon(shift).pvalue

# how much more often Grokipedia uses the phrases behind the Republican shift, per word
republican_articles = politicians[politicians.us_party == 'Republican']
growth = []
for phrase in ['fiscal restraint', 'limit govern', 'tax relief']:
    rates = [republican_articles[f'{side}_bigrams'].map(lambda counts: counts[phrase]).sum()
             / republican_articles[f'{side}_text'].str.split().str.len().sum() for side in SIDES]
    growth.append(rates[1] / rates[0])
results['gst_driver_growth_min'] = min(growth)
results['gst_driver_growth_max'] = max(growth)


# %% RQ4: Kruskal-Wallis effect sizes and selected group means
groupings = {'domain': 'domain', 'era': 'birth_era', 'gender': 'gender',
             'continent': 'continent', 'ethnicity': 'ethnicity'}
for name, column in groupings.items():
    results[f'eta_bias_{name}'] = eta(pairs, column, 'bias_score_diff')
results['eta_bias_domain_over_ethnicity'] = results['eta_bias_domain'] / results['eta_bias_ethnicity']
for name in ['domain', 'era']:
    results[f'eta_stance_con_{name}_pct'] = eta(pairs, groupings[name], 'stance_con_diff') * 100


def group_mean(column, label, metric):
    return pairs.loc[pairs[column] == label, metric].mean()


for name, column, label in [('humanities', 'domain', 'Humanities'), ('sports', 'domain', 'Sports'),
                            ('pre1500', 'birth_era', 'Pre-1500'), ('1980plus', 'birth_era', '1980+')]:
    results[f'bias_diff_{name}'] = group_mean(column, label, 'bias_score_diff')
results['bias_frac_diff_sports'] = group_mean('domain', 'Sports', 'bias_frac_diff')


# numpy scalars become plain JSON numbers
results_json = json.dumps(results, indent=2, default=lambda value: value.item())
(out_dir / 'results.json').write_text(results_json + '\n')


# %% Figures in the report's style, serif 9 pt at the report's text width
mpl.use('agg')  # no GUI backend, which would round figure sizes to whole pixels
sns.set_theme(style='whitegrid', context='paper', rc={
    'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.constrained_layout.use': True, 'pdf.fonttype': 42, 'grid.color': '0.9'})


def save(figure, name):
    figure.savefig(out_dir / f'{name}.pdf')
    plt.close(figure)


# %% fig:rq1, word counts per platform and the GP/WP length ratio per domain
figure, (counts_axis, ratio_axis) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 0.5 * TEXTWIDTH),
                                                 width_ratios=[1.1, 1])
word_counts = pairs[['wp_word_count', 'gp_word_count']]
bins = np.geomspace(word_counts.min().min(), word_counts.max().max(), 70)
top = 0
for side, label, color in [('wp', 'Wikipedia', WP), ('gp', 'Grokipedia', GP)]:
    values = pairs[f'{side}_word_count']
    counts, _, _ = counts_axis.hist(values, bins=bins, color=color, alpha=0.8,
                                    label=f'{label} (median {values.median():,.0f})')
    top = max(top, counts.max())
for side, color in [('wp', WP), ('gp', GP)]:
    counts_axis.vlines(pairs[f'{side}_word_count'].median(), 0, top * 1.05, color=color, ls='--', lw=1.2)
counts_axis.set(xscale='log', ylim=(0, top * 1.4), xlabel='Word count', ylabel='Articles')
counts_axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
counts_axis.legend(loc='upper right', frameon=False)

# 'Other' is outside the eight Pantheon domains
domain_pairs = pairs[pairs.domain != 'Other'].assign(
    word_count_ratio=lambda frame: frame.gp_word_count / frame.wp_word_count)
order = domain_pairs.groupby('domain').word_count_ratio.median().sort_values(ascending=False).index
sns.boxplot(data=domain_pairs, y='domain', x='word_count_ratio', order=order, ax=ratio_axis,
            color=DIFF, showfliers=False, width=0.6, linewidth=0.8)
ratio_axis.set(xscale='log', xlabel='GP / WP word count ratio', ylabel='')
ratio_axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
ratio_axis.set_xticks([0.1, 0.3, 1, 3, 10])
ratio_axis.minorticks_off()
ratio_axis.axvline(1, color=REF, ls='--', lw=1)
save(figure, 'rq1_structural')


# %% fig:party, (a) mean lean per party with 95% CI, (b) share of politicians above each shift
figure, (mean_axis, shift_axis) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 0.30 * TEXTWIDTH),
                                               width_ratios=[1.1, 1])
party_names = ['Democratic', 'Republican']
by_party = politicians.groupby('us_party')
means = by_party[['wp_lean', 'gp_lean']].mean().reindex(party_names)
mean_axis.hlines(range(2), means.wp_lean, means.gp_lean, color=DIFF, lw=2, zorder=1)
# WP hollow on top of GP filled, so both stay visible where a party does not move
for side, label, color in [('wp', 'Wikipedia', WP), ('gp', 'Grokipedia', GP)]:
    errors = 1.96 * by_party[f'{side}_lean'].sem().reindex(party_names)
    mean_axis.errorbar(means[f'{side}_lean'], range(2), xerr=errors, fmt='o', color=color,
                       mfc='none' if side == 'wp' else color, mew=1.5, ms=6, lw=1, capsize=0,
                       zorder=3 if side == 'wp' else 2)
    # direct labels above the wider Republican row
    mean_axis.text(means.loc['Republican', f'{side}_lean'], 0.7, label, color=color,
                   ha='center', va='bottom')
mean_axis.set(yticks=range(2), yticklabels=party_names, ylim=(1.6, -0.6), xlim=(-0.13, 0.23),
              xlabel='Mean lean score (right = conservative)')
mean_axis.grid(axis='y', visible=False)
mean_axis.set_title('(a)', loc='left')

# share at each shift counts the article at that shift itself, so the curve ends at 1/n
for party_name, style in [('Republican', '-'), ('Democratic', '--')]:
    shifts = np.sort(politicians.loc[politicians.us_party == party_name, 'lean_shift'])
    shift_axis.step(shifts, np.linspace(100, 0, len(shifts), endpoint=False), where='post',
                    color=DIFF, ls=style, lw=1.2, label=party_name)
shift_axis.set(xlabel='Lean shift, GP $-$ WP', ylabel='Share of articles (%)', xlim=(-0.4, 0.8),
               ylim=(0, 100))
shift_axis.legend(loc='upper right', frameon=False)
shift_axis.set_title('(b)', loc='left')
for axis in [mean_axis, shift_axis]:
    axis.axvline(0, color=REF, lw=0.8)
save(figure, 'rq3_party')


# %% fig:groups, mean bias difference with 95% CI per subgroup, one block per grouping
rows = []  # label, mean, CI, whether the row is a grouping header
for short, column, title in [('domain', 'domain', 'Domain'), ('era', 'birth_era', 'Birth era'),
                             ('continent', 'continent', 'Continent'), ('gender', 'gender', 'Gender')]:
    by_group = pairs.groupby(column).bias_score_diff.agg(['mean', 'sem', 'size'])
    by_group = by_group.drop(index=['Other', 'Unknown'], errors='ignore')
    by_group = by_group[by_group['size'] >= MIN_GROUP]
    by_group = by_group.reindex(ERAS) if short == 'era' else by_group.sort_values('mean', ascending=False)
    rows.append((f"{title} ($\\eta^2_H$ = {results[f'eta_bias_{short}']:.2f})", np.nan, np.nan, True))
    names = {'F': 'Female', 'M': 'Male', 'Science & Technology': 'Science & Tech.'}
    rows += [(names.get(label, label), row['mean'], 1.96 * row['sem'], False)
             for label, row in by_group.iterrows()]
    rows.append(('', np.nan, np.nan, False))
rows.pop()

figure, axis = plt.subplots(figsize=(TEXTWIDTH, 0.62 * TEXTWIDTH))
header_position = mpl.transforms.blended_transform_factory(axis.transAxes, axis.transData)
for y, (label, value, ci, header) in enumerate(rows):
    if header:
        axis.text(-0.012, y, label, ha='right', va='center', fontweight='bold',
                  transform=header_position)
    elif label:
        axis.errorbar(value, y, xerr=ci, fmt='o', color=DIFF, ms=4, lw=1, capsize=0)
ticks = [(y, row[0]) for y, row in enumerate(rows) if row[0] and not row[3]]
axis.set_yticks([y for y, _ in ticks], [label for _, label in ticks])
axis.set(ylim=(len(rows) - 0.5, -0.5), xlim=(0, 0.1), xlabel='Mean bias score difference, GP $-$ WP')
axis.grid(axis='y', visible=False)
axis.axvline(pairs.bias_score_diff.mean(), color=REF, lw=0.8, ls='--')
save(figure, 'rq4_bias_by_group')
