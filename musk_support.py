"""Do Grokipedia articles lean further right in the races Elon Musk's America PAC spent on?

    python musk_support.py FEATURES_DIR NOV_DATA_DIR NOV_FEATURES_DIR EXT_DIR OUT_DIR

FEATURES_DIR, NOV_DATA_DIR and NOV_FEATURES_DIR are as for rescrape.py. EXT_DIR receives the public data this
script downloads, and anything already there is reused: FEC independent expenditures and
candidate master files of the 2024 and 2026 cycles, Voteview members of the 119th Congress,
and the FEC and bioguide ids of the politicians on Wikidata. America PAC only ever helped
Republicans, by backing them or attacking their Democratic opponents, so a race it spent on
marks its Republican as Musk-backed and its Democrat as Musk-opposed. Its 2026 money came
after the February scrape, so it marks Musk's preferences, not a cause of the February text.
Writes OUT_DIR/results.json and the figure musk_support as PDF.
"""
# %% Setup
import json
import re
import sys
import unicodedata
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
DIFF, REF = '#777777', 'black'  # GP vs WP quantities, reference lines
MIN_TREATED = 10  # fewer politicians in Musk races than this give no test
AMERICA_PAC = 'C00879510'  # FEC id of Musk's super PAC
CYCLES = [2024, 2026]
FEB_SCRAPE = pd.Timestamp('2026-02-13', tz='UTC')
CAMPAIGN = pd.Timestamp('2026-09-01', tz='UTC')  # Grokipedia edits resumed in September
CONTESTED = 1_000_000  # outside general-election spending in a race, America PAC left out
MUSK_MIN = 100_000  # America PAC money that marks a race, above the few thousands of stray ads
MAX_ITEM = 100_000_000  # single expenditures above this are junk filings (one claims 10 billion)
FEC_URL = 'https://www.fec.gov/files/bulk-downloads/{year}/{name}'
VOTEVIEW_URL = 'https://voteview.com/static/data/out/members/HS119_members.csv'
SPARQL = 'https://query.wikidata.org/sparql'
CN_COLUMNS = ['cand_id', 'cand_name', 'party', 'election_year', 'state', 'office', 'district', 'ici',
              'status', 'pcc', 'street1', 'street2', 'city', 'mail_state', 'zip']
PARTIES = {'Democratic': 'DEM', 'Republican': 'REP'}
SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'mr', 'mrs', 'ms', 'dr', 'hon'}
# first names that share no prefix with the formal name the FEC files may use
NICKNAMES = {'bill': 'william', 'will': 'william', 'bob': 'robert', 'rob': 'robert', 'mike': 'michael',
             'jim': 'james', 'jimmy': 'james', 'dick': 'richard', 'rick': 'richard', 'chuck': 'charles',
             'ted': 'edward', 'ed': 'edward', 'tony': 'anthony', 'andy': 'andrew', 'jack': 'john',
             'liz': 'elizabeth', 'beth': 'elizabeth', 'betsy': 'elizabeth', 'peggy': 'margaret',
             'kate': 'katherine', 'katie': 'katherine', 'kathy': 'katherine', 'jerry': 'gerald',
             'buddy': 'earl', 'hank': 'henry', 'harry': 'henry', 'ben': 'benjamin', 'zach': 'zachary',
             'nick': 'nicholas', 'greg': 'gregory', 'jeff': 'jeffrey', 'steve': 'stephen',
             'ken': 'kenneth', 'tom': 'thomas', 'tim': 'timothy', 'joe': 'joseph', 'dan': 'daniel',
             'dave': 'david', 'don': 'donald', 'ron': 'ronald', 'pat': 'patrick', 'sam': 'samuel',
             'chris': 'christopher', 'matt': 'matthew', 'gabe': 'gabriel', 'abe': 'abraham',
             'al': 'albert', 'max': 'maxwell', 'val': 'valerie', 'jen': 'jennifer', 'sue': 'susan'}


def add_lean(frame):
    """Lean is P(supports conservatives) minus P(supports liberals), shift is GP lean minus WP lean."""
    for side in SIDES:
        frame[f'{side}_lean'] = frame[f'{side}_stance_con'] - frame[f'{side}_stance_lib']
    frame['lean_shift'] = frame.gp_lean - frame.wp_lean
    return frame


def download(url, path):
    if not path.exists():
        response = get(url)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def name_tokens(name):
    name = ''.join(char for char in unicodedata.normalize('NFD', name) if unicodedata.category(char) != 'Mn')
    name = re.sub(r'[^a-z ]', ' ', name.lower().replace("'", ''))
    return [token for token in name.split() if token not in SUFFIXES]


def same_first_name(ours, theirs):
    """Equal, one a prefix of the other (Chris, Christopher) or a known nickname (Mike, Michael)."""
    if ours == theirs or NICKNAMES.get(ours) == theirs or NICKNAMES.get(theirs) == ours:
        return True
    return min(len(ours), len(theirs)) >= 3 and (ours.startswith(theirs) or theirs.startswith(ours))


def ols(frame, outcome, regressors):
    """OLS with heteroskedasticity-robust (HC1) standard errors and normal p-values."""
    frame = frame.dropna(subset=[outcome, *regressors])
    if frame[regressors[0]].nunique() < 2:  # the February panel has no Democrat in a Musk race
        return {'n': len(frame), 'coef': np.nan, 'se': np.nan, 'p': np.nan}
    regressors = [r for r in regressors if frame[r].nunique() > 1 or r == regressors[0]]
    X = np.column_stack([np.ones(len(frame))] + [frame[r].astype(float) for r in regressors])
    y = frame[outcome].to_numpy(float)
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residual = y - X @ beta
    covariance = xtx_inv @ (X.T * residual ** 2) @ X @ xtx_inv * len(y) / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(covariance))
    p = 2 * stats.norm.sf(np.abs(beta / se))
    return {'n': len(y), 'coef': beta[1], 'se': se[1], 'p': p[1]}


features_dir, nov_data_dir, nov_features_dir, ext_dir, out_dir = map(Path, sys.argv[1:6])
ext_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
results = {}


# %% Downloads
for year in CYCLES:
    for name in [f'independent_expenditure_{year}.csv', f'cn{year % 100}.zip']:
        download(FEC_URL.format(year=year, name=name), ext_dir / name)
download(VOTEVIEW_URL, ext_dir / 'HS119_members.csv')


# %% Politicians: November lean, February lean of the panel, Wikidata ids
people = pd.read_parquet(nov_data_dir / 'pairs_clean.parquet',
                         columns=['slug', 'wd_id', 'name', 'us_party', 'sources', 'panel', 'gp_date_modified'])
people = add_lean(people.merge(pd.read_parquet(nov_features_dir / 'features.parquet'), on='slug'))
feb = add_lean(pd.read_parquet(features_dir / 'features.parquet',
                               filters=[('slug', 'in', list(people.slug[people.panel]))]))
people = people.merge(feb[['slug', 'lean_shift', 'gp_lean']].rename(
    columns={'lean_shift': 'lean_shift_feb', 'gp_lean': 'gp_lean_feb'}), on='slug', how='left')
modified = pd.to_datetime(people.gp_date_modified, utc=True)
people['gp_edited'] = modified > FEB_SCRAPE
people['gp_edited_campaign'] = modified > CAMPAIGN

wikidata_file = ext_dir / 'wikidata_ids.csv'
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
wikidata = pd.read_csv(wikidata_file)
members = pd.read_csv(ext_dir / 'HS119_members.csv', usecols=['bioguide_id', 'state_abbrev', 'nominate_dim1'])
bioguide = wikidata.dropna(subset=['bioguide']).drop_duplicates('wd_id').set_index('wd_id').bioguide
people['nominate'] = people.wd_id.map(bioguide).map(members.groupby('bioguide_id').nominate_dim1.mean())
people['state'] = people.wd_id.map(bioguide).map(members.groupby('bioguide_id').state_abbrev.first())
people['member'] = people.nominate.notna()
people['office_2026'] = np.select([people.sources.str.contains('nominee_senate'),
                                   people.sources.str.contains('nominee_house')], ['S', 'H'], '')
people['nominee'] = people.office_2026 != ''
people['senate'] = people.office_2026 == 'S'


# %% FEC candidate ids per politician: Wikidata, then name and party in the candidate files
candidates = pd.concat([pd.read_csv(ext_dir / f'cn{year % 100}.zip', sep='|', names=CN_COLUMNS, dtype=str)
                        .assign(cycle=year) for year in CYCLES], ignore_index=True)
candidates = candidates[candidates.office.isin(['H', 'S']) & candidates.cand_name.notna()].copy()
candidates['party'] = candidates.party.replace({'DFL': 'DEM'})  # Minnesota Democrats
candidates['last'] = candidates.cand_name.str.split(',').str[0].map(name_tokens)
candidates['given'] = candidates.cand_name.str.split(',', n=1).str[1].fillna('').map(name_tokens)
candidates = candidates[candidates['last'].str.len() > 0]
candidates['last_word'] = candidates['last'].str[-1]
by_last_word = {word: group for word, group in candidates.groupby('last_word')}
running = candidates[(candidates.cycle == 2026) & (candidates.election_year == '2026')]


def name_matches(name, party):
    """Same last name and party, a compatible first name, and no clash of middle initials,
    which keeps John H. Sununu apart from his son John E."""
    words = name_tokens(name)
    hits = by_last_word.get(words[-1], candidates.iloc[:0])
    found = set()
    for row in hits.itertuples():
        if row.party != party or words[-len(row.last):] != row.last:
            continue
        if not any(same_first_name(words[0], given) for given in row.given):
            continue
        middle = words[1:len(words) - len(row.last)]
        if middle and len(row.given) > 1 and middle[0][0] not in {given[0] for given in row.given[1:]}:
            continue
        found.add(row.cand_id)
    return found


def nominee_fallback(person):
    """A nominee filed under another name (Jay Feely as Thomas, Ashley Hinson as Arenholz): the
    one candidate of the party and office with the same last name, or with the same first name
    in the member's state."""
    words = name_tokens(person.name)
    pool = running[(running.office == person.office_2026) & (running.party == PARTIES[person.us_party])]
    for found in [set(pool.cand_id[pool.last_word == words[-1]]),
                  set(pool.cand_id[(pool.state == person.state)
                                   & pool.given.map(lambda given: bool(given) and same_first_name(words[0], given[0]))])]:
        if len(found) == 1:
            return found
    return set()


# FEC ids carry the state in characters 3 and 4 (S4MI00595 is a Michigan Senate candidate)
wikidata_fec = wikidata.dropna(subset=['fec_id']).groupby('wd_id').fec_id.apply(set)
known_ids, matched_ids, n_ambiguous = {}, {}, 0
for person in people.itertuples():
    known = wikidata_fec.get(person.wd_id, set())
    matched = name_matches(person.name, PARTIES[person.us_party]) - known
    states = {cand[2:4] for cand in known} | ({person.state} if pd.notna(person.state) else set())
    if states:  # a namesake elsewhere is someone else
        matched = {cand for cand in matched if cand[2:4] in states}
    elif len({cand[2:4] for cand in matched}) > 1:
        n_ambiguous += 1
        matched = set()
    known_ids[person.slug], matched_ids[person.slug] = known, matched
# an id that the names give to two politicians, or that Wikidata gives to another, stays unassigned
claims = pd.Series([cand for ids in [*known_ids.values(), *matched_ids.values()] for cand in ids]).value_counts()
people['fec_ids'] = [known_ids[slug] | {cand for cand in matched_ids[slug] if claims[cand] == 1}
                     for slug in people.slug]
nominee_ids = {office: set(running.cand_id[running.office == office]) for office in ['H', 'S']}
unmatched = people.nominee & np.array([not (fec & nominee_ids.get(office, set()))
                                       for fec, office in zip(people.fec_ids, people.office_2026)])
people['fec_ids'] = [fec | nominee_fallback(person) if missing else fec
                     for fec, person, missing in zip(people.fec_ids, people.itertuples(), unmatched)]
results['n_nominees_fallback'] = int(unmatched.sum())
results['n_fec_linked'] = int((people.fec_ids.str.len() > 0).sum())
results['n_fec_ambiguous'] = n_ambiguous


# %% Races: America PAC and other outside spending per race, the race of each politician
spending = pd.concat([pd.read_csv(ext_dir / f'independent_expenditure_{year}.csv', dtype=str)
                      .assign(cycle=year) for year in CYCLES], ignore_index=True)
spending['amount'] = pd.to_numeric(spending.exp_amo, errors='coerce')
spending = spending[spending.can_office.isin(['H', 'S']) & spending.amount.between(0, MAX_ITEM)]
spending = spending.drop_duplicates(['cycle', 'spe_id', 'tran_id', 'file_num'])
spending['district'] = np.where(spending.can_office == 'S', '00', spending.can_office_dis.str.zfill(2))
spending['race'] = list(zip(spending.cycle, spending.can_office_state, spending.can_office, spending.district))
general = spending[spending.ele_type == 'G']
america = spending[spending.spe_id == AMERICA_PAC]
america_race = america[america.ele_type == 'G'].groupby('race').amount.sum()
america_direct = america.dropna(subset=['cand_id']).groupby('cand_id').amount.sum()
outside_race = general[general.spe_id != AMERICA_PAC].groupby('race').amount.sum()

# 2026 nominees run in the race of their 2026 filing. A 2024 filing counts only where outside
# groups named the candidate in the general election, which leaves out primary losers
candidates['district'] = np.where(candidates.office == 'S', '00', candidates.district.str.zfill(2))
candidates['race'] = list(zip(candidates.cycle, candidates.state, candidates.office, candidates.district))
filed = candidates[candidates.election_year == candidates.cycle.astype(str)].groupby('cand_id').race.apply(set)
named = general.dropna(subset=['cand_id']).groupby('cand_id').race.apply(set)


def races(fec, cycle, office=None):
    found = set()
    for cand in fec:
        mine = filed.get(cand, set())
        found |= mine if cycle == 2026 else mine & named.get(cand, set())
    return {race for race in found if race[0] == cycle and (office is None or race[2] == office)}


people['race_2026'] = [races(fec, 2026, office) if office else set()
                       for fec, office in zip(people.fec_ids, people.office_2026)]
people['race_2024'] = [races(fec, 2024) for fec in people.fec_ids]
people['america_2026'] = [sum(america_race.get(race, 0) for race in found) for found in people.race_2026]
people['america_2024'] = [sum(america_race.get(race, 0) for race in found) for found in people.race_2024]
people['america_direct'] = [sum(america_direct.get(cand, 0) for cand in fec) for fec in people.fec_ids]
people['outside_2026'] = [sum(outside_race.get(race, 0) for race in found) for found in people.race_2026]
people['musk_race_2024'] = people.america_2024 >= MUSK_MIN
people['musk_race_2026'] = (people.america_2026 >= MUSK_MIN) | (people.america_direct >= MUSK_MIN)
people['musk_race'] = people.musk_race_2024 | people.musk_race_2026
people['musk_any'] = (people[['america_2026', 'america_2024', 'america_direct']] > 0).any(axis=1)
people['contested'] = people.outside_2026 >= CONTESTED
people['log_outside'] = np.log10(1 + people.outside_2026)
results['n_nominees_without_race'] = int((people.nominee & (people.race_2026.str.len() == 0)).sum())


# %% Tests per party: Musk races against contested races without Musk money, and against the rest
people['bias_shift'] = people.gp_bias_score - people.wp_bias_score
people['america_log'] = np.log10(1 + np.maximum(people.america_2026 + people.america_2024, people.america_direct))
groups = {}
for short, party in [('rep', 'Republican'), ('dem', 'Democratic')]:
    frame = people[people.us_party == party]
    musk, rest = frame[frame.musk_race], frame[~frame.musk_race]
    contested = rest[rest.contested]
    groups[short] = {'Not contested': rest[~rest.contested], 'Contested, no Musk money': contested,
                     'Musk race': musk}
    results[f'n_musk_{short}'] = len(musk)
    results[f'n_musk_2024_{short}'] = int(frame.musk_race_2024.sum())
    results[f'n_contested_{short}'] = len(contested)
    for snapshot, column in [('nov', 'lean_shift'), ('feb', 'lean_shift_feb')]:
        results[f'{snapshot}_n_musk_{short}'] = int(musk[column].count())
        if musk[column].count() < MIN_TREATED:  # the February panel holds few of today's candidates
            continue
        for label, group in [('musk', musk), ('contested', contested), ('rest', rest)]:
            values = group[column].dropna()
            results[f'{snapshot}_lean_shift_{label}_{short}'] = values.mean()
            results[f'{snapshot}_lean_shift_ci_{label}_{short}'] = 1.96 * values.sem()
        a, b = musk[column].dropna(), contested[column].dropna()
        results[f'{snapshot}_musk_minus_contested_{short}'] = a.mean() - b.mean()
        results[f'{snapshot}_musk_minus_contested_ci_{short}'] = 1.96 * np.hypot(a.sem(), b.sem())
        results[f'{snapshot}_musk_vs_contested_mannwhitney_p_{short}'] = stats.mannwhitneyu(a, b).pvalue
        results[f'{snapshot}_musk_vs_rest_mannwhitney_p_{short}'] = stats.mannwhitneyu(
            a, rest[column].dropna()).pvalue

    # November regressions. Nominees: outside spending, chamber and incumbency held fixed.
    # Members: ideology too. Then any America PAC money, and its amount
    nominees = frame[frame.nominee]
    specifications = [('all', frame, ['musk_race'], ['lean_shift']),
                      ('nominees', nominees, ['musk_race', 'log_outside', 'senate', 'member'],
                       ['lean_shift', 'bias_shift', 'overlap_3gram']),
                      ('nominee_members', nominees[nominees.member],
                       ['musk_race', 'log_outside', 'senate', 'nominate'], ['lean_shift']),
                      ('nominees_any', nominees, ['musk_any', 'log_outside', 'senate', 'member'], ['lean_shift']),
                      ('nominees_dose', nominees, ['america_log', 'log_outside', 'senate', 'member'],
                       ['lean_shift'])]
    for label, sample, regressors, outcomes in specifications:
        for outcome in outcomes:
            for key, value in ols(sample, outcome, regressors).items():
                results[f'ols_{label}_{outcome}_{key}_{short}'] = value

    # campaign rewrites: did Grokipedia edit Musk races more often from September on?
    for label, group in [('musk', musk), ('contested', contested), ('rest', rest)]:
        results[f'pct_edited_campaign_{label}_{short}'] = group.gp_edited_campaign.mean() * 100
    table = [[group.gp_edited_campaign.sum(), (~group.gp_edited_campaign).sum()] for group in [musk, contested]]
    results[f'edited_campaign_fisher_p_{short}'] = stats.fisher_exact(table).pvalue

results_json = json.dumps(results, indent=2, default=lambda value: value.item())
(out_dir / 'results.json').write_text(results_json + '\n')


# %% Figure in the report's style: November lean shift per group and party, 95% CI
mpl.use('agg')  # no GUI backend, which would round figure sizes to whole pixels
sns.set_theme(style='whitegrid', context='paper', rc={
    'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.constrained_layout.use': True, 'pdf.fonttype': 42, 'grid.color': '0.9'})

figure, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 0.3 * TEXTWIDTH), sharex=True, sharey=True)
for axis, (short, title) in zip(axes, [('dem', 'Democrats'), ('rep', 'Republicans')]):
    for y, group in enumerate(groups[short].values()):
        mean, ci = group.lean_shift.mean(), 1.96 * group.lean_shift.sem()
        axis.errorbar(mean, y, xerr=ci, fmt='o', color=DIFF, ms=5, lw=1)
        axis.text(mean + ci + 0.01, y, f'{len(group)}', va='center', fontsize=8, color=DIFF)
    axis.axvline(0, color=REF, lw=0.8)
    axis.set(title=title, xlabel='Lean shift, GP $-$ WP', yticks=range(3),
             yticklabels=list(groups[short]), ylim=(2.5, -0.5), xlim=(-0.1, 0.27))
    axis.grid(axis='y', visible=False)
figure.savefig(out_dir / 'musk_support.pdf')
