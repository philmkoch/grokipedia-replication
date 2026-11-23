"""Collects the February corpus: the people of Pantheon 2.0, their Wikipedia and Grokipedia
articles, the cleaned article pairs and the party labels of the politicians.

    python collect.py RAW_DIR DATA_DIR

RAW_DIR receives the Pantheon file, the Grokipedia sitemap and one JSON file per article.
Nothing that is already there is fetched again, so an interrupted scrape resumes and a
complete RAW_DIR only rebuilds the pairs. DATA_DIR holds the two review files
(pair_reviews.csv, pair_mismatches.csv) and receives pairs_clean.parquet, party.parquet and
funnel.json, the number of pairs left after each step.

A rerun today collects different texts than the report's corpus, because both sites keep
editing their articles.
"""
# %% Setup
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scraping import (file_name, get, grokipedia_article, markdown_to_text, party_labels,
                      section_count, strip_boilerplate, wikipedia_article, wikipedia_client)

PANTHEON = 'https://storage.googleapis.com/pantheon-public-data/person_2025_update.csv.bz2'
SITEMAP = 'https://assets.grokipedia.com/sitemap/sitemap-index.xml'
MIN_WORDS = {'wp': 50, 'gp': 100}  # an article with at most this many words is a stub
# Pantheon's eight domains and their occupations, https://pantheon.world/profile/occupation/
DOMAINS = {
    'Institutions': [
        'POLITICIAN', 'MILITARY PERSONNEL', 'NOBLEMAN', 'DIPLOMAT', 'JUDGE', 'PILOT',
        'PUBLIC WORKER', 'RELIGIOUS FIGURE'
    ],
    'Science & Technology': [
        'PHYSICIST', 'CHEMIST', 'BIOLOGIST', 'MATHEMATICIAN', 'ASTRONOMER', 'INVENTOR',
        'COMPUTER SCIENTIST', 'PHYSICIAN', 'ENGINEER', 'PSYCHOLOGIST', 'ECONOMIST', 'GEOGRAPHER',
        'GEOLOGIST', 'ANTHROPOLOGIST', 'ARCHAEOLOGIST', 'POLITICAL SCIENTIST', 'SOCIOLOGIST',
        'STATISTICIAN'
    ],
    'Humanities': [
        'PHILOSOPHER', 'HISTORIAN', 'WRITER', 'JOURNALIST', 'LINGUIST', 'CRITIC'
    ],
    'Arts': [
        'ACTOR', 'FILM DIRECTOR', 'SINGER', 'MUSICIAN', 'COMPOSER', 'PAINTER', 'SCULPTOR',
        'ARCHITECT', 'PHOTOGRAPHER', 'DESIGNER', 'COMEDIAN', 'COMIC ARTIST', 'CONDUCTOR', 'DANCER',
        'FASHION DESIGNER', 'GAME DESIGNER', 'ARTIST'
    ],
    'Sports': [
        'SOCCER PLAYER', 'BASKETBALL PLAYER', 'TENNIS PLAYER', 'CYCLIST', 'ATHLETE', 'SWIMMER',
        'BOXER', 'RACING DRIVER', 'CHESS PLAYER', 'AMERICAN FOOTBALL PLAYER', 'BADMINTON PLAYER',
        'BASEBALL PLAYER', 'BULLFIGHTER', 'COACH', 'CRICKETER', 'FENCER', 'GAMER', 'GO PLAYER',
        'GOLFER', 'GYMNAST', 'HANDBALL PLAYER', 'HOCKEY PLAYER', 'MARTIAL ARTS', 'MOUNTAINEER',
        'POKER PLAYER', 'REFEREE', 'RUGBY PLAYER', 'SKATER', 'SKIER', 'SNOOKER',
        'TABLE TENNIS PLAYER', 'VOLLEYBALL PLAYER', 'WRESTLER'
    ],
    'Public Figure': [
        'CELEBRITY', 'COMPANION', 'EXTREMIST', 'INSPIRATION', 'MAFIOSO', 'MAGICIAN', 'MODEL',
        'OCCULTIST', 'PIRATE', 'PORNOGRAPHIC ACTOR', 'PRESENTER', 'SOCIAL ACTIVIST', 'CHEF',
        'YOUTUBER'
    ],
    'Business & Law': [
        'BUSINESSPERSON', 'LAWYER', 'PRODUCER'
    ],
    'Exploration': [
        'ASTRONAUT', 'EXPLORER'
    ],
}

raw_dir, data_dir = Path(sys.argv[1]), Path(sys.argv[2])
for folder in [raw_dir / 'wikipedia', raw_dir / 'grokipedia', data_dir]:
    folder.mkdir(parents=True, exist_ok=True)
funnel = {}


# %% Pantheon 2.0: one row per person, the occupation mapped to its domain
pantheon_path = raw_dir / 'person_2025_update.csv.bz2'
if not pantheon_path.exists():
    pantheon_path.write_bytes(get(PANTHEON).content)
columns = ['name', 'slug', 'wd_id', 'occupation', 'gender', 'birthyear', 'hpi', 'bplace_country']
people = pd.read_csv(pantheon_path, usecols=columns)[columns]
domain_of = {occupation: domain for domain, occupations in DOMAINS.items() for occupation in occupations}
people['domain'] = people.occupation.map(domain_of).fillna('Other')
funnel['pantheon'] = len(people)


# %% Grokipedia sitemap: the slug of every article
sitemap_path = raw_dir / 'sitemap_slugs.txt'
if not sitemap_path.exists():
    def locations(url):
        tree = ElementTree.fromstring(get(url).content)
        return [node.text for node in tree.iter() if node.tag.endswith('loc')]
    pages = [page for sitemap in locations(SITEMAP) for page in locations(sitemap)]
    slugs = {unquote(page.split('/page/', 1)[1]) for page in pages if '/page/' in page}
    sitemap_path.write_text('\n'.join(sorted(slugs)))
grokipedia_slugs = sitemap_path.read_text().splitlines()
exact = set(grokipedia_slugs)
# where slugs differ only in case, the capitalized one wins (it sorts first)
lowercase = {slug.lower(): slug for slug in sorted(grokipedia_slugs, reverse=True)}


# %% Match every Pantheon slug (a Wikipedia title) to a Grokipedia slug
def grokipedia_slug(slug):
    """(Grokipedia slug, match type) for a Pantheon slug, (None, None) without a match. Tries the
    slug itself, then looser variants of it, each first as written and then ignoring case."""
    accents_stripped = ''.join(char for char in unicodedata.normalize('NFD', slug)
                               if unicodedata.category(char) != 'Mn')
    variants = [('exact', slug),
                ('strip_parens', re.sub(r'_?\(.*?\)', '', slug).rstrip('_')),  # Madonna_(entertainer)
                ('strip_disambig', slug.split('_of_')[0]),
                ('strip_disambig', slug.split('_the_')[0]),
                ('hyphen', slug.replace('-', '_')),  # Kim_Il-sung
                ('strip_accents', accents_stripped)]  # Shinzō_Abe
    for match_type, variant in variants:
        if variant in exact:
            return variant, match_type
        if variant.lower() in lowercase:
            return lowercase[variant.lower()], 'case' if match_type == 'exact' else match_type
    return None, None


people['gp_slug'], people['match_type'] = zip(*people.slug.map(grokipedia_slug))
matches = people.dropna(subset='gp_slug').reset_index(drop=True)
funnel['matched'] = len(matches)


# %% Scrape both articles of every match
wiki = wikipedia_client()
for match in matches.itertuples():
    wikipedia_path = raw_dir / 'wikipedia' / file_name(match.slug)
    if not wikipedia_path.exists():
        article = wikipedia_article(wiki, match.slug)
        if article:
            wikipedia_path.write_text(json.dumps(article, ensure_ascii=False))
        time.sleep(0.1)
    grokipedia_path = raw_dir / 'grokipedia' / file_name(match.gp_slug)
    if not grokipedia_path.exists():
        article, _ = grokipedia_article(match.gp_slug)
        if article:
            grokipedia_path.write_text(json.dumps(article, ensure_ascii=False))
        time.sleep(0.5)


# %% Pair the articles. The texts don't fit in memory at once, so the first pass over the files
# only keeps the counts that the cleaning needs, and a second pass writes the texts.
def read_pair(match):
    """The stored Wikipedia and Grokipedia article of a match with their cleaned texts, None if
    one of them could not be retrieved."""
    paths = [raw_dir / 'wikipedia' / file_name(match.slug), raw_dir / 'grokipedia' / file_name(match.gp_slug)]
    if not all(path.exists() for path in paths):
        return None
    wp, gp = (json.loads(path.read_text()) for path in paths)
    wp['text'] = strip_boilerplate(wp['full_text'])
    gp['text'] = markdown_to_text(gp['markdown'])
    return wp, gp


rows = []
for match in matches.itertuples(index=False):
    pair = read_pair(match)
    if pair:
        wp, gp = pair
        rows.append({**match._asdict(), 'wp_title': wp['title'], 'gp_title': gp['title'],
                     'wp_word_count': len(wp['text'].split()), 'gp_word_count': len(gp['text'].split()),
                     'wp_section_count': len(wp['sections']),
                     'gp_section_count': section_count(gp['markdown'])})
pairs = pd.DataFrame(rows)
funnel['retrieved'] = len(pairs)


# %% Cleaning
pairs = pairs.drop_duplicates('slug')  # Pantheon lists a few people twice
funnel['after_duplicates'] = len(pairs)

# pairs with a Sentence-BERT cosine below 0.4, reviewed by hand
reviews = pd.read_csv(data_dir / 'pair_reviews.csv')
pairs = pairs[~pairs.slug.isin(reviews[reviews.decision == 'mismatch'].slug)]
funnel['after_mismatches'] = len(pairs)

pairs = pairs[(pairs.wp_word_count > MIN_WORDS['wp']) & (pairs.gp_word_count > MIN_WORDS['gp'])]
funnel['after_stubs'] = len(pairs)

# where several people share a Grokipedia article, the one with the highest HPI keeps it
pairs = pairs.sort_values('hpi', ascending=False, kind='stable').drop_duplicates('gp_slug').sort_index()
funnel['unique_grokipedia'] = len(pairs)

# pairs whose openings a language model read and judged not to describe the same person
pairs = pairs[~pairs.slug.isin(pd.read_csv(data_dir / 'pair_mismatches.csv').slug)]
funnel['final'] = len(pairs)
(data_dir / 'funnel.json').write_text(json.dumps(funnel, indent=2) + '\n')


# %% Write the cleaned pairs with their texts, 1,000 pairs per row group
schema = pa.Schema.from_pandas(pairs, preserve_index=False)
for side in ['wp', 'gp']:
    schema = schema.append(pa.field(f'{side}_text', schema.field('slug').type))
with pq.ParquetWriter(data_dir / 'pairs_clean.parquet', schema, compression='zstd') as writer:
    for start in range(0, len(pairs), 1000):
        chunk = pairs.iloc[start:start + 1000].copy()
        texts = [[article['text'] for article in read_pair(pair)] for pair in chunk.itertuples()]
        chunk['wp_text'], chunk['gp_text'] = zip(*texts)
        writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))


# %% Party of the politicians, from Wikidata
party_path = data_dir / 'party.parquet'
if not party_path.exists():
    politicians = pairs.loc[pairs.occupation == 'POLITICIAN', ['slug', 'name', 'bplace_country', 'wd_id']]
    labels = party_labels(politicians.wd_id.dropna().unique().tolist())
    politicians.merge(labels, on='wd_id', how='left').to_parquet(party_path, index=False)
