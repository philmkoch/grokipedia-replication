"""Collects the November corpus: US politicians of the two major parties, scraped on both
platforms after the 2026 midterms.

    python collect_november.py DATA_DIR NOV_DIR

Everybody enters through membership in a defined group, never by name:

- panel: the politicians with a US party in DATA_DIR/party.parquet (the February corpus)
- senate, house, governor, cabinet: the member tables of the Wikipedia lists in MEMBER_LISTS
- nominee_senate, nominee_governor, nominee_house: every major-party candidate in the results
  tables of the three 2026 election pages
- mayor: cities of at least a million people in Wikipedia's list of the 50 largest cities
- president: Wikidata P35 of the United States

A person is kept with an English Wikipedia article of their own, a party and a Grokipedia page.
The party is the Wikidata label of collect.py, or the ballot label for nominees without any
party claim on Wikidata.

NOV_DIR receives the list pages (pages/, with revision ids in source_pages.json), the
memberships and people tables, one file per article and pairs_clean.parquet. Nothing that is
already there is fetched again.
"""
# %% Setup
import gzip
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
from bs4 import BeautifulSoup

from scraping import (GROKIPEDIA_PAGE, SPARQL, file_name, get, grokipedia_article,
                      markdown_to_text, now, party_labels, section_count, strip_boilerplate,
                      wikipedia_article, wikipedia_client)

WIKIPEDIA_API = 'https://en.wikipedia.org/w/api.php'
WIKIDATA_API = 'https://www.wikidata.org/w/api.php'
GROKIPEDIA_SEARCH = 'https://grokipedia.com/api/full-text-search'
GROKIPEDIA_DELAY = 0.5
# (source, Wikipedia page, column with the person, every matching table or only the largest)
MEMBER_LISTS = [('senate', 'List of current United States senators', 'Senator', False),
                ('house', 'List of current United States representatives', 'Member', False),
                ('governor', 'List of current United States governors', 'Governor', False),
                ('cabinet', 'Cabinet of the United States', 'Name', True)]
ELECTION_PAGES = [('nominee_senate', '2026 United States Senate elections'),
                  ('nominee_governor', '2026 United States gubernatorial elections'),
                  ('nominee_house', '2026 United States House of Representatives elections')]
MAYOR_PAGE = 'List of mayors of the 50 largest cities in the United States'
MAYOR_MIN_POPULATION = 1_000_000
# tables under these headings are not general elections for a state seat on 3 November
EXCLUDED_SECTIONS = {'Special elections', 'Non-voting delegates', 'Territories and federal district'}
# ballot labels of the two parties. DFL and Democratic-NPL are the Democrats of Minnesota and
# North Dakota
BALLOT_PARTY = {'Democratic': 'Democratic', 'DFL': 'Democratic', 'Democratic-NPL': 'Democratic',
                'Republican': 'Republican'}
MIN_WORDS = {'wp': 50, 'gp': 100}  # stubs, as in collect.py
MISMATCHES = ['William_Perry']  # his Grokipedia article is about another person

data_dir, nov_dir = Path(sys.argv[1]), Path(sys.argv[2])
for folder in ['pages', 'wikipedia', 'grokipedia', 'grokipedia_html']:
    (nov_dir / folder).mkdir(parents=True, exist_ok=True)
page_revisions = {}


def wikidata_id(binding, variable):
    return binding[variable]['value'].rsplit('/', 1)[1]


# %% Tables of the Wikipedia list pages
def list_page(title):
    """Rendered HTML of a Wikipedia page, frozen in NOV_DIR/pages on first fetch, so the
    population is tied to one revision of every list."""
    path = nov_dir / 'pages' / (title.replace(' ', '_') + '.json')
    if not path.exists():
        response = get(WIKIPEDIA_API, {'action': 'parse', 'page': title, 'prop': 'text|revid',
                                       'redirects': 1, 'format': 'json', 'formatversion': 2})
        path.write_text(json.dumps({**response.json()['parse'], 'fetched': now()}, ensure_ascii=False))
    page = json.loads(path.read_text())
    page_revisions[title] = {'revid': page['revid'], 'fetched': page['fetched']}
    return BeautifulSoup(page['text'], 'lxml')


def table_grid(table):
    """Rows of a wikitable with rowspan and colspan expanded, so the columns line up."""
    grid, carried = [], {}  # carried: column -> (cell, rows it still covers)
    for table_row in table.find_all('tr'):
        row = {column: cell for column, (cell, _) in carried.items()}
        carried = {column: (cell, rows - 1) for column, (cell, rows) in carried.items() if rows > 1}
        column = 0
        for cell in table_row.find_all(['th', 'td'], recursive=False):
            while column in row:
                column += 1
            rowspan, colspan = (int(re.match(r'\d+', cell.get(span, '1')).group()) for span in ['rowspan', 'colspan'])
            for _ in range(colspan):
                row[column] = cell
                if rowspan > 1:
                    carried[column] = (cell, rowspan - 1)
                column += 1
        grid.append([row[column] for column in sorted(row)])
    return grid


def header_position(grid, header):
    """(row, column) of the first header cell whose text starts with `header`, else None."""
    for row_index, row in enumerate(grid[:3]):
        for column, cell in enumerate(row):
            if cell.name == 'th' and cell.get_text(' ', strip=True).startswith(header):
                return row_index, column
    return None


def tables_with(page, header):
    """(body rows, column of `header`) of every wikitable with such a column, outside
    EXCLUDED_SECTIONS. Body rows are the rows below the header that hold data cells."""
    found = []
    for table in page.select('table.wikitable'):
        heading = table.find_previous(['h2', 'h3', 'h4'])
        if heading and heading.get_text(strip=True) in EXCLUDED_SECTIONS:
            continue
        grid = table_grid(table)
        position = header_position(grid, header)
        if position:
            body = [row for row in grid[position[0] + 1:] if any(cell.name == 'td' for cell in row)]
            found.append((body, position[1]))
    return found


def linked_person(node):
    """(article title, name) of the first article link in `node`. The title is None for a red
    link or no link."""
    for link in node.find_all('a', href=True):
        href = link['href']
        if 'redlink=1' in href:
            return None, link.get_text(' ', strip=True)
        if href.startswith('/wiki/') and ':' not in href[6:]:
            return unquote(href.removeprefix('/wiki/').split('#')[0]).replace('_', ' '), link.get_text(' ', strip=True)
    return None, node.get_text(' ', strip=True)


# %% Memberships: one row per person and group, as listed on the page
def members():
    rows = []
    for source, page, header, every_table in MEMBER_LISTS:
        tables = tables_with(list_page(page), header)
        if not every_table:
            tables = [max(tables, key=lambda table: len(table[0]))]
        for body, column in tables:
            for row in body:
                if column < len(row):
                    title, name = linked_person(row[column])
                    rows.append({'source': source, 'wp_title': title, 'name': name,
                                 'detail': row[0].get_text(' ', strip=True)})
    return rows


def candidates():
    """Every major-party candidate in the Candidates cells of the results tables. Top-two,
    top-four and jungle ballots can give a race several candidates of one party."""
    rows = []
    for source, page in ELECTION_PAGES:
        for body, column in tables_with(list_page(page), 'Candidates'):
            for row in body:
                for item in row[column].find_all('li') if column < len(row) else []:
                    text = item.get_text(' ', strip=True).replace('▌', '').strip()
                    label = re.search(r'\(([^)]*)\)', text)
                    if label and label.group(1) in BALLOT_PARTY:
                        rows.append({'source': source, 'wp_title': linked_person(item)[0],
                                     'name': re.sub(r'^Y\s+', '', text[:label.start()].strip()),
                                     'detail': row[0].get_text(' ', strip=True),
                                     'wp_party': label.group(1), 'won': text.startswith('Y ')})
    return rows


def mayors():
    page = list_page(MAYOR_PAGE)
    (body, name_column), = tables_with(page, 'Name')
    (_, population_column), = tables_with(page, 'Population')
    (_, city_column), = tables_with(page, 'City')
    rows = []
    for row in body:
        if int(re.sub(r'\D', '', row[population_column].get_text()) or 0) >= MAYOR_MIN_POPULATION:
            title, name = linked_person(row[name_column])
            rows.append({'source': 'mayor', 'wp_title': title, 'name': name,
                         'detail': row[city_column].get_text(' ', strip=True)})
    return rows


def presidents():
    """The president, with the title of the English Wikipedia article."""
    response = get(SPARQL, {'query': 'SELECT ?person WHERE { wd:Q30 wdt:P35 ?person . }', 'format': 'json'})
    ids = [wikidata_id(binding, 'person') for binding in response.json()['results']['bindings']]
    response = get(WIKIDATA_API, {'action': 'wbgetentities', 'ids': '|'.join(ids), 'props': 'sitelinks',
                                  'sitefilter': 'enwiki', 'format': 'json'})
    entities = response.json()['entities']
    titles = [entities[person].get('sitelinks', {}).get('enwiki', {}).get('title') for person in ids]
    return pd.DataFrame({'source': 'president', 'wd_id': ids, 'wp_title': titles, 'wp_current': titles})


# %% Lookups on Wikipedia, Wikidata and Grokipedia
def resolve_titles(titles):
    """Current title and Wikidata id per Wikipedia title, following redirects. Both are missing
    where the page does not exist."""
    rows = []
    for start in range(0, len(titles), 50):
        batch = titles[start:start + 50]
        query = get(WIKIPEDIA_API, {'action': 'query', 'titles': '|'.join(batch), 'prop': 'pageprops',
                                    'ppprop': 'wikibase_item', 'redirects': 1, 'format': 'json',
                                    'formatversion': 2}).json()['query']
        moved = {step['from']: step['to'] for step in query.get('normalized', []) + query.get('redirects', [])}
        pages = {page['title']: page for page in query['pages']}
        for title in batch:
            current = title
            while current in moved:
                current = moved[current]
            page = pages.get(current, {})
            exists = 'missing' not in page and 'invalid' not in page
            rows.append({'wp_title': title, 'wp_current': current if exists else None,
                         'wd_id': page.get('pageprops', {}).get('wikibase_item') if exists else None})
        time.sleep(0.2)
    return pd.DataFrame(rows)


def humans(wikidata_ids):
    """The Wikidata ids that are instances of human (P31 Q5)."""
    found = set()
    for start in range(0, len(wikidata_ids), 300):
        values = ' '.join(f'wd:{person}' for person in wikidata_ids[start:start + 300])
        query = f'SELECT ?person WHERE {{ VALUES ?person {{ {values} }} ?person wdt:P31 wd:Q5 }}'
        bindings = get(SPARQL, {'query': query, 'format': 'json'}).json()['results']['bindings']
        found |= {wikidata_id(binding, 'person') for binding in bindings}
    return found


def on_grokipedia(slug):
    response = get(GROKIPEDIA_PAGE + slug, method='HEAD')
    time.sleep(GROKIPEDIA_DELAY)
    if response.status_code not in [200, 404]:
        response.raise_for_status()
    return response.status_code == 200


def normalized(title):
    title = ''.join(char for char in unicodedata.normalize('NFD', title) if unicodedata.category(char) != 'Mn')
    return re.sub(r'[\s_]+', ' ', title).strip().lower()


def grokipedia_slug(title, panel_slug):
    """(Grokipedia slug, match type) of a person, (None, 'none') without a page. First hit of
    the panel's February slug, the Wikipedia title, a title that redirects to the Wikipedia
    article (Grokipedia slugs are the Wikipedia titles of its import, so mostly former names)
    and a search hit with the same title. Existence is checked live, because the sitemap is stale.
    """
    if panel_slug and on_grokipedia(panel_slug):
        return panel_slug, 'panel'
    slug = title.replace(' ', '_')
    if slug != panel_slug and on_grokipedia(slug):
        return slug, 'exact'
    pages = get(WIKIPEDIA_API, {'action': 'query', 'titles': title, 'prop': 'redirects', 'rdlimit': 'max',
                                'rdnamespace': 0, 'format': 'json', 'formatversion': 2}).json()['query']['pages']
    for redirect in [redirect['title'] for page in pages for redirect in page.get('redirects', [])]:
        if on_grokipedia(redirect.replace(' ', '_')):
            return redirect.replace(' ', '_'), 'redirect'
    hits = get(GROKIPEDIA_SEARCH, {'query': title, 'limit': 10}).json().get('results', [])
    time.sleep(GROKIPEDIA_DELAY)
    for hit in hits:
        if normalized(hit['title']) == normalized(title) and on_grokipedia(hit['slug']):
            return hit['slug'], 'search'
    return None, 'none'


# %% The population: one row per person, `keep` marks who gets scraped
def build_people():
    panel = pd.read_parquet(data_dir / 'party.parquet').dropna(subset='us_party')
    panel = panel.merge(pd.read_parquet(data_dir / 'pairs_clean.parquet', columns=['slug', 'gp_slug']), on='slug', how='left')
    panel = pd.DataFrame({'source': 'panel', 'wp_title': panel.slug.str.replace('_', ' '), 'name': panel['name'],
                          'panel_wd_id': panel.wd_id, 'panel_slug': panel.slug, 'panel_gp_slug': panel.gp_slug})
    # two redistricted incumbents share one candidates cell, Louisiana lists runoff candidates twice
    memberships = pd.concat([panel, pd.DataFrame(members() + candidates() + mayors())], ignore_index=True)
    memberships = memberships.drop_duplicates(['source', 'detail', 'name', 'wp_title'])
    memberships = memberships.merge(resolve_titles(sorted(memberships.wp_title.dropna().unique())),
                                    on='wp_title', how='left')
    memberships = pd.concat([memberships, presidents()], ignore_index=True)
    memberships.to_parquet(nov_dir / 'memberships.parquet', index=False)
    (nov_dir / 'source_pages.json').write_text(json.dumps(page_revisions, indent=1))

    # panel people keep their Pantheon id, even if their old title now resolves elsewhere
    memberships['person'] = memberships.panel_wd_id.fillna(memberships.wd_id)
    memberships['redirected'] = memberships.wp_current != memberships.wp_title
    people = memberships.dropna(subset='person').groupby('person').agg(
        wp_title=('wp_current', 'first'), redirected=('redirected', 'all'), name=('name', 'first'),
        sources=('source', lambda sources: '|'.join(sorted(set(sources)))),
        panel_slug=('panel_slug', 'first'), panel_gp_slug=('panel_gp_slug', 'first'),
    ).reset_index().rename(columns={'person': 'wd_id'})

    # a name that redirects to something that is not a person (mostly the race page) has no
    # article of its own
    redirected = people[people.redirected].wd_id
    people['has_wp'] = people.wp_title.notna() & ~people.wd_id.isin(set(redirected) - humans(redirected.tolist()))

    people = people.merge(party_labels(people.wd_id.tolist()), on='wd_id', how='left')
    people['party_source'] = people.us_party.notna().map({True: 'wikidata', False: None})
    ballot = memberships.dropna(subset='wp_party').assign(party=lambda rows: rows.wp_party.map(BALLOT_PARTY))
    ballot = ballot.groupby('person').party.agg(lambda parties: parties.iloc[0] if parties.nunique() == 1 else None)
    fallback = people.parties.isna() & people.wd_id.map(ballot).notna()
    people.loc[fallback, 'us_party'] = people.loc[fallback, 'wd_id'].map(ballot)
    people.loc[fallback, 'party_source'] = 'ballot'

    eligible = people.has_wp & people.us_party.notna()
    lookups = [grokipedia_slug(person.wp_title, person.panel_gp_slug if isinstance(person.panel_gp_slug, str) else None)
               for person in people[eligible].itertuples()]
    people.loc[eligible, ['gp_slug', 'gp_match']] = lookups
    people['keep'] = eligible & people.gp_slug.notna()
    people.to_parquet(nov_dir / 'people.parquet', index=False)


if not (nov_dir / 'people.parquet').exists():
    build_people()
people = pd.read_parquet(nov_dir / 'people.parquet').query('keep')
# panel people keep their February slug, so the two scrapes join on it
people['slug'] = people.panel_slug.fillna(people.wp_title.str.replace(' ', '_'))


# %% Scrape both articles of every person, Grokipedia right after Wikipedia
wiki = wikipedia_client()
for person in people.itertuples():
    wikipedia_path = nov_dir / 'wikipedia' / file_name(person.slug)
    if not wikipedia_path.exists():
        article = wikipedia_article(wiki, person.wp_title)
        if article:
            wikipedia_path.write_text(json.dumps(article, ensure_ascii=False))
        time.sleep(0.2)
    grokipedia_path = nov_dir / 'grokipedia' / file_name(person.gp_slug)
    if not grokipedia_path.exists():
        article, html = grokipedia_article(person.gp_slug)
        if html:  # kept, so a layout change only needs a reparse
            (nov_dir / 'grokipedia_html' / file_name(person.gp_slug)).with_suffix('.html.gz').write_bytes(gzip.compress(html))
        if article:
            grokipedia_path.write_text(json.dumps(article, ensure_ascii=False))
        time.sleep(GROKIPEDIA_DELAY)


# %% Pair the articles, cleaned like the February corpus (no stubs, one pair per Grokipedia article)
rows = []
for person in people.itertuples():
    paths = [nov_dir / 'wikipedia' / file_name(person.slug), nov_dir / 'grokipedia' / file_name(person.gp_slug)]
    if not all(path.exists() for path in paths):
        continue
    wp, gp = (json.loads(path.read_text()) for path in paths)
    wp_text, gp_text = strip_boilerplate(wp['full_text']), markdown_to_text(gp['markdown'])
    rows.append({'slug': person.slug, 'gp_slug': person.gp_slug, 'wd_id': person.wd_id, 'name': person.name,
                 'us_party': person.us_party, 'party_source': person.party_source, 'sources': person.sources,
                 'panel': isinstance(person.panel_slug, str),
                 'wp_title': wp['title'], 'wp_revid': wp['revid'], 'wp_fetched_at': wp['fetched_at'],
                 'gp_title': gp['title'], 'gp_date_modified': gp['date_modified'], 'gp_fetched_at': gp['fetched_at'],
                 'wp_text': wp_text, 'gp_text': gp_text,
                 'wp_word_count': len(wp_text.split()), 'gp_word_count': len(gp_text.split()),
                 'wp_section_count': len(wp['sections']),
                 'gp_section_count': section_count(gp['markdown'])})
pairs = pd.DataFrame(rows)
pairs = pairs[(pairs.wp_word_count > MIN_WORDS['wp']) & (pairs.gp_word_count > MIN_WORDS['gp'])]
pairs = pairs[~pairs.slug.isin(MISMATCHES)]
pairs.drop_duplicates('gp_slug').to_parquet(nov_dir / 'pairs_clean.parquet', index=False, compression='zstd')
