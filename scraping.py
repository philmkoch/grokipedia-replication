"""Fetching and parsing shared by collect.py and collect_november.py: Wikipedia articles,
Grokipedia articles and party membership from Wikidata."""
import copy
import json
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import wikipediaapi
from bs4 import BeautifulSoup

HEADERS = {'User-Agent': 'grokipedia-bias-research/1.0 (university student project)'}
GROKIPEDIA_PAGE = 'https://grokipedia.com/page/'
SPARQL = 'https://query.wikidata.org/sparql'
DEMOCRATIC, REPUBLICAN = 'Q29552', 'Q29468'
# Wikipedia sections without a counterpart on Grokipedia. The text is cut at the first of them.
BOILERPLATE = {'See also', 'Notes', 'References', 'Further reading', 'External links',
               'Works cited', 'Bibliography', 'Sources'}


def get(url, params=None, method='GET'):
    """HTTP request that waits out 429 and 5xx answers (Retry-After, or 10, 20, 40 s)."""
    for attempt in range(4):
        response = requests.request(method, url, params=params, headers=HEADERS, timeout=120)
        if response.status_code != 429 and response.status_code < 500:
            return response
        time.sleep(int(response.headers.get('Retry-After', 10 * 2 ** attempt)))
    response.raise_for_status()


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def file_name(slug):
    return slug.replace('/', '__') + '.json'


# %% Wikipedia
def wikipedia_client():
    return wikipediaapi.Wikipedia(user_agent=HEADERS['User-Agent'], language='en',
                                  extract_format=wikipediaapi.ExtractFormat.WIKI)


def wikipedia_article(wiki, title):
    """Plain text, sections (two levels), revision id and fetch time of an article. None if the
    page does not exist."""
    for attempt in range(4):
        try:
            page = wiki.page(title)
            if not page.exists():
                return None
            sections = [{'title': section.title, 'text': section.text}
                        for top in page.sections for section in [top, *top.sections]]
            return {'title': page.title, 'revid': page.lastrevid, 'fetched_at': now(),
                    'full_text': page.text, 'sections': sections}
        except Exception:  # wikipedia-api raises rate limits as JSON or HTTP errors
            time.sleep(10 * 2 ** attempt)
    return None


def strip_boilerplate(text):
    """Wikipedia text up to its first boilerplate section."""
    starts = [text.find(f'\n{title}\n') for title in BOILERPLATE]
    return text[:min([start for start in starts if start > 0], default=len(text))].strip()


# %% Grokipedia
def inline_text(node):
    """Visible text of a block, without citation markers and anchor buttons."""
    node = copy.copy(node)
    for junk in node.find_all(['sup', 'button', 'svg']):
        junk.decompose()
    for line_break in node.find_all('br'):
        line_break.replace_with('\n')
    lines = [re.sub(r'\s+', ' ', line).strip() for line in node.get_text().split('\n')]
    return '\n'.join(line for line in lines if line)


def markdown_lines(node):
    """The children of the article body as markdown lines. Headings become #-lines, table rows
    |-lines and list items "- " lines. Figures are dropped."""
    lines = []
    for child in node.find_all(recursive=False):
        if child.name in ['h2', 'h3', 'h4', 'h5']:
            lines.append('#' * int(child.name[1]) + ' ' + inline_text(child))
        elif child.name == 'figure':
            continue
        elif child.name == 'table' or (child.name == 'div' and child.find('table')):
            for row in child.find_all('tr'):
                lines.append('| ' + ' | '.join(inline_text(cell) for cell in row.find_all(['th', 'td'])) + ' |')
        elif child.name in ['ul', 'ol']:
            lines += ['- ' + inline_text(item) for item in child.find_all('li', recursive=False)]
        elif child.name == 'div':
            lines += markdown_lines(child)
        elif inline_text(child):
            lines.append(inline_text(child))
    return lines


def grokipedia_article(slug):
    """(article, raw HTML) of a Grokipedia page. The article holds the title, the body as markdown,
    dateModified and the fetch time, and is None if the page is missing or has no article body.

    The page is server-rendered HTML since spring 2026, with the body in div.article-body. The
    February 2026 scrape read the same markdown out of the page's Next.js scripts instead.
    """
    response = get(GROKIPEDIA_PAGE + slug)
    if response.status_code != 200:
        return None, None
    response.encoding = 'utf-8'  # the server sends no charset
    soup = BeautifulSoup(response.text, 'lxml')
    article = soup.find('article')
    body = article.select_one('.article-body') if article else None
    if body is None:
        return None, response.content
    title = inline_text(article.find('h1')) if article.find('h1') else slug.replace('_', ' ')
    date_modified = None
    for script in soup.find_all('script', type='application/ld+json'):
        linked_data = json.loads(script.string or '{}')
        if isinstance(linked_data, dict) and linked_data.get('@type') == 'Article':
            date_modified = linked_data.get('dateModified')
    markdown = '\n\n'.join([f'# {title}', *markdown_lines(body)])
    return {'title': title, 'slug': slug, 'date_modified': date_modified, 'fetched_at': now(),
            'markdown': markdown}, response.content


def section_count(markdown):
    """Number of ## and ### headings of a Grokipedia article."""
    return len(re.findall(r'^#{2,3}\s+', markdown, re.MULTILINE))


def markdown_to_text(markdown):
    """Grokipedia markdown as plain text. Drops the title, tables, images and comments, and
    strips heading, link, emphasis and code markup."""
    lines = markdown.split('\n')
    first = lines[0].strip() if lines else ''
    title = re.sub(r'^#\s+', '', first) if re.match(r'^#\s+', first) else None
    kept = []
    for position, line in enumerate(lines):
        line = line.strip()
        if position == 0 and title is not None:
            continue
        if not line or line.startswith(('|', '<!--', '![')):
            continue
        line = re.sub(r'^#+\s*', '', line)
        line = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', line)
        line = re.sub(r'\[\]\([^)]*\)', '', line)  # what a citation link nested in a link leaves behind
        line = re.sub(r'\*{1,3}', '', line)
        line = re.sub(r'`([^`]*)`', r'\1', line).strip()
        # Grokipedia often repeats the name as a caption before the body starts
        if line and not (title and not kept and line == title):
            kept.append(line)
    return '\n'.join(kept)


# %% Wikidata
def party_labels(wikidata_ids):
    """One row per person with a P102 claim (member of political party). `parties` joins all
    party names with " | ". `us_party` is Democratic or Republican if exactly one of the two
    is among the claims, otherwise missing."""
    claims = []
    for start in range(0, len(wikidata_ids), 300):
        values = ' '.join(f'wd:{wikidata_id}' for wikidata_id in wikidata_ids[start:start + 300])
        query = f'''SELECT ?item ?party ?partyLabel WHERE {{
            VALUES ?item {{ {values} }}
            ?item wdt:P102 ?party .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}'''
        response = get(SPARQL, {'query': query, 'format': 'json'})
        response.raise_for_status()
        claims += [(binding['item']['value'].rsplit('/', 1)[1], binding['party']['value'].rsplit('/', 1)[1],
                    binding['partyLabel']['value']) for binding in response.json()['results']['bindings']]
        time.sleep(1)
    claims = pd.DataFrame(claims, columns=['wd_id', 'party_id', 'party'])
    people = claims.groupby('wd_id').agg(parties=('party', lambda names: ' | '.join(sorted(set(names)))),
                                         party_ids=('party_id', set)).reset_index()
    democrat, republican = (people.party_ids.map(lambda ids: party in ids) for party in [DEMOCRATIC, REPUBLICAN])
    people['us_party'] = None
    people.loc[democrat & ~republican, 'us_party'] = 'Democratic'
    people.loc[republican & ~democrat, 'us_party'] = 'Republican'
    return people.drop(columns='party_ids')
