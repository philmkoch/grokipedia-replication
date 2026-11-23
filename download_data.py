"""Downloads the two text corpora the analysis reads and checks them against their SHA-256.

    python download_data.py            # fetch what is missing or damaged
    python download_data.py --check    # only verify what is on disk

The corpora hold the article texts of both encyclopedias (1.4 GB for the 98,013 February pairs, 42 MB for
the 1,614 November politicians). They are too large for git, so they sit in a release. To host them
elsewhere, upload the two files under the names in FILES and change BASE_URL, or set GROKIPEDIA_DATA_URL.
The checksums stay the same wherever the files are hosted. A file that is already on disk with the right
checksum is skipped, and a download is written to a .part file first, so an interrupted run leaves no
half file behind.
"""
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE_URL = os.environ.get('GROKIPEDIA_DATA_URL',
                          'https://github.com/alexander-posztos/grokipedia-bias/releases/download/data-v1')
# name in the release, path in this repository, size in bytes, SHA-256
FILES = [
    ('pairs_clean.parquet', 'data/pairs_clean.parquet', 1445991682,
     '1392deb8aff409eddcb47dbdd7e954d4a888ac51e4707213e9fed94fc903127c'),
    ('pairs_clean_november.parquet', 'data/november/pairs_clean.parquet', 42140492,
     'c4b45ac59222a3e93333e9339cc7249845b57e585c0ea89d541da5c6aedb5b23'),
]
CHUNK = 1 << 24


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        while block := file.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def intact(path, size, checksum):
    return path.exists() and path.stat().st_size == size and sha256(path) == checksum


def download(url, path, size):
    part = path.with_name(path.name + '.part')
    path.parent.mkdir(parents=True, exist_ok=True)
    done, shown = 0, -1
    with urllib.request.urlopen(url) as response, open(part, 'wb') as file:
        while block := response.read(CHUNK):
            file.write(block)
            done += len(block)
            if done * 20 // size != shown:  # a line per 5%
                shown = done * 20 // size
                print(f'  {done / 1e6:,.0f} of {size / 1e6:,.0f} MB', flush=True)
    part.replace(path)


def main(check_only):
    failed = False
    for name, target, size, checksum in FILES:
        path = ROOT / target
        if intact(path, size, checksum):
            print(f'ok       {target}')
            continue
        if check_only:
            print(f'MISSING or damaged: {target}')
            failed = True
            continue
        print(f'fetching {target} from {BASE_URL}/{name}')
        try:
            download(f'{BASE_URL}/{name}', path, size)
        except OSError as error:  # URLError and HTTPError are OSErrors
            path.with_name(path.name + '.part').unlink(missing_ok=True)
            sys.exit(f'could not fetch {BASE_URL}/{name}: {error}\nCheck BASE_URL in download_data.py.')
        if not intact(path, size, checksum):
            path.unlink()
            sys.exit(f'{target} does not match its checksum, so it was deleted. Is BASE_URL right?')
        print(f'ok       {target}')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main('--check' in sys.argv[1:])
