"""Recreates every figure and statistic of the manuscript, then checks them against the manuscript.

    python reproduce.py                  # download the corpora if missing, run all steps, check
    python reproduce.py --skip-download  # the corpora are already in data/
    python reproduce.py ideology musk    # only these steps (names in STEPS), in the order given

Needs Python 3.11, the packages in requirements.txt and about 7 GB of memory (the results step). The
figures end up in out/figures/, the numbers in out/*/results.json, the check in out/manuscript_check.md.
The order matters: paper_figures.py compares itself with out/ideology/results.json.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEB = ['data', 'features']
NOV = ['data/november', 'features/november']
STEPS = {
    # name: list of (script, arguments), directories relative to this folder
    'ideology': [('ideology.py', ['features', *NOV, 'data/external', 'out/ideology'])],
    'musk': [('musk_support.py', ['features', *NOV, 'data/external', 'out/musk'])],
    'results': [('results.py', [*FEB, 'out'])],
    'november': [('rescrape.py', [*FEB, *NOV, 'out/november'])],
    'figures': [('paper_figures.py', [*FEB, *NOV, 'data/external', 'out/paper']),
                ('fig1_reference.py', [*FEB, *NOV, 'out/paper'])],
    'check': [('manuscript_numbers.py', ['.'])],
}


def collect_figures():
    """Fig. 1 is fig1_v9 (the one with the 98,013 biographies as reference), Fig. 2 is fig2."""
    figures = ROOT / 'out' / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    for source, target in [('fig1_v9', 'fig1'), ('fig2', 'fig2')]:
        for extension in ['pdf', 'png']:
            shutil.copy(ROOT / 'out' / 'paper' / f'{source}.{extension}', figures / f'{target}.{extension}')


def main(arguments):
    names = [argument for argument in arguments if not argument.startswith('--')] or list(STEPS)
    unknown = set(names) - set(STEPS)
    if unknown:
        sys.exit(f'unknown step {sorted(unknown)}, choose from {list(STEPS)}')
    if '--skip-download' not in arguments and ({'results', 'november', 'figures', 'ideology', 'musk'} & set(names)):
        subprocess.run([sys.executable, 'download_data.py'], cwd=ROOT, check=True)
    environment = {**os.environ, 'PYTHONUTF8': '1', 'MPLBACKEND': 'agg'}
    for name in names:
        for script, script_arguments in STEPS[name]:
            print(f'\n=== {name}: {script} {" ".join(script_arguments)}', flush=True)
            start = time.time()
            done = subprocess.run([sys.executable, script, *script_arguments], cwd=ROOT, env=environment)
            print(f'--- {script} {"finished" if done.returncode == 0 else "FAILED"} in {time.time() - start:.0f} s',
                  flush=True)
            if done.returncode:
                sys.exit(done.returncode)
        if name == 'figures':
            collect_figures()


if __name__ == '__main__':
    main(sys.argv[1:])
