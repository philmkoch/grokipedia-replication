"""Per-pair features for the Grokipedia (GP) vs Wikipedia (WP) comparison.

    python features.py DATA_DIR OUT_DIR [SLUGS_FILE]

Reads DATA_DIR/pairs_clean.parquet, writes OUT_DIR/features.parquet (one row per pair).
SLUGS_FILE restricts the pairs to one slug per line.
"""
# %% Setup
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SIDES = ['wp', 'gp']
MIN_SENTENCE_WORDS = 5  # shorter "sentences" are mostly headings and list fragments
LEAD_WORDS = 750  # the lead is the sentences up to and including the one crossing word 750
BATCH = 32  # classifier batch that fits DEBATE-large on a 16 GB Mac

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

data_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
slugs = set(Path(sys.argv[3]).read_text().splitlines()) if len(sys.argv) > 3 else None
pairs_path = data_dir / 'pairs_clean.parquet'
out_dir.mkdir(parents=True, exist_ok=True)


def clean(text):
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def pair_batches():
    """Yield the pairs 128 at a time with cleaned texts, only those in SLUGS_FILE if given.
    The full texts don't fit in memory at once."""
    columns = ['slug', 'gp_section_count', 'wp_text', 'gp_text']
    for batch in pq.ParquetFile(pairs_path).iter_batches(batch_size=128, columns=columns):
        pairs = batch.to_pandas()
        if slugs is not None:
            pairs = pairs[pairs.slug.isin(slugs)]
        if pairs.empty:
            continue
        for side in SIDES:
            pairs[f'{side}_text'] = pairs[f'{side}_text'].map(clean)
        yield pairs


# %% RQ2: exact word-trigram overlap (the GP section count is passed through from the corpus)
def trigrams(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return set(zip(words, words[1:], words[2:]))


def lexical(pairs):
    rows = []
    for pair in pairs.itertuples():
        wp, gp = trigrams(pair.wp_text), trigrams(pair.gp_text)
        jaccard = len(wp & gp) / len(wp | gp)
        overlap = len(wp & gp) / min(len(wp), len(gp))
        rows.append([pair.slug, pair.gp_section_count, np.float32(jaccard), np.float32(overlap)])
    return pd.DataFrame(rows, columns=['slug', 'gp_section_count', 'jaccard_3gram', 'overlap_3gram'])



# %% RQ3: sentence-level bias (DA-RoBERTa) and stance (Political DEBATE, zero-shot NLI)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner', 'lemmatizer', 'attribute_ruler'])
nlp.enable_pipe('senter')
nlp.max_length = 500_000


def sentences(texts):
    """One list of sentences per article."""
    articles = []
    for doc in nlp.pipe(texts, batch_size=64):
        kept = [sentence.text.strip() for sentence in doc.sents]
        articles.append([sentence for sentence in kept if len(sentence.split()) >= MIN_SENTENCE_WORDS])
    return articles


def classify(articles, tokenizer, model, hypothesis=None):
    """Score every sentence and return one probability array per article. Without a hypothesis
    it's P(biased), class 1 of DA-RoBERTa. With one it's P(entailment), class 0 of DEBATE."""
    flat = [sentence for article in articles for sentence in article]
    # length-sorted batches need little padding
    order = np.argsort([len(sentence.split()) for sentence in flat])
    probabilities = np.zeros(len(flat), dtype=np.float32)
    for start in range(0, len(order), BATCH):
        indices = order[start:start + BATCH]
        batch = [flat[i] for i in indices]
        hypotheses = [hypothesis] * len(batch) if hypothesis else None
        inputs = tokenizer(batch, hypotheses, padding=True, truncation=True, max_length=512,
                           return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits.float()
        probabilities[indices] = logits.softmax(dim=-1)[:, 0 if hypothesis else 1].cpu().numpy()
    return np.split(probabilities, np.cumsum([len(article) for article in articles])[:-1])


def mean(values):
    return float(values.mean()) if len(values) else np.nan


def lead_length(article):
    """Number of sentences in the lead."""
    return np.searchsorted(np.cumsum([len(sentence.split()) for sentence in article]), LEAD_WORDS) + 1


def bias_features(pairs, tokenizer, model):
    result = pd.DataFrame({'slug': pairs.slug.to_numpy()})
    for side in SIDES:
        articles = sentences(pairs[f'{side}_text'])
        scores = classify(articles, tokenizer, model)
        result[f'{side}_bias_score'] = [mean(article) for article in scores]
        result[f'{side}_bias_frac'] = [mean(article > 0.5) for article in scores]
        result[f'{side}_bias_score_lead'] = [mean(scored[:lead_length(article)])
                                             for scored, article in zip(scores, articles)]
    return result


def stance_features(pairs, tokenizer, model):
    result = pd.DataFrame({'slug': pairs.slug.to_numpy()})
    for side in SIDES:
        articles = sentences(pairs[f'{side}_text'])
        for key, camp in [('con', 'conservatives'), ('lib', 'liberals')]:
            scores = classify(articles, tokenizer, model, f'The author of this text supports {camp}.')
            result[f'{side}_stance_{key}'] = [mean(article) for article in scores]
    return result


# both models stay loaded so the 2 GB pairs file is read once, pinned to the Hugging Face commit
# we used; stance runs in fp16 on GPU like the published run
stance_dtype = torch.float32 if DEVICE == 'cpu' else torch.float16
models = [('mediabiasgroup/da-roberta-babe-ft', 'a1e1b9e2ac97f87387fb0b78e5a1914b5e25532f',
           torch.float32, bias_features),
          ('mlburnham/Political_DEBATE_large_v1.0', '1a3aff1ecb97ad93a6de598f7414b1707de7e7c3',
           stance_dtype, stance_features)]
scorers = []
for name, revision, dtype, score in models:
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision, dtype=dtype)
    scorers.append((score, tokenizer, model.to(DEVICE).eval()))


# %% One pass over the pairs, 128 at a time
parts = []
for pairs in pair_batches():
    part = lexical(pairs)
    for score, tokenizer, model in scorers:
        part = part.merge(score(pairs, tokenizer, model), on='slug')
    parts.append(part)
    print(f'Scored {sum(map(len, parts))} pairs', flush=True)
pd.concat(parts, ignore_index=True).to_parquet(out_dir / 'features.parquet', index=False)
