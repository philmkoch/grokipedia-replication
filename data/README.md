# Data

Everything the analysis reads is here except the two text corpora, which `download_data.py` fetches
(see the main README). Paths are relative to the repository root.

## Text corpora (downloaded, not in git)

| File | Rows | What it is |
|---|--:|---|
| `data/pairs_clean.parquet` | 98,013 | February 2026 snapshot (Grokipedia scraped on 11 and 12 February). One row per person of the Pantheon 2.0 list who has a biography on both English Wikipedia and Grokipedia: metadata, `wp_text`, `gp_text`, word and section counts. |
| `data/november/pairs_clean.parquet` | 1,614 | 8 November 2026 snapshot of US two-party politicians. `panel` is true for the 1,162 politicians of the February sample, `sources` lists the groups a person was drawn from (panel, house, senate, governor, cabinet, mayor, president, nominee_*), `us_party` is Democratic or Republican, `gp_date_modified` is Grokipedia's own edit date. |

SHA-256 and sizes are in `download_data.py`. The corpora contain the full article texts of both sites. They
are not covered by this repository's MIT licence. Terms as read on 23 November 2026:

- **Wikipedia text** (`wp_text`) is CC BY-SA 4.0. Reuse needs attribution (the article title is in `wp_title`,
  and each article's page history names its authors) and the same licence for adaptations. The texts were cleaned
  and cut at the first boilerplate section, so they count as adaptations.
- **Grokipedia text** (`gp_text`): xAI's Terms of Service (https://x.ai/legal/terms-of-service, and the versions of
  4 November 2025 and 10 April 2026) designate Grokipedia content as material under the xAI Community License
  Agreement (https://huggingface.co/xai-org/grok-2/blob/main/LICENSE) and say that some of it is also CC BY-SA 4.0.
  That agreement allows use, copying, distribution and modification for non-commercial and research purposes.
  Whoever distributes the material must include the agreement, add the copyright notice that section 3 of it
  prescribes and display "Powered by xAI". Nobody may use the material to train or improve foundation, large
  language or general-purpose AI models. The licence is revocable and ends if its terms are breached.

Use the corpora for replication and other research under these terms only. Anyone who hosts them has to
attach the same conditions to the download.

## Scores and metadata (in git)

| File | What it is |
|---|---|
| `features/features.parquet` | One row per February pair: trigram overlap, DA-RoBERTa bias scores and Political DEBATE stance (`*_stance_con`, `*_stance_lib`) for WP and GP. Written by `features.py` on a GPU with both models pinned to fixed commits. |
| `features/november/features.parquet` | The same for the November pairs. |
| `data/party.parquet` | US party of politicians among the Pantheon people, from Wikidata property P102 (`parties`) reduced to Democratic or Republican when exactly one of the two appears (`us_party`). |
| `data/dewine_sentences.parquet` | Per-sentence P(supports conservatives) of the Mike DeWine pair (Fig. 1a). `side` is wp or gp, `index` the sentence position. The `sentence` column holds the text of the sentences, so the file reproduces most of both articles and falls under the text terms above. |
| `data/gentzkow_shapiro/` | Partisan bigrams per Congress (104 to 114), positive is Republican, and the Snowball stopword list. The phrase lists are those of Gentzkow, Shapiro and Taddy (2019, Econometrica 87(4)). |
| `data/ethnicity.parquet`, `ethnicity_cross_validation.parquet`, `ethnicity_validation.csv` | Ethnicity labels (GPT-5 mini from name and birthplace) and their validation. Used by `results.py` for the project report, not by the manuscript. The labeling script is not included. |
| `data/pair_reviews.csv`, `data/pair_mismatches.csv` | The two review steps that removed Wikipedia–Grokipedia mismatches when the 98,013 pairs were built (Sentence-BERT cosine below 0.4, and a second model read of borderline pairs), with the decision and reason for each pair. The scripts are not included. |

## External public data (in git, `data/external/`)

| File | Source | Used for |
|---|---|---|
| `HSall_members.csv` | Voteview, all Congresses, https://voteview.com/data (Lewis et al.) | DW-NOMINATE first dimension, averaged over each politician's career |
| `HS119_members.csv` | Voteview, 119th Congress | the current-Congress check |
| `wikidata_ids.csv` | Wikidata SPARQL (properties P1839 FEC id, P1157 bioguide id) for the politicians in the November corpus | matching politicians to Voteview and to the FEC |
| `independent_expenditure_2024.csv`, `independent_expenditure_2026.csv` | FEC bulk downloads, independent expenditures | America PAC (committee C00879510) spending |
| `cn24.zip`, `cn26.zip` | FEC bulk downloads, candidate master files | linking politicians to FEC candidate ids |

All of these were downloaded on 19 November 2026 and are kept as they were. The FEC files grow every day,
so a fresh download would give slightly different 2026 totals. Voteview and the FEC publish them as public
records. The FEC asks that its data not be used to solicit contributions or for commercial purposes.
