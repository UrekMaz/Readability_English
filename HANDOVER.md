# Readability English — Project Handover

As of 17 Nov 2026

## Overview

This repository holds the English automated readability assessment (ARA) work: predicting the school grade (2–8) of a text from handcrafted linguistic features, calibrated on NCERT textbooks. The write-up is the paper draft *Readability Assistant: A Tool for Automated Analysis and Leveling of Primary-Grade English Texts for Indian School Curricula* (EACL / ARR August submission).

The pipeline has three parts:

1. **Feature extraction** — features grouped into dimensions D1–D10 (surface/phonological, morphology, lexical sophistication, syntax, grammar, cohesion, semantic coherence, discourse, semantics/pragmatics, readability indices). Code is in `features/`.
2. **Modelling** — 5-fold outer CV, per-model 3-stage feature selection, nested hyperparameter tuning, ensembles, and baselines B1–B6. Code is in the root notebooks.
3. **Evaluation** — Cohen's d per grade transition, ablations, significance tests, and cross-corpus checks on a Telangana textbook set and OneStopEnglish (OSE).

The repo is not committed yet: `main` has no commits and every file is untracked. Notebooks were run on Kaggle/Colab, so their paths point at `/kaggle/input/...` or `/content/...`, not this folder.

## Folder map

The working set is the repo root; `readability_english_15.11.26/` is a byte-identical backup copy of it (checked with a recursive diff) and can be ignored or deleted.

```
Readability_English/
├── data/                                   Input datasets (2 Excel files)
├── features/                               Feature-extraction code (3 .py modules, 3 notebooks)
├── papers/                                 Reference literature (22 PDFs)
├── readability_english_15.11.26/           Backup snapshot of everything above (15 Nov 2026)
├── 2-08-26-final-methodology (3).ipynb     MAIN notebook: full pipeline + evaluation
├── baseline-01-08-26 (3).ipynb             Baselines B1–B8 only
├── upsampling (1).ipynb                    Class-resampling (SMOTE etc.) ablation
├── ncert_readability_experiments.zip       Archive of the May–June 2026 experiment repo + results
├── EACL_Readability_Assistant_ARR_Aug.pdf  Paper draft
├── Analysis (1).docx                       Notes on preprocessing + feature selection
├── feature2.docx                           Feature brainstorming notes (Jan 2026)
├── ADDITIONAL FEATURES.docx                Literature notes on extra features/models (Mar 2026)
├── phonological_features.pdf               Reference paper (same file as papers/phonetics.pdf)
└── HANDOVER.md                             This document
```

## Folder by folder

### data/

Both files hold one row per text: `index`, `grade`, `content`, then ~700 precomputed feature columns.

| File | Rows | Columns | Grades (texts per grade) | Role |
| --- | --- | --- | --- | --- |
| `english_supp_14.05.26_cleant.xlsx` | 172 | 710 | 2: 26 · 3: 19 · 4: 21 · 5: 27 · 6: 27 · 7: 26 · 8: 26 | Main NCERT training set. Features include Coh-Metrix-style codes (DESWC, PCNARz, …). Note the column header is `"grade "` with a trailing space. |
| `final_telangana_dataset.xlsx` | 71 | 707 | 2: 8 · 3: 7 · 4: 9 · 5: 8 · 6: 15 · 7: 12 · 8: 12 | Telangana state textbooks, used only for cross-corpus validation. Features use LingFeat-style names (t_word, a_word_ps, …). |

Two other inputs the notebooks need are not in `data/`: the LingFeat baseline features (`english_no_trunc_baseline_ling_feat…`) and `onestop_coh.csv` (OSE, available inside the zip).

### features/

Feature extractors, one per dimension. The `.py` modules are standalone: import and call the function on a text to get a flat `dict[str, float]`.

| File | Dimension | Entry point | What it computes |
| --- | --- | --- | --- |
| `D1_surfaceWordLevel 1.py` | D1 Surface, phonological, orthographic | `compute_d1(text)` | 11 features: word/syllable length and variation, polysyllabic ratio, espeak-ng IPA phonology (GPC regularity, consonant clusters, onset complexity), letter surprisal from Gutenberg |
| `D2_morphologicalComplexity 1.py` | D2 Morphology | `compute_all(text)` | TAMMI 2.0 (42, on MorphoLex-en), 3 spaCy inflection features, 15 legacy manual morpheme features |
| `D3_lexicalSophistication 1.py` | D3 Lexical sophistication | `compute_d3(text)` | 12 features: wordfreq Zipf bands, Oxford 5000 and Dale-Chall coverage, Gutenberg n-gram frequency and surprisal, Coltheart N |
| `D8_additional.ipynb` | D8 Discourse (AdSem) | notebook | LDA topic features on a Wikipedia background corpus (WSKF-style), with unit tests; writes `adsem_features_full(check).csv` |
| `D9.ipynb` | D9 Semantics, narrative, pragmatics | notebook | 25 features: MRC/Brysbaert norms, PINEAPPLE personification, Indian entity lexicon, idea density, sentiment, and XLM-R metaphor density |
| `miscellaneous.ipynb` | Affective + readability | notebook | 5 features from Warriner affective norms and textstat; writes `english_only_feature_full_PLUS5.xlsx` |

The " 1" in the `.py` filenames is a copy suffix; the spaces stop them being imported as-is, so rename them (e.g. `d1_surface_word_level.py`) before reuse. Features for D4–D7 and D10 have no extractor here; they arrive precomputed in the Excel files.

### papers/

22 reference PDFs on readability assessment, feature engineering and evaluation, e.g. neural pairwise ranking, perplexity features, ElasticNet stability selection, cross-validation, TF-IDF classification, the Velleman CEFR tool and a text-difficulty thesis. Two are duplicates (`l2_d.pdf` = `Text_Readability_Assessment_for_Second_Language_Le_d.pdf`), and `PDF Editor - … PDF Guru.pdf` has no extractable text, so its content is unknown.

### readability_english_15.11.26/

An exact backup of the root as of 15 Nov 2026 (same files, same sizes, no differences). It is about 70 MB, mostly the duplicate zip.

## Root files

### Notebooks

All three share Sections 0–7 (imports, config, data loading + textstat augmentation, feature taxonomy, density normalisation, outer CV, 3-stage feature selection, nested tuning). They differ in what follows.

| Notebook | Cells | What it adds after the shared sections | Inputs |
| --- | --- | --- | --- |
| `2-08-26-final-methodology (3).ipynb` | 98 | The final pipeline: dimension-level analysis, Cohen's d, baselines (trivial, B2, B4 LingFeat neural, B5 LingFeat + XGBoost, B6 LingFeat hybrid), tuned pipelines, cross-corpus validation (Telangana), 3-level NCERT classifier + OSE evaluation, dimension ablation and feature importance | NCERT xlsx, LingFeat xlsx, `onestop_coh.csv`, Telangana xlsx |
| `baseline-01-08-26 (3).ipynb` | 59 | Baselines only, with shared transformer code for B3, B5–B8; writes `baselines_final.csv` | NCERT xlsx, LingFeat xlsx |
| `upsampling (1).ipynb` | 51 | Resampling ablation (imblearn, smote_variants); writes `resampling_ablation.csv` | NCERT xlsx |

Start with `2-08-26-final-methodology (3).ipynb`; it is the source of the paper's results.

### Documents

| File | Dated | Content |
| --- | --- | --- |
| `EACL_Readability_Assistant_ARR_Aug.pdf` | Aug 2026 | Anonymous paper draft: an NCERT-calibrated tool for grade-levelling primary English texts and oral reading fluency passages |
| `Analysis (1).docx` | — | Method notes: per-word normalisation, z-scoring, ANOVA F and mutual-information ranking, CV voting for top-K features; 19 figures |
| `feature2.docx` | 5 Jan 2026 | Early feature brainstorm: limits of surface features, plans for language-model perplexity, entity density, lexical chains, coreference, entity grid |
| `ADDITIONAL FEATURES.docx` | 14 Mar 2026 | Literature notes on extra features and models, starting with BERT-FP-LBL (topic features, projection fusion, length-balanced loss) |
| `phonological_features.pdf` | — | Pinney et al., *Incorporating Word-level Phonemic Decoding into Readability Assessment*; the basis for D1's phonological features. Duplicate of `papers/phonetics.pdf`. |

### ncert_readability_experiments.zip (45 MB)

A full snapshot of the earlier experiment folder `ncert_readability/` (16 May – 7 Jun 2026), including its own `.git`. Unzip it to recover history and outputs.

- **Notebooks:** `ENGLISH_FINAL_PIPELINE.ipynb`, `methodology-18-05-26 (3).ipynb`, `readability-pipeline_v7_evs.ipynb`, `english_non_truncates.ipynb`, `baseline-ncert.ipynb`, `adsem_wiki.ipynb`, `d9_17_05_26.ipynb`, `missing_features_5feat_16_05_26.ipynb`.
- **Data:** both `data/` files, `english_final_truncated.xlsx` (truncated-text variant), `onestop_coh.csv` (OSE features), `dimesion.txt` (dimension list).
- **Result folders**, each with per-fold tuned models (`.pkl`), selected features per fold (`.json`), Cohen's d heatmaps, radar plots, error analysis and `unified_results.csv`:
  - `results_no_trunc/` — full (non-truncated) texts; also significance tests (bootstrap CI, McNemar, Wilcoxon), Telangana mapping and OSE results
  - `results_no_trunc_1/` — a rerun adding PCA Cohen's d and Spearman monotonicity
  - `results_evs_getting_added_check_no_cap/` — ensemble variants (rank average, weighted rank, loss diversity)
  - `results_no_trunc_3level/`, `results_no_trunc_3level_3level/` — 3-level classifier runs
- **Figures:** ablation plots (`ablation_*.png/pdf`), Cohen's d with and without truncation, CV fold diagrams (`fold.drawio`, `fold_1–3.png`, `s fold cv.png`), feature-selection diagram `fs.png`.

## Getting started and open issues

To reproduce the main results, run the final methodology notebook on Kaggle or Colab (GPU helps for the transformer baselines and D9 metaphor model).

1. Upload `data/english_supp_14.05.26_cleant.xlsx`, `data/final_telangana_dataset.xlsx`, `onestop_coh.csv` (from the zip) and the LingFeat baseline file as a Kaggle dataset.
2. Edit the paths in Section 1 (`CSV_PATH`, `LINGFEAT_PATH`, `OSE_PATH`, `TELANGANA_PATH`, `CKPT_DIR`) to point at your copies.
3. Install the packages Section 0 imports: pandas, numpy, scipy, scikit-learn, xgboost, lightgbm, mord, textstat, torch, transformers, sentencepiece, seaborn, matplotlib (plus imblearn and smote_variants for the upsampling notebook).
4. For new texts, run the extractors in `features/`. They also need spaCy, espeak-ng (D1), the `morphemes` package and MorphoLex (D2), wordfreq, NLTK Gutenberg and an Oxford 5000 list (D3), and gensim with a Wikipedia corpus (D8).

Open issues for whoever takes over:

- [ ] Make a first commit; nothing is under version control on `main` yet. Exclude the zip and the backup folder, or use Git LFS.
- [ ] Delete or archive `readability_english_15.11.26/` (exact duplicate).
- [ ] Locate the LingFeat baseline file; it is referenced by the notebooks but not in this folder.
- [ ] Add the missing extractors for D4–D7 and D10, or document which tool produced those columns.
- [ ] Replace hard-coded `/kaggle/input/...` and `/content/...` paths with relative paths.
- [ ] Rename files with spaces and copy suffixes (" 1", " (3)") and add a `requirements.txt`.
