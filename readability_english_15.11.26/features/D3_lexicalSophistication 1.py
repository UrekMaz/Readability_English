"""
D3 - Lexical Sophistication (standalone)
========================================

Self-contained module: drop into any project and call ``compute_d3(text)``
to obtain a flat ``dict[str, float]`` of 12 features.

Feature list
------------

  Frequency (wordfreq Zipf scale, log10 per-billion):
    log_word_frequency_zipf
    low_freq_ratio_zipf_under_2
    low_freq_ratio_zipf_under_3
    low_freq_ratio_zipf_under_4
    mean_zipf_below_median

  Vocabulary band coverage:
    oxford_5000_coverage          # Oxford 3000 + 5000 (lemma+POS, lemma fallback)
    dale_chall_3000_coverage      # textstat-bundled Dale-Chall easy list

  N-gram frequency (Gutenberg ~2.6M tokens, per-million):
    avg_bigram_freq_per_million
    avg_trigram_freq_per_million

  Surprisal (Gutenberg-conditioned, bits):
    lexical_surprisal_mean              # -log2 P(w) with add-1 smoothing
    forward_conditional_surprisal_mean  # -log2 P(w_t | w_{t-1})

  Orthographic neighbourhood (Coltheart 1977):
    coltheart_n_mean              # mean # of one-letter-substitution
                                  # neighbours of each token among
                                  # Gutenberg types occurring >= 5 times

No NCERT-corpus conditioning anywhere -- this file is fully shareable.
The reference corpus throughout is **NLTK Gutenberg** (Austen, Carroll,
Chesterton, Bible KJV, etc. -- ~2.6M tokens of British literary English),
matching D1's letter-surprisal source.

Dependencies
------------
Python:
    pip install spacy wordfreq nltk pandas openpyxl textstat

System / data:
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('gutenberg', quiet=True)"

Wordlist data files (Oxford 5000 vocabulary bands):
    Default path:  /Users/yashbhake/Desktop/Readability/data/wordlists/
                       oxford_3000.xlsx
                       oxford_5000_additional.xlsx
    Override by setting ``D3_lexicalSophistication._WORDLIST_DIR`` before
    first call to ``compute_d3``.

Usage
-----
    from D3_lexicalSophistication import compute_d3
    feats = compute_d3("Once upon a time there was a little girl...")
    # dict with 12 float entries

Bulk:
    import pandas as pd
    rows = [compute_d3(t) for t in all_stories]
    df = pd.DataFrame(rows)

The first call is slow (~5--15 s) while the Gutenberg corpus is loaded
and the unigram/bigram/trigram counts are built.  All subsequent calls
reuse the cached state and complete in <100 ms for typical NCERT-sized
stories.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import string
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import spacy
from wordfreq import zipf_frequency

# ---------------------------------------------------------------------------
# Singletons (cheap)
# ---------------------------------------------------------------------------

_NLP = spacy.load("en_core_web_sm")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_WORDLIST_DIR = Path("/Users/yashbhake/Desktop/Readability/data/wordlists")
_LETTERS = string.ascii_lowercase


# ---------------------------------------------------------------------------
# Oxford 5000 (3000 base + 2000 additional)
# ---------------------------------------------------------------------------

def _load_oxford(path: Path) -> tuple[dict[tuple[str, str], str], set[str]]:
    df = pd.read_excel(path)
    lookup: dict[tuple[str, str], str] = {}
    word_set: set[str] = set()
    for _, row in df.iterrows():
        word = str(row["Word"]).strip().lower()
        word_set.add(word)
        try:
            pos_cefr = json.loads(row["POS : CEFR"])
            for pos_label, cefr in pos_cefr.items():
                lookup[(word, pos_label)] = cefr
        except Exception:
            pass
    return lookup, word_set


_OX3_LOOKUP, _OX3_WORDS = _load_oxford(_WORDLIST_DIR / "oxford_3000.xlsx")
_OX5_LOOKUP, _OX5_WORDS = _load_oxford(_WORDLIST_DIR / "oxford_5000_additional.xlsx")
_OX5_FULL_LOOKUP = {**_OX3_LOOKUP, **_OX5_LOOKUP}
_OX5_FULL_WORDS = _OX3_WORDS | _OX5_WORDS

_SPACY_POS_TO_OXFORD = {
    "NOUN": "n.", "VERB": "v.", "ADJ": "adj.", "ADV": "adv.",
    "ADP": "prep.", "DET": "det.", "PRON": "pron.",
    "CCONJ": "conj.", "SCONJ": "conj.",
    "NUM": "number", "INTJ": "exclam.", "AUX": "v.",
    "PART": "infinitive marker",
}


# ---------------------------------------------------------------------------
# Dale-Chall easy 3000
# ---------------------------------------------------------------------------

def _load_dale_chall() -> set[str]:
    import textstat as _ts
    p = Path(_ts.__file__).parent / "resources" / "en" / "easy_words.txt"
    return set(p.read_text().strip().splitlines())


_DALE_CHALL_WORDS = _load_dale_chall()


# ---------------------------------------------------------------------------
# Gutenberg corpus state (unigram / bigram / trigram counts + lexicon set)
#
# Built once, lazily, on first feature request.  Approx. 2.6 M tokens of
# British literary English.  Build time ~6 s; memory ~120 MB.
# ---------------------------------------------------------------------------


class _GutenbergCorpus:

    def __init__(self):
        self._uni: Optional[Counter] = None
        self._bi: Optional[Counter] = None
        self._tri: Optional[Counter] = None
        self._lex: Optional[set] = None     # types >=5 occurrences
        self._N: int = 0                    # total tokens
        self._V: int = 0                    # vocabulary size
        self._N_bi: int = 0
        self._N_tri: int = 0

    def _ensure(self):
        if self._uni is not None:
            return
        import nltk
        try:
            from nltk.corpus import gutenberg
            gutenberg.words()
        except LookupError:
            nltk.download("gutenberg", quiet=True)
            from nltk.corpus import gutenberg
        tokens = [w.lower() for w in gutenberg.words() if w.isalpha()]
        self._N = len(tokens)
        self._uni = Counter(tokens)
        self._bi = Counter(zip(tokens, tokens[1:]))
        self._tri = Counter(zip(tokens, tokens[1:], tokens[2:]))
        self._N_bi = sum(self._bi.values())
        self._N_tri = sum(self._tri.values())
        self._V = len(self._uni)
        # Coltheart lexicon: types occurring >=5 times (filters typos /
        # archaic one-offs).  Final size ~12-15 K.
        self._lex = {w for w, c in self._uni.items() if c >= 5}

    # ---- unigram surprisal (-log2 P(w)) with add-1 smoothing ----------
    def lexical_surprisal(self, word: str) -> float:
        self._ensure()
        c = self._uni.get(word, 0)
        # P(w) = (c + 1) / (N + V); surprisal = -log2 P(w)
        return -math.log2((c + 1) / (self._N + self._V))

    # ---- forward bigram surprisal (-log2 P(w_t | w_{t-1})) -----------
    def forward_conditional_surprisal(self, prev: str, curr: str) -> float:
        self._ensure()
        bi_c = self._bi.get((prev, curr), 0)
        prev_c = self._uni.get(prev, 0)
        # P(curr|prev) = (bi_c + 1) / (prev_c + V)  Laplace smoothing
        return -math.log2((bi_c + 1) / (prev_c + self._V))

    # ---- raw n-gram counts, per-million normalised --------------------
    def avg_bigram_per_million(self, bigrams: list[tuple[str, str]]) -> float:
        self._ensure()
        if not bigrams:
            return 0.0
        scale = 1_000_000 / max(1, self._N_bi)
        return statistics.mean(self._bi.get(bg, 0) * scale for bg in bigrams)

    def avg_trigram_per_million(self, trigrams: list[tuple[str, str, str]]) -> float:
        self._ensure()
        if not trigrams:
            return 0.0
        scale = 1_000_000 / max(1, self._N_tri)
        return statistics.mean(self._tri.get(tg, 0) * scale for tg in trigrams)

    @property
    def lex(self) -> set:
        self._ensure()
        return self._lex


_GUT = _GutenbergCorpus()


# ---------------------------------------------------------------------------
# Coltheart's N (orthographic neighbourhood density, Coltheart 1977)
#
# For each target word, count the number of words in the reference
# lexicon formed by substituting exactly one letter.  Cached because
# stories repeat words.  Bounded by 25 * len(word) lookups -- cheap.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=200_000)
def _coltheart_n(word: str) -> int:
    """Number of one-letter-substitution neighbours of *word* in the
    Gutenberg lexicon (types with frequency >= 5).  Original (target)
    word is excluded.  Returns 0 for words too short (< 2) or non-alpha.
    """
    w = word.lower()
    if not w.isalpha() or len(w) < 2:
        return 0
    lex = _GUT.lex
    count = 0
    for i in range(len(w)):
        original = w[i]
        prefix, suffix = w[:i], w[i + 1:]
        for ch in _LETTERS:
            if ch == original:
                continue
            if (prefix + ch + suffix) in lex:
                count += 1
    return count


# ---------------------------------------------------------------------------
# spaCy helpers
# ---------------------------------------------------------------------------


def _ner_filtered(doc):
    """Return tokens that are alphabetic and NOT named-entity / proper noun.
    Proper nouns are excluded so Indian-name-heavy NCERT prose does not
    skew vocabulary stats."""
    return [tok for tok in doc
            if tok.is_alpha and not tok.ent_type_ and tok.pos_ != "PROPN"]


def _oxford_pos(tok) -> str:
    return _SPACY_POS_TO_OXFORD.get(tok.pos_, "")


# ===========================================================================
# Public API
# ===========================================================================


D3_FEATURES: tuple[str, ...] = (
    # Frequency (Zipf)
    "log_word_frequency_zipf",
    "low_freq_ratio_zipf_under_2",
    "low_freq_ratio_zipf_under_3",
    "low_freq_ratio_zipf_under_4",
    "mean_zipf_below_median",
    # Vocabulary band
    "oxford_5000_coverage",
    "dale_chall_3000_coverage",
    # N-gram
    "avg_bigram_freq_per_million",
    "avg_trigram_freq_per_million",
    # Surprisal
    "lexical_surprisal_mean",
    "forward_conditional_surprisal_mean",
    # Orthographic neighbourhood
    "coltheart_n_mean",
)


def _zero_features() -> dict[str, float]:
    return {k: 0.0 for k in D3_FEATURES}


def compute_d3(text: str) -> dict[str, float]:
    """Return all 12 D3 features for *text* as a flat dict.

    Stateless apart from the lazy Gutenberg-corpus / wordlist load
    cached at module level.  No corpus-specific fitting required.
    """
    doc = _NLP(text)
    tokens = _ner_filtered(doc)
    if not tokens:
        return _zero_features()

    lemmas  = [tok.lemma_.lower() for tok in tokens]
    surface = [tok.text.lower()  for tok in tokens]

    # ---- Frequency (wordfreq Zipf) ----
    zipfs = [zipf_frequency(lem, "en") for lem in lemmas]
    avg_zipf = statistics.mean(zipfs)
    n = len(tokens)
    lf2 = sum(1 for z in zipfs if z < 2.0) / n
    lf3 = sum(1 for z in zipfs if z < 3.0) / n
    lf4 = sum(1 for z in zipfs if z < 4.0) / n
    median = statistics.median(zipfs)
    below = [z for z in zipfs if z <= median]
    mean_zipf_below = statistics.mean(below) if below else 0.0

    # ---- Vocabulary band coverage ----
    ox5_count = 0
    for tok, lem in zip(tokens, lemmas):
        ox_pos = _oxford_pos(tok)
        if (lem, ox_pos) in _OX5_FULL_LOOKUP or lem in _OX5_FULL_WORDS:
            ox5_count += 1
    ox5_cov = ox5_count / n
    dale_count = sum(
        1 for s, lem in zip(surface, lemmas)
        if s in _DALE_CHALL_WORDS or lem in _DALE_CHALL_WORDS
    )
    dale_cov = dale_count / n

    # ---- N-gram corpus frequencies (per million, Gutenberg) ----
    bigrams  = list(zip(surface, surface[1:]))
    trigrams = list(zip(surface, surface[1:], surface[2:]))
    bi_pm  = _GUT.avg_bigram_per_million(bigrams)
    tri_pm = _GUT.avg_trigram_per_million(trigrams)

    # ---- Surprisal (Gutenberg-conditioned) ----
    # Unigram surprisal averaged over non-entity content words.
    lex_sur = [
        _GUT.lexical_surprisal(s) for s in surface
    ]
    lexical_sur_mean = statistics.mean(lex_sur) if lex_sur else 0.0

    # Forward conditional surprisal: P(w_t | w_{t-1}) over adjacent pairs.
    fwd_surs: list[float] = []
    for prev, curr in zip(surface, surface[1:]):
        fwd_surs.append(_GUT.forward_conditional_surprisal(prev, curr))
    fwd_sur_mean = statistics.mean(fwd_surs) if fwd_surs else 0.0

    # ---- Coltheart's N (orthographic neighbourhood) ----
    n_values = [_coltheart_n(s) for s in surface]
    coltheart_mean = statistics.mean(n_values) if n_values else 0.0

    return {
        "log_word_frequency_zipf":            avg_zipf,
        "low_freq_ratio_zipf_under_2":        lf2,
        "low_freq_ratio_zipf_under_3":        lf3,
        "low_freq_ratio_zipf_under_4":        lf4,
        "mean_zipf_below_median":             mean_zipf_below,
        "oxford_5000_coverage":               ox5_cov,
        "dale_chall_3000_coverage":           dale_cov,
        "avg_bigram_freq_per_million":        bi_pm,
        "avg_trigram_freq_per_million":       tri_pm,
        "lexical_surprisal_mean":             lexical_sur_mean,
        "forward_conditional_surprisal_mean": fwd_sur_mean,
        "coltheart_n_mean":                   coltheart_mean,
    }


# Alias for back-compat
compute_all = compute_d3


# ===========================================================================
# Smoke tests
# ===========================================================================


if __name__ == "__main__":
    import time
    sample = ("Once there was a little girl named Tara. She lived with her "
              "mother. They were very poor. One day she went to the forest. "
              "There she met an old woman who gave her a magic pot.")

    # First call: triggers Gutenberg corpus build (~5-10 s).
    t0 = time.time()
    feats = compute_d3(sample)
    print(f"first call: {time.time() - t0:.2f}s  (Gutenberg corpus build)")
    print()
    for k, v in feats.items():
        print(f"  {k:38s} {v:.4f}")

    print()
    # Second call: cached.
    t0 = time.time()
    feats = compute_d3(sample)
    print(f"second call (cached): {time.time() - t0:.3f}s")
