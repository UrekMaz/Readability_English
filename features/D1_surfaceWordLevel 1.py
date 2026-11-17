"""
D1 - Surface, Phonological & Orthographic Features (standalone)
===============================================================

Self-contained module: drop into any project and call ``compute_d1(text)``
to obtain a flat ``dict[str, float]`` of 11 features.

Feature list
------------

  Length statistics:
    avg_word_length_chars
    cv_word_length_chars        # coefficient of variation = sd/mean
    avg_syllables_per_word
    cv_syllables_per_word

  Type-level:
    type_polysyllabic_ratio     # over unique types, syllables >= 3

  Phonological complexity (espeak-ng IPA):
    gpc_regularity              # mean |1 - n_alpha_chars / n_phonemes|
    consonant_cluster_density   # IPA clusters of length >= 2
    hard_consonant_cluster_ratio# proportion of words with IPA cluster >= 3
    onset_complexity_mean       # mean onset size per syllable
    syllable_complexity         # open-syllable ratio (sonority-aware split)

  Orthographic complexity:
    letter_surprisal            # interpolated bigram+trigram letter
                                # surprisal trained on NLTK Gutenberg
                                # (British literary corpus, ~2.6M tokens)

IPA generation
--------------
All IPA-based features use **phonemizer** with the **espeak-ng** backend
in *British English (`en-gb`)*.  Stress markers are stripped.
There is currently no `en-IN` voice in espeak-ng; the closest free
phonemiser for Indian English does not exist (see README in the
companion notebook).

Why British? D1 features are corpus-conditioned only on the *letter
n-gram model* (used for ``letter_surprisal``), which is built from
NLTK's Gutenberg corpus — Austen, Carroll, Chesterton, KJV, etc., i.e.
British literary English.  Phonemisation likewise uses en-gb.  There is
*no* NCERT-corpus conditioning anywhere in this module — D1 is text-only
and shareable without any project-internal state.

Dependencies
------------
System:
    espeak-ng                       # brew install espeak-ng    (macOS)
                                    # apt install espeak-ng    (Linux)

Python:
    pip install phonemizer pyphen nltk

NLTK data (one-time):
    python -c "import nltk; nltk.download('gutenberg', quiet=True)"

Usage
-----
    from D1_surfaceWordLevel import compute_d1
    features = compute_d1("The old woodcutter trudged into the dark forest.")
    # features is a dict with 11 float entries

Bulk:
    import pandas as pd
    rows = [compute_d1(t) for t in all_stories]
    df = pd.DataFrame(rows)
"""
from __future__ import annotations

import math
import os
import re
import statistics
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# espeak-ng dylib auto-discovery (macOS Homebrew installs to /opt/homebrew)
# ---------------------------------------------------------------------------

if "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
    for _cand in (
        "/opt/homebrew/lib/libespeak-ng.dylib",
        "/opt/homebrew/lib/libespeak-ng.1.dylib",
        "/usr/local/lib/libespeak-ng.dylib",
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
    ):
        if Path(_cand).exists():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = _cand
            break

import pyphen
from phonemizer import phonemize
from phonemizer.separator import Separator

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_PYPHEN = pyphen.Pyphen(lang="en_GB")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# IPA inventories (espeak-ng en-gb output)
_IPA_VOWEL_CHARS = set("iɪeɛæaɑɒɔəɜʊuʌ")
_IPA_VOWELS = set(
    "i iː ɪ e ɛ eɪ æ a aɪ aʊ ɑː "
    "ɒ ɔː ɔɪ ə əʊ ɜː ʊ "
    "uː ʌ".split()
)

# Sonority scale for English IPA consonants (higher = more sonorous).
_SONORITY = {
    "p": 1, "b": 1, "t": 1, "d": 1, "k": 1, "g": 1, "ɡ": 1,
    "tʃ": 1, "dʒ": 1,
    "f": 2, "v": 2, "θ": 2, "ð": 2, "s": 2, "z": 2,
    "ʃ": 2, "ʒ": 2, "h": 2,
    "m": 3, "n": 3, "ŋ": 3,
    "l": 4, "ɹ": 4, "r": 4, "ʁ": 4, "ɾ": 4,
    "w": 5, "j": 5,
}


def _tokenise(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _is_vowel_phone(phone: str) -> bool:
    if phone in _IPA_VOWELS:
        return True
    return bool(phone) and phone[0] in _IPA_VOWEL_CHARS


def _is_consonant_phone(phone: str) -> bool:
    return bool(phone) and not _is_vowel_phone(phone)


def _sonority(p: str) -> int:
    return _SONORITY.get(p, 2)


def _legal_onset(cluster: list[str]) -> bool:
    """Sonority-rising up to the nucleus; English caps onsets at 3 (str-,
    spl-, spr-, ...) — we approximate with sonority sequencing."""
    if len(cluster) <= 1:
        return True
    if len(cluster) > 3:
        return False
    for i in range(1, len(cluster)):
        if _sonority(cluster[i]) < _sonority(cluster[i - 1]):
            return False
    return True


def _syllabify_phones(phones: list[str]) -> list[list[str]]:
    """Sonority-aware Maximum Onset Principle on the IPA phone sequence."""
    if not phones:
        return []
    vowel_idx = [i for i, p in enumerate(phones) if _is_vowel_phone(p)]
    if not vowel_idx:
        return [phones]
    boundaries = [0]
    for k in range(1, len(vowel_idx)):
        prev_v, curr_v = vowel_idx[k - 1], vowel_idx[k]
        cluster = phones[prev_v + 1:curr_v]
        nC = len(cluster)
        if nC == 0:
            split = prev_v + 1
        else:
            split = curr_v - nC
            for onset_size in range(min(nC, 3), 0, -1):
                cand = cluster[nC - onset_size:]
                if _legal_onset(cand):
                    split = curr_v - onset_size
                    break
            else:
                split = curr_v
        boundaries.append(split)
    boundaries.append(len(phones))
    return [phones[boundaries[i]:boundaries[i + 1]]
            for i in range(len(boundaries) - 1)]


# ---------------------------------------------------------------------------
# Phonemiser (espeak-ng en-gb)
#
# Used by: gpc_regularity, consonant_cluster_density,
#          hard_consonant_cluster_ratio, onset_complexity_mean,
#          syllable_complexity.
#
# Falls back to empty phone-lists if espeak-ng isn't installed; the
# IPA-derived features then return 0.0 with a one-time warning.
# ---------------------------------------------------------------------------

_ESPEAK_AVAILABLE: Optional[bool] = None


def _phonemize_words(words: list[str]) -> list[list[str]]:
    global _ESPEAK_AVAILABLE
    if not words:
        return []
    if _ESPEAK_AVAILABLE is False:
        return [[] for _ in words]
    try:
        sep = Separator(phone=" ", word="")
        raw = phonemize(
            "\n".join(words), language="en-gb", backend="espeak",
            strip=True, with_stress=False, separator=sep,
            language_switch="remove-flags",
        )
    except (RuntimeError, OSError) as exc:
        if _ESPEAK_AVAILABLE is None:
            print(f"[D1] espeak-ng unavailable ({exc}); IPA features "
                  f"will return 0.0.  Install: brew install espeak-ng")
        _ESPEAK_AVAILABLE = False
        return [[] for _ in words]
    _ESPEAK_AVAILABLE = True
    out = [
        [p for p in line.strip().split() if p]
        for line in raw.strip().split("\n")
    ]
    while len(out) < len(words):
        out.append([])
    return out


# ---------------------------------------------------------------------------
# Letter n-gram model (Gutenberg = British literary, ~2.6M tokens)
# ---------------------------------------------------------------------------


class _GutenbergLetterModel:
    """Interpolated bigram + trigram letter model with add-1 smoothing.

    Trained lazily on first ``avg_surprisal`` call against the NLTK
    Gutenberg corpus.  Approx. 2.6M tokens of British literary English
    (Austen, Carroll, Chesterton, Bible KJV, etc.).
    """

    def __init__(self):
        self._uni: Optional[Counter] = None
        self._bi: dict = {}
        self._tri: dict = {}
        self._V: int = 1

    def _ensure_corpus(self):
        import nltk
        try:
            from nltk.corpus import gutenberg
            gutenberg.words()
        except LookupError:
            nltk.download("gutenberg", quiet=True)

    def _build(self):
        if self._uni is not None:
            return
        self._ensure_corpus()
        from nltk.corpus import gutenberg
        words = [w.lower() for w in gutenberg.words() if w.isalpha()]
        uni, bi, tri = Counter(), Counter(), Counter()
        for w in words:
            padded = "^" + w + "$"
            for ch in padded:
                uni[ch] += 1
            for i in range(len(padded) - 1):
                bi[padded[i:i + 2]] += 1
            for i in range(len(padded) - 2):
                tri[padded[i:i + 3]] += 1
        V = max(1, len(uni))
        self._V = V
        self._uni = uni
        self._bi = {bg: math.log2((c + 1) / (uni[bg[0]] + V))
                    for bg, c in bi.items()}
        self._tri = {tg: math.log2((c + 1) / (bi.get(tg[:2], 0) + V))
                     for tg, c in tri.items()}

    def avg_surprisal(self, text: str) -> float:
        self._build()
        words = _tokenise(text)
        if not words:
            return 0.0
        floor = math.log2(1 / self._V)
        total, n = 0.0, 0
        for w in words:
            padded = "^" + w.lower() + "$"
            for i in range(1, len(padded)):
                bg = padded[i - 1:i + 1]
                bi_lp = self._bi.get(bg, floor)
                if i >= 2:
                    tg = padded[i - 2:i + 1]
                    tri_lp = self._tri.get(tg, floor)
                    lp = 0.5 * bi_lp + 0.5 * tri_lp
                else:
                    lp = bi_lp
                total += -lp
                n += 1
        return total / n if n else 0.0


_LETTER_MODEL = _GutenbergLetterModel()


# ---------------------------------------------------------------------------
# Length statistics  (avg, CV) for chars and syllables
# ---------------------------------------------------------------------------


@lru_cache(maxsize=200_000)
def _syllable_count(word: str) -> int:
    """Pyphen-based syllable count (orthographic hyphenation).  Used for
    the legacy length features (avg_syllables_per_word, cv_*) so that
    those stay backward-compatible."""
    return max(1, len(_PYPHEN.inserted(word.lower()).split("-")))


def _ipa_syllable_count(phones: list[str]) -> int:
    """Count vowel phonemes in the IPA sequence.  This is the
    *phonological* syllable count and is more accurate than pyphen for
    words like 'rhythmic' (orthographic 1 vs phonological 2) or
    'forest' (orth 1 vs phon 2)."""
    return max(1, sum(1 for p in phones if _is_vowel_phone(p)))


def _coef_var(xs: list[int]) -> float:
    """Coefficient of variation = sd / mean.  Returns 0 if mean == 0."""
    if not xs:
        return 0.0
    mean = statistics.mean(xs)
    if mean == 0:
        return 0.0
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return sd / mean


# ===========================================================================
# Public API
# ===========================================================================


D1_FEATURES: tuple[str, ...] = (
    "avg_word_length_chars",
    "cv_word_length_chars",
    "avg_syllables_per_word",
    "cv_syllables_per_word",
    "type_polysyllabic_ratio",
    "mean_chars_per_syllable",          # NEW: orthographic-depth proxy
    "gpc_regularity",
    "consonant_cluster_density",
    "hard_consonant_cluster_ratio",
    "onset_complexity_mean",
    "syllable_complexity",
    "letter_surprisal",
)


def compute_d1(text: str) -> dict[str, float]:
    """Return all 11 D1 features for *text* as a flat dict.

    The function is stateless and corpus-independent: no NCERT-specific
    fitting required.
    """
    words = _tokenise(text)
    if not words:
        return {k: 0.0 for k in D1_FEATURES}

    char_lens = [len(w) for w in words]
    syll_counts = [_syllable_count(w) for w in words]

    types = {w.lower() for w in words}
    poly_types = sum(1 for t in types if _syllable_count(t) >= 3)
    type_poly = poly_types / len(types) if types else 0.0

    # ---- shared phonemisation ----
    phone_lists = _phonemize_words(words)

    devs: list[float] = []
    cc_total = 0
    hard_hits = 0
    syll_open = syll_total = 0
    onsets: list[int] = []

    nW = len(words)
    for w, phones in zip(words, phone_lists):
        if not phones:
            continue
        # gpc regularity
        n_char = sum(1 for c in w.lower() if c.isalpha())
        devs.append(abs(1.0 - n_char / len(phones)))
        # clusters
        consec, c2, hit3 = 0, 0, False
        for p in phones:
            if _is_consonant_phone(p):
                consec += 1
            else:
                if consec >= 2: c2 += 1
                if consec >= 3: hit3 = True
                consec = 0
        if consec >= 2: c2 += 1
        if consec >= 3: hit3 = True
        cc_total += c2
        if hit3:
            hard_hits += 1
        # syllabification (Maximum Onset, sonority-aware)
        for syl in _syllabify_phones(phones):
            if not syl:
                continue
            syll_total += 1
            if _is_vowel_phone(syl[-1]):
                syll_open += 1
            onset_n = 0
            for p in syl:
                if _is_vowel_phone(p):
                    break
                onset_n += 1
            onsets.append(onset_n)

    # chars-per-syllable: orthographic-depth proxy.  Regular English is
    # ~3.0; words with cps >> 3 contain digraphs and/or silent letters.
    # Use IPA-vowel count when phones are available (phonologically more
    # accurate than pyphen's orthographic hyphenation) — falls back to
    # pyphen if espeak-ng is unavailable.
    cps_per_word = []
    for w, c, phones in zip(words, char_lens, phone_lists):
        s = _ipa_syllable_count(phones) if phones else _syllable_count(w)
        if s > 0:
            cps_per_word.append(c / s)

    return {
        "avg_word_length_chars":        statistics.mean(char_lens),
        "cv_word_length_chars":         _coef_var(char_lens),
        "avg_syllables_per_word":       statistics.mean(syll_counts),
        "cv_syllables_per_word":        _coef_var(syll_counts),
        "type_polysyllabic_ratio":      type_poly,
        "mean_chars_per_syllable":      statistics.mean(cps_per_word) if cps_per_word else 0.0,
        "gpc_regularity":               statistics.mean(devs) if devs else 0.0,
        "consonant_cluster_density":    cc_total / nW,
        "hard_consonant_cluster_ratio": hard_hits / nW,
        "onset_complexity_mean":        statistics.mean(onsets) if onsets else 0.0,
        "syllable_complexity":          syll_open / syll_total if syll_total else 0.0,
        "letter_surprisal":             _LETTER_MODEL.avg_surprisal(text),
    }


# Alias for backward compatibility with the previous module API.
compute_all = compute_d1


# ---------------------------------------------------------------------------
# Diagnostic: per-word GPC breakdown
# ---------------------------------------------------------------------------


SHORT_WORD_THRESHOLD = 3  # words with n_chars <= 3 are too short to flag


def gpc_per_word(text: str) -> list[dict]:
    """Per-word orthographic-complexity diagnostics.

    The previous ``gpc_deviation = |1 - n_chars/n_phonemes|`` signal is
    visually noisy because espeak-ng emits long vowels (``iː``) and
    digraphs (``ʃ``, ``ð``, ``tʃ``) as single phonemes -- so perfectly
    regular short words like *she* (3 chars / 2 phones) and *they*
    (4 chars / 2 phones) flag highly irregular.  Those are not actually
    hard to decode.

    The cleaner per-word irregularity signal, especially when visualising
    in the notebook, is **chars per syllable**.  English averages ~3.0
    chars/syllable; words at 4+ chars/syllable nearly always contain
    silent letters or unusual orthographic patterns (knight, through,
    yacht, Wednesday) -- the genuinely hard cases for L2 learners.

    Returned per-word dict has:
        word               - original surface form
        n_chars            - alphabetic character count
        n_phonemes         - espeak-ng en-gb IPA phone count (may be 0)
        n_syllables        - pyphen en_GB syllable count
        ipa                - space-separated IPA string
        chars_per_syllable - orthographic-depth proxy (lower=more regular)
        gpc_deviation      - legacy |1 - n_chars/n_phonemes| (kept for
                             back-compat; not recommended for viz)
        is_short           - True for n_chars <= SHORT_WORD_THRESHOLD
                             (visualisation should skip these)
    """
    words = _tokenise(text)
    phone_lists = _phonemize_words(words)
    out: list[dict] = []
    for w, phones in zip(words, phone_lists):
        n_phone = len(phones)
        n_char = sum(1 for c in w.lower() if c.isalpha())
        n_syll = (_ipa_syllable_count(phones) if phones
                  else _syllable_count(w))
        cps = n_char / n_syll if n_syll else 0.0
        ppc = n_phone / n_char if n_char else 0.0
        entry = {
            "word": w,
            "n_chars": n_char,
            "n_phonemes": n_phone,
            "n_syllables": n_syll,
            "syll_source": "ipa" if phones else "pyphen",
            "ipa": " ".join(phones),
            "chars_per_syllable": cps,
            "phonemes_per_char": ppc,
            "is_short": n_char <= SHORT_WORD_THRESHOLD,
        }
        entry["gpc_deviation"] = (
            abs(1.0 - n_char / n_phone) if n_phone else None
        )
        out.append(entry)
    return out


if __name__ == "__main__":
    sample = ("The old woodcutter had lived at the edge of the enchanted "
              "forest for as long as anyone could remember.  He would rise "
              "before dawn, sharpen his axe, and trudge into the darkness.")
    for k, v in compute_d1(sample).items():
        print(f"  {k:32s} {v:.4f}")
