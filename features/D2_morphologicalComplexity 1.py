"""
D2 - Morphological Complexity (TAMMI + legacy manual features)
==============================================================

This module exposes **three** feature families on the same input text:

1. **TAMMI 2.0** (Crossley & MacArthur, linguisticanalysistools.org) — 42
   features built on the MorphoLex-en dataframe (Sánchez-Gutiérrez et
   al. 2017):

     - Basic morpheme counts (11)
     - MorphoLex per-cw means (25):  family size, frequency,
       log-frequency, length, in-hapax, and PFMF for
       {prefix, root, suffix, all-affix}.
     - Morphological complexity (6):  mean subset variety, TTR, and
       MCI (Brezina & Pallotti 2019) over 10-content-word windows.

2. **spaCy supplements** (3) — inflectional features TAMMI does not
   compute:

     irregular_verb_density_per_100w
     comparative_superlative_density
     inflection_class_entropy

3. **Legacy manual features** (15 unique, re-introduced) — the
   pre-TAMMI hand-crafted inventory.  Computed locally via the
   ``morphemes`` package (MorphoLex-backed parser with a compound-
   split fallback) and spaCy.  Some overlap with TAMMI / spaCy is
   accepted by design.

     avg_morphemes_per_word
     total_free_morphemes
     total_bound_morphemes
     morpheme_type_token_ratio
     derivational_suffix_density
     prefix_density
     derivational_depth
     morpheme_family_size            (aliased to TAMMI root_family_size_per_cw)
     inflected_form_ratio
     verb_inflection_diversity
     irregular_verb_ratio
     compound_word_density
     hyphenated_compound_density
     polymorph_ratio
     max_morpheme_count

Total D2 feature count: **60** (42 TAMMI + 3 spaCy + 15 manual).
``comparative_superlative_density`` appears once and is shared.

Dependencies
------------
    pip install spacy morphemes
    python -m spacy download en_core_web_sm

Morpheme database (~10 MB) is downloaded automatically by the
``morphemes`` package on first use.
"""
from __future__ import annotations

import math
import re
import tempfile
from collections import Counter
from functools import lru_cache
from statistics import mean
from typing import Optional

import spacy

from .tammi_features import compute_tammi, TAMMI_FEATURES


# ---------------------------------------------------------------------------
# Singletons (spaCy is reused for the supplemental inflectional features)
# ---------------------------------------------------------------------------

_NLP = spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Irregular-verb lexicon (Coxhead-adjacent pedagogical inventory, ~180 verbs)
# ---------------------------------------------------------------------------

_IRREGULAR_BASE_FORMS = {
    "arise","awake","be","bear","beat","become","begin","bend","bet","bind",
    "bite","bleed","blow","break","breed","bring","build","burn","burst",
    "buy","catch","choose","cling","come","cost","creep","cut","deal","dig",
    "do","draw","dream","drink","drive","eat","fall","feed","feel","fight",
    "find","flee","fling","fly","forbid","forget","forgive","freeze","get",
    "give","go","grind","grow","hang","have","hear","hide","hit","hold",
    "hurt","keep","kneel","knit","know","lay","lead","lean","leap","learn",
    "leave","lend","let","lie","light","lose","make","mean","meet","mow",
    "overcome","pay","prove","put","quit","read","rid","ride","ring","rise",
    "run","say","see","seek","sell","send","set","sew","shake","shed",
    "shine","shoot","show","shrink","shut","sing","sink","sit","sleep",
    "slide","sling","smell","sow","speak","speed","spell","spend","spill",
    "spin","spit","split","spread","spring","stand","steal","stick","sting",
    "stink","strike","strive","swear","sweep","swim","swing","take","teach",
    "tear","tell","think","throw","understand","wake","wear","weave","weep",
    "win","wind","withdraw","wring","write",
}
_IRREGULAR_PAST_PP = {
    "arose","arisen","awoke","awoken","was","were","been","bore","borne",
    "beat","beaten","became","begun","began","bent","bet","bound","bit",
    "bitten","bled","blew","blown","broke","broken","bred","brought",
    "built","burnt","burst","bought","caught","chose","chosen","clung",
    "came","cost","crept","cut","dealt","dug","did","done","drew","drawn",
    "dreamt","drank","drunk","drove","driven","ate","eaten","fell","fallen",
    "fed","felt","fought","found","fled","flung","flew","flown","forbade",
    "forbidden","forgot","forgotten","forgave","forgiven","froze","frozen",
    "got","gotten","gave","given","went","gone","ground","grew","grown",
    "hung","had","heard","hid","hidden","hit","held","hurt","kept","knelt",
    "knit","knew","known","laid","led","leant","leapt","learnt","left",
    "lent","let","lay","lain","lit","lost","made","meant","met","mowed",
    "mown","overcame","paid","proved","proven","put","quit","read","rid",
    "rode","ridden","rang","rung","rose","risen","ran","said","saw","seen",
    "sought","sold","sent","set","sewed","sewn","shook","shaken","shed",
    "shone","shot","showed","shown","shrank","shrunk","shut","sang","sung",
    "sank","sunk","sat","slept","slid","slung","smelt","sowed","sown",
    "spoke","spoken","sped","spelt","spent","spilt","spun","spat","split",
    "spread","sprang","sprung","stood","stole","stolen","stuck","stung",
    "stank","stunk","struck","strove","striven","swore","sworn","swept",
    "swam","swum","swung","took","taken","taught","tore","torn","told",
    "thought","threw","thrown","understood","woke","woken","wore","worn",
    "wove","woven","wept","won","wound","withdrew","withdrawn","wrung",
    "wrote","written",
}


def _supplemental_features(text: str) -> dict[str, float]:
    """Three inflectional features TAMMI does not compute."""
    doc = _NLP(text)
    content = [tok for tok in doc
               if tok.is_alpha and not tok.is_punct and not tok.is_space]
    n_words = sum(1 for _ in doc if _.is_alpha)

    # 1) Irregular-verb density per 100 words
    irregular = 0
    for tok in content:
        if tok.pos_ not in ("VERB", "AUX"):
            continue
        form, lemma = tok.text.lower(), tok.lemma_.lower()
        if (lemma in _IRREGULAR_BASE_FORMS and form in _IRREGULAR_PAST_PP) \
                or form in _IRREGULAR_PAST_PP:
            irregular += 1
    irregular_density = 100.0 * irregular / n_words if n_words else 0.0

    # 2) Comparative / superlative density
    adj_adv = [tok for tok in doc
               if tok.tag_ in ("JJ", "JJR", "JJS", "RB", "RBR", "RBS")
               and tok.is_alpha]
    comp_sup = sum(1 for tok in adj_adv
                   if tok.tag_ in ("JJR", "JJS", "RBR", "RBS"))
    comp_sup_density = comp_sup / len(adj_adv) if adj_adv else 0.0

    # 3) Inflection-class entropy over tok.morph tuples
    classes: list[tuple] = []
    for tok in content:
        if tok.pos_ not in ("NOUN", "VERB", "AUX", "ADJ", "ADV", "PROPN"):
            continue
        feats = tok.morph.to_dict()
        if feats:
            classes.append(tuple(sorted(feats.items())))
    if classes:
        counts = Counter(classes)
        total = sum(counts.values())
        entropy = -sum((c / total) * math.log2(c / total)
                       for c in counts.values())
    else:
        entropy = 0.0

    return {
        "irregular_verb_density_per_100w": irregular_density,
        "comparative_superlative_density": comp_sup_density,
        "inflection_class_entropy": entropy,
    }


# ---------------------------------------------------------------------------
# Legacy manual D2 features (re-introduced alongside TAMMI; some overlap
# with TAMMI's root_* / Compounds metrics is intentional).
# ---------------------------------------------------------------------------

_MORPH_PARSER = None
_HYPHENATED_RE = re.compile(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b")


def _get_morph_parser():
    global _MORPH_PARSER
    if _MORPH_PARSER is None:
        from morphemes import Morphemes
        _MORPH_PARSER = Morphemes(tempfile.gettempdir())
    return _MORPH_PARSER


def _flatten_morph_tree(tree: list) -> list[tuple[str, str]]:
    """Flatten the morphemes-package tree into [(text, label), ...]
    with label in {'prefix', 'root', 'suffix'}."""
    out: list[tuple[str, str]] = []
    for node in tree:
        if "children" in node:
            for child in node["children"]:
                t = child.get("type", "root")
                label = "suffix" if t == "bound" else t
                out.append((child["text"].lower(), label))
        else:
            t = node.get("type", "root")
            label = "suffix" if t == "bound" else t
            out.append((node["text"].lower(), label))
    return out


_COMMON_INFLECTIONS = ("ies", "ing", "ed", "es", "s", "er", "est", "ly")


def _strip_inflection(surface: str,
                      segs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """If surface has trailing characters past the joined canonical form,
    record them as an explicit inflectional suffix.  Mirrors the dashboard
    morph-parser logic so the standalone module behaves the same way."""
    if not segs:
        return segs
    canon = "".join(t for t, _ in segs)
    if surface == canon:
        return segs
    i = 0
    while i < min(len(surface), len(canon)) and surface[i] == canon[i]:
        i += 1
    if i == len(canon):
        leftover = surface[i:]
        if leftover in _COMMON_INFLECTIONS or 1 <= len(leftover) <= 4:
            return segs + [(leftover, "inflection")]
    return segs


def _try_compound_split(word_lower: str) -> Optional[list[tuple[str, str]]]:
    """Cut a not-in-DB word into two halves both found in MorphoLex."""
    if len(word_lower) < 6:
        return None
    candidates: list[tuple[int, list[tuple[str, str]]]] = []
    for cut in range(3, len(word_lower) - 2):
        l, r = word_lower[:cut], word_lower[cut:]
        ls = _parse_one(l)
        rs = _parse_one(r)
        if ls and rs:
            candidates.append((abs(len(l) - len(r)), ls + rs))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


@lru_cache(maxsize=50000)
def _parse_one(word_lower: str) -> Optional[list[tuple[str, str]]]:
    """Direct MorphoLex lookup with no fallback.  Cached because the
    compound-split loop hits the same short fragments repeatedly."""
    parser = _get_morph_parser()
    try:
        result = parser.parse(word_lower)
    except Exception:
        return None
    if result.get("status") != "FOUND_IN_DATABASE":
        return None
    return _flatten_morph_tree(result["tree"])


@lru_cache(maxsize=20000)
def _morph_segments(surface: str) -> tuple[tuple[str, str], ...]:
    """Return ((text, label), ...) for *surface*.  Empty tuple for
    parse failures / non-alpha / very short tokens."""
    if not surface or not surface.isalpha() or len(surface) <= 2:
        return ()
    w = surface.lower()
    segs = _parse_one(w)
    if segs is None:
        segs = _try_compound_split(w)
    if segs is None:
        return ()
    segs = _strip_inflection(w, list(segs))
    return tuple(segs)


def _shannon_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total)
                for c in counts.values() if c)


def _legacy_manual_features(text: str,
                            tammi_out: dict[str, float]
                            ) -> dict[str, float]:
    """The 15 unique legacy manual features.  ``tammi_out`` is reused for
    ``morpheme_family_size`` (= TAMMI's ``root_family_size_per_cw``)."""
    doc = _NLP(text)
    words = [tok for tok in doc if tok.is_alpha and not tok.is_space]
    n_words = len(words)

    if n_words == 0:
        return {k: 0.0 for k in _LEGACY_FEATURE_NAMES}

    # ---- Morpheme decomposition pass ---------------------------------
    decomps = [_morph_segments(tok.text) for tok in words]
    nonempty = [s for s in decomps if s]

    morph_counts = [len(s) for s in nonempty]
    all_morphs = [t for s in nonempty for t, _ in s]
    free_count = sum(1 for s in nonempty for _, lbl in s if lbl == "root")
    bound_count = sum(1 for s in nonempty for _, lbl in s
                      if lbl in ("prefix", "suffix", "inflection"))

    avg_morphemes_per_word = mean(morph_counts) if morph_counts else 0.0
    morpheme_ttr = (len(set(all_morphs)) / len(all_morphs)
                    if all_morphs else 0.0)

    has_prefix = sum(1 for s in nonempty if any(lbl == "prefix" for _, lbl in s))
    has_suffix = sum(1 for s in nonempty if any(lbl == "suffix" for _, lbl in s))
    prefix_density = has_prefix / n_words
    derivational_suffix_density = has_suffix / n_words

    n_parsed = len(nonempty)
    derivational_depth = (bound_count / n_parsed) if n_parsed else 0.0

    compound = sum(1 for s in nonempty
                   if sum(1 for _, lbl in s if lbl == "root") >= 2)
    compound_word_density = compound / n_words

    polymorph_count = sum(1 for s in nonempty if len(s) >= 3)
    polymorph_ratio = polymorph_count / n_words

    max_morpheme_count = max(morph_counts, default=0)

    # ---- Inflection-side (spaCy) -------------------------------------
    content = [tok for tok in words
               if tok.pos_ in ("NOUN", "VERB", "AUX", "ADJ", "ADV")]
    inflected = sum(1 for tok in content
                    if tok.text.lower() != tok.lemma_.lower())
    inflected_form_ratio = (inflected / len(content)) if content else 0.0

    verb_tag_counts = Counter(tok.tag_ for tok in words
                              if tok.tag_ in ("VBD", "VBG", "VBN", "VBP", "VBZ"))
    verb_inflection_diversity = _shannon_entropy(verb_tag_counts)

    past_verbs = [tok for tok in words if tok.tag_ in ("VBD", "VBN")]
    irreg_past = sum(1 for tok in past_verbs
                     if tok.text.lower() in _IRREGULAR_PAST_PP)
    irregular_verb_ratio = (irreg_past / len(past_verbs)
                            if past_verbs else 0.0)

    # ---- Surface (regex on raw text) ---------------------------------
    hyphenated_compound_density = (
        len(_HYPHENATED_RE.findall(text)) / n_words
    )

    # ---- Aliased from TAMMI ------------------------------------------
    morpheme_family_size = float(
        tammi_out.get("root_family_size_per_cw", 0.0)
    )

    return {
        "avg_morphemes_per_word":       avg_morphemes_per_word,
        "total_free_morphemes":         float(free_count),
        "total_bound_morphemes":        float(bound_count),
        "morpheme_type_token_ratio":    morpheme_ttr,
        "derivational_suffix_density":  derivational_suffix_density,
        "prefix_density":               prefix_density,
        "derivational_depth":           derivational_depth,
        "morpheme_family_size":         morpheme_family_size,
        "inflected_form_ratio":         inflected_form_ratio,
        "verb_inflection_diversity":    verb_inflection_diversity,
        "irregular_verb_ratio":         irregular_verb_ratio,
        "compound_word_density":        compound_word_density,
        "hyphenated_compound_density":  hyphenated_compound_density,
        "polymorph_ratio":              polymorph_ratio,
        "max_morpheme_count":           float(max_morpheme_count),
    }


_LEGACY_FEATURE_NAMES: tuple[str, ...] = (
    "avg_morphemes_per_word",
    "total_free_morphemes",
    "total_bound_morphemes",
    "morpheme_type_token_ratio",
    "derivational_suffix_density",
    "prefix_density",
    "derivational_depth",
    "morpheme_family_size",
    "inflected_form_ratio",
    "verb_inflection_diversity",
    "irregular_verb_ratio",
    "compound_word_density",
    "hyphenated_compound_density",
    "polymorph_ratio",
    "max_morpheme_count",
)


# ===========================================================================
# Public
# ===========================================================================

D2_FEATURES: tuple[str, ...] = TAMMI_FEATURES + (
    # spaCy supplements TAMMI does not cover
    "irregular_verb_density_per_100w",
    "comparative_superlative_density",
    "inflection_class_entropy",
) + _LEGACY_FEATURE_NAMES


def compute_all(text: str, **_ignored) -> dict[str, float]:
    """Compute every D2 feature: 42 TAMMI + 3 spaCy supplements + 15 legacy
    manual features (60 unique keys total).
    """
    out: dict[str, float] = {}
    tammi_out = compute_tammi(text)
    out.update(tammi_out)
    out.update(_supplemental_features(text))
    out.update(_legacy_manual_features(text, tammi_out))
    return out


if __name__ == "__main__":
    sample = ("The woodcutter recognised the unhappiness of the village "
              "carpenters who walked through the unenchanted forest, "
              "discovering broken windmills and forgotten heirlooms.")
    for k, v in compute_all(sample).items():
        print(f"  {k:55s} {v:.4f}")
