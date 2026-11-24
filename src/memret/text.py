"""Text analysis: turning a string into the terms an index stores.

The default pipeline copies the one Anserini uses for its BEIR baselines
(Lucene's StandardTokenizer, possessive removal, lower-casing, Lucene's 33
English stopwords, Porter stemming), so our BM25 can be checked against
published numbers. The tokenizer is a regular-expression approximation of
Lucene's Unicode word-break rules, not a port of them; the difference is
measured in the Phase 1 report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import Stemmer

# Lucene's EnglishAnalyzer.ENGLISH_STOP_WORDS_SET.
LUCENE_STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such "
    "that the their then there these they this to was will with".split()
)

# A word is a run of letters, digits or underscores. Like Lucene's tokenizer, it
# stays whole across an internal full stop or apostrophe ("e.g", "don't", "3.14")
# and across a comma between digits ("1,000"). Hyphens and slashes split words.
_WORD = re.compile(r"\w+(?:(?:[.'’]|(?<=\d),(?=\d))\w+)*")


@dataclass(frozen=True)
class Analyzer:
    """Splits text into index terms.

    Not thread-safe: the PyStemmer object inside keeps a cache. Give each thread its own Analyzer.
    """

    stopwords: frozenset[str] = LUCENE_STOPWORDS
    stem: bool = True
    _stemmer: Stemmer.Stemmer = field(default_factory=lambda: Stemmer.Stemmer("porter"), repr=False, compare=False)

    def __call__(self, text: str) -> list[str]:
        words = []
        for word in _WORD.findall(text.lower()):
            if word.endswith(("'s", "’s")):  # possessive: "author's" -> "author"
                word = word[:-2]
            if word and word not in self.stopwords:
                words.append(word)
        return self._stemmer.stemWords(words) if self.stem else words


NO_STEM = Analyzer(stem=False)
NO_STOPWORDS = Analyzer(stopwords=frozenset())
DEFAULT = Analyzer()
