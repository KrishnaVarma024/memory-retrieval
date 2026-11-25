"""Splitting documents into pieces the embedding model can read whole.

The model reads at most 512 tokens, 510 after [CLS] and [SEP]. Anything longer
is silently cut off, so the end of a long document is invisible to search.
Chunking splits such documents into windows that each fit, and a document's
score becomes the score of its best chunk.

The rule is structure-aware with a hard cap:

- A document that fits stays whole. Most do: 91% of SciFact, 96% of FiQA.
- Otherwise, a window ends at the last sentence boundary in its second half,
  so chunks don't stop mid-sentence, and never runs past the cap.
- Consecutive windows overlap by up to `overlap` tokens, starting at a sentence
  boundary where one falls inside the overlap, so a fact that straddles a
  boundary appears whole in at least one chunk.

Chunk boundaries are character offsets into the original text, computed from
the tokenizer's own token spans, so chunk text is always an exact slice.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Where a new sentence starts: after ., ! or ? followed by whitespace, or after a line break.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Chunk:
    doc: int  # position of the document in the corpus
    ordinal: int  # position of the chunk within its document
    start: int  # character offsets into the document text
    end: int
    text: str


def chunk_spans(text: str, spans: Sequence[tuple[int, int]], max_tokens: int, overlap: int) -> list[tuple[int, int]]:
    """Character ranges of the chunks of text, given the character span of each of its tokens."""
    if max_tokens < 1 or not 0 <= overlap < max_tokens:
        raise ValueError(f"need max_tokens >= 1 and 0 <= overlap < max_tokens, got {max_tokens}, {overlap}")
    n = len(spans)
    if n <= max_tokens:
        return [(0, len(text))]
    breaks = [m.end() for m in _SENTENCE_BREAK.finditer(text)]
    # Token i starts a sentence if a sentence break falls between token i-1's end and token i's start.
    starts_sentence = [False] * n
    for i in range(1, n):
        j = bisect.bisect_left(breaks, spans[i - 1][1])
        starts_sentence[i] = j < len(breaks) and breaks[j] <= spans[i][0]
    sentence_starts = [i for i in range(n) if starts_sentence[i]]

    ranges = []
    start = 0
    while True:
        end = min(start + max_tokens, n)
        if end < n:
            # Prefer to end just before the last sentence that starts in the window's second half.
            j = bisect.bisect_right(sentence_starts, end) - 1
            if j >= 0 and sentence_starts[j] > start + max_tokens // 2:
                end = sentence_starts[j]
        ranges.append((start, end))
        if end == n:
            break
        nxt = max(end - overlap, start + 1)
        j = bisect.bisect_left(sentence_starts, nxt)  # snap to a sentence start inside the overlap
        if j < len(sentence_starts) and sentence_starts[j] < end:
            nxt = sentence_starts[j]
        start = nxt
    return [(spans[a][0], spans[b - 1][1]) for a, b in ranges]


def chunk_corpus(docs: Sequence[str], token_spans: Callable[[str], list[tuple[int, int]]], *,
                 max_tokens: int = 510, overlap: int = 64) -> list[Chunk]:
    """Every document's chunks, in corpus order. Every document gets at least one chunk."""
    chunks = []
    for d, text in enumerate(docs):
        for ordinal, (a, b) in enumerate(chunk_spans(text, token_spans(text), max_tokens, overlap)):
            chunks.append(Chunk(d, ordinal, a, b, text[a:b]))
    return chunks
