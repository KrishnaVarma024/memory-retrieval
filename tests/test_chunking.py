import re

import pytest

from memret.chunking import chunk_corpus, chunk_spans


def spans(text):
    """A stand-in tokenizer: one token per run of non-space characters."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def words(n, sentence_every=7):
    return " ".join(f"w{i}." if i % sentence_every == sentence_every - 1 else f"w{i}" for i in range(n))


def tokens_of(text, a, b):
    return [text[s:e] for s, e in spans(text) if s >= a and e <= b]


def test_a_text_that_fits_stays_whole_including_its_whitespace():
    text = "  Short title. Short body.  "
    assert chunk_spans(text, spans(text), 10, 2) == [(0, len(text))]


def test_empty_text_gives_one_empty_chunk():
    assert chunk_spans("", [], 10, 2) == [(0, 0)]


def test_every_chunk_respects_the_cap_and_every_token_is_covered():
    text = words(200)
    ranges = chunk_spans(text, spans(text), 20, 5)
    covered = set()
    for a, b in ranges:
        toks = tokens_of(text, a, b)
        assert 1 <= len(toks) <= 20
        covered.update(toks)
    assert covered == set(text.split())


def test_chunks_end_at_sentence_boundaries_when_one_falls_in_the_second_half():
    text = words(60)
    for a, b in chunk_spans(text, spans(text), 20, 5)[:-1]:
        assert text[a:b].endswith(".")


def test_a_long_run_without_sentence_breaks_is_cut_hard_at_the_cap():
    text = " ".join(f"t{i}" for i in range(45))
    ranges = chunk_spans(text, spans(text), 20, 0)
    assert [len(tokens_of(text, a, b)) for a, b in ranges] == [20, 20, 5]


def test_consecutive_chunks_overlap_and_always_move_forward():
    text = " ".join(f"t{i}" for i in range(100))
    ranges = chunk_spans(text, spans(text), 20, 5)
    for (a1, b1), (a2, b2) in zip(ranges, ranges[1:]):
        assert a1 < a2 < b1 < b2
        assert len(tokens_of(text, a2, b1)) == 5


def test_overlap_snaps_to_a_sentence_start_inside_it():
    text = words(60)
    ranges = chunk_spans(text, spans(text), 20, 8)
    for a, _ in ranges[1:]:
        before = text[:a].rstrip()
        assert before.endswith(".")


@pytest.mark.parametrize("max_tokens, overlap", [(0, 0), (5, 5), (5, -1)])
def test_bad_settings_are_rejected(max_tokens, overlap):
    with pytest.raises(ValueError):
        chunk_spans("a b c", spans("a b c"), max_tokens, overlap)


def test_chunk_corpus_keeps_document_positions_and_exact_slices():
    docs = ["tiny", words(50)]
    chunks = chunk_corpus(docs, spans, max_tokens=20, overlap=5)
    assert chunks[0].doc == 0 and chunks[0].text == "tiny"
    assert [c.doc for c in chunks[1:]] == [1] * (len(chunks) - 1)
    assert [c.ordinal for c in chunks[1:]] == list(range(len(chunks) - 1))
    for c in chunks:
        assert docs[c.doc][c.start:c.end] == c.text
