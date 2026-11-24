from memret.text import DEFAULT, LUCENE_STOPWORDS, Analyzer

RAW = Analyzer(stopwords=frozenset(), stem=False)


def test_lowercases_and_splits_on_punctuation():
    assert RAW("Hello, World! (Again)") == ["hello", "world", "again"]


def test_keeps_internal_full_stops_apostrophes_and_digit_commas():
    assert RAW("e.g. 3.14 and 1,000 don't") == ["e.g", "3.14", "and", "1,000", "don't"]


def test_comma_between_letters_still_splits():
    assert RAW("red,green") == ["red", "green"]


def test_hyphens_and_slashes_split():
    assert RAW("state-of-the-art and/or") == ["state", "of", "the", "art", "and", "or"]


def test_possessives_are_removed_with_either_apostrophe():
    assert RAW("the author's view and the editor’s") == ["the", "author", "view", "and", "the", "editor"]


def test_unicode_letters_are_words():
    assert RAW("Café NAÏVE") == ["café", "naïve"]


def test_lucene_stopword_list_has_33_words():
    assert len(LUCENE_STOPWORDS) == 33
    assert {"the", "a", "not", "with"} <= LUCENE_STOPWORDS
    assert "very" not in LUCENE_STOPWORDS


def test_stopwords_removed_before_stemming():
    assert Analyzer(stem=False)("the cat is on the mat") == ["cat", "mat"]


def test_porter_stemming_conflates_inflections():
    assert DEFAULT("running runs") == ["run", "run"]
    assert DEFAULT("connections connected connecting") == ["connect"] * 3


def test_empty_and_punctuation_only_text():
    assert DEFAULT("") == []
    assert DEFAULT(" -- !! ") == []
