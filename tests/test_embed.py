"""The embedder's own logic, checked with a tiny randomly initialised BERT: no download, no network."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from memret.embed import BGE_SMALL, Embedder, ModelSpec  # noqa: E402

WORDS = ["the", "heart", "rate", "pulse", "lower", "how", "to", "my", "reduce", "a", "b", "c"]


@pytest.fixture(scope="module")
def embedder():
    torch.manual_seed(0)
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS])}
    tokenizer = transformers.BertTokenizer(vocab=vocab)
    config = transformers.BertConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=16, max_position_embeddings=32)
    model = transformers.BertModel(config)
    spec = ModelSpec(name="tiny", revision="0", dim=8, max_length=16, query_prefix="how to ")
    return Embedder(spec, batch_size=2, model=model, tokenizer=tokenizer)


def test_vectors_have_unit_length_and_the_right_shape(embedder):
    out = embedder.encode(["the heart rate", "pulse", "a b c a b c"])
    assert out.shape == (3, 8) and out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)


def test_order_is_restored_after_sorting_by_length(embedder):
    texts = ["a b c a b c a", "pulse", "the heart rate"]
    together = embedder.encode(texts)
    one_by_one = np.vstack([embedder.encode([t]) for t in texts])
    np.testing.assert_allclose(together, one_by_one, atol=1e-5)


def test_the_vector_is_the_normalised_cls_output(embedder):
    enc = embedder.tokenizer(["lower my pulse"], return_tensors="pt")
    with torch.inference_mode():
        cls = embedder.model(**enc).last_hidden_state[0, 0].numpy()
    np.testing.assert_allclose(embedder.encode(["lower my pulse"])[0], cls / np.linalg.norm(cls), atol=1e-5)


def test_long_texts_are_truncated_not_rejected(embedder):
    assert embedder.encode([" ".join(["heart"] * 100)]).shape == (1, 8)


def test_queries_get_the_prefix_and_passages_do_not(embedder):
    assert embedder.query_text("lower my pulse") == "how to lower my pulse"


def test_token_spans_point_into_the_text(embedder):
    text = "Heart rate, pulse."
    assert [text[a:b] for a, b in embedder.token_spans(text)] == ["Heart", "rate", ",", "pulse", "."]


def test_empty_input(embedder):
    assert embedder.encode([]).shape == (0, 8)


def test_the_fingerprint_names_everything_that_changes_vectors():
    fp = BGE_SMALL.fingerprint
    assert BGE_SMALL.name in fp and BGE_SMALL.revision in fp and "512" in fp
