"""Turning text into vectors with a sentence-embedding model.

The model is BAAI/bge-small-en-v1.5: a 33-million-parameter BERT encoder that
maps a piece of text to 384 numbers. Texts that mean similar things land close
together, so search becomes "find the stored vectors nearest the query's".

Four steps, each visible below rather than hidden in a library call:

1. Tokenise into WordPiece tokens, adding [CLS] at the start and [SEP] at the
   end, and truncate to the model's limit of 512 tokens.
2. Run the encoder. It returns one 384-number vector per token.
3. Pool: bge uses the vector of the [CLS] token as the vector of the whole text.
4. Normalise each vector to length 1, so a dot product equals cosine similarity.

Queries get an instruction prefix the model was trained with; passages don't.
The model is pinned to an exact commit of its weights, like the datasets are,
so a rerun next year produces the same vectors.

torch and transformers are imported only when an Embedder is created, so the
Phase 1 commands don't need them installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    revision: str  # exact commit of the weights on the Hugging Face Hub
    dim: int
    max_length: int = 512  # tokens, including [CLS] and [SEP]
    query_prefix: str = ""

    @property
    def fingerprint(self) -> str:
        """Everything that changes the vectors. Cached vectors are keyed by it."""
        return f"{self.name}@{self.revision}|cls|max{self.max_length}|l2"


BGE_SMALL = ModelSpec(
    name="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    dim=384,
    query_prefix="Represent this sentence for searching relevant passages: ",
)


class Embedder:
    """Encodes texts into L2-normalised float32 vectors, in batches.

    Texts are sorted by length before batching, so each batch pads to a similar
    length. On SciFact-length documents this alone is three times faster than
    batching in corpus order, because padding tokens cost as much as real ones.
    """

    def __init__(self, spec: ModelSpec = BGE_SMALL, *, batch_size: int = 32, threads: int | None = None,
                 model=None, tokenizer=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if threads is not None:
            torch.set_num_threads(threads)
        self.spec = spec
        self.batch_size = batch_size
        self._torch = torch
        self._auto_model = AutoModel
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(spec.name, revision=spec.revision)
        self._model = model.eval() if model is not None else None

    @property
    def model(self):
        """The encoder, loaded on first use: a run whose vectors are all cached never loads it."""
        if self._model is None:
            self._model = self._auto_model.from_pretrained(self.spec.name, revision=self.spec.revision).eval()
        return self._model

    @property
    def dim(self) -> int:
        return self.spec.dim

    def query_text(self, query: str) -> str:
        return self.spec.query_prefix + query

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Character span of every token in text, without [CLS] and [SEP]. Used by the chunker."""
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(span) for span in enc["offset_mapping"]]

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        return [len(ids) for ids in self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """One unit-length vector per text, in the order given."""
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        if not texts:
            return out
        order = np.argsort(self.token_counts(texts), kind="stable")
        torch = self._torch
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                enc = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                     max_length=self.spec.max_length, return_tensors="pt")
                cls = self.model(**enc).last_hidden_state[:, 0]
                out[idx] = cls.float().numpy()
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)
        return out
