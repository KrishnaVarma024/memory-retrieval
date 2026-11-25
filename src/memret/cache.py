"""A content-addressed cache of embeddings.

Embedding is by far the most expensive step: about 4,000 tokens per second on
four CPU cores, so FiQA's 57,638 documents take about 40 minutes. Nothing
should ever be embedded twice.

A vector is stored under the SHA-256 of (model fingerprint, exact text). So:

- The same text embedded by the same model is a hit, wherever it came from: a
  rerun, a second chunking setting that leaves short documents whole, a
  duplicate document.
- A different model, revision or pooling gives a different fingerprint, so its
  vectors can never be mixed up with these. Each fingerprint gets its own folder.

On disk, vectors are written in shards of a few thousand. Each shard is written
to a temporary file, flushed to disk, then renamed into place, so a shard is
either complete or absent. That makes long embedding runs resumable: kill one
halfway and the next run starts where it stopped. NumPy's .npz format is a zip
file, whose CRC-32 per member catches a damaged shard when it is read.

Phase 5 moves this into SQLite with reference counts, so entries can be deleted.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KEY_BYTES = 32  # a SHA-256 digest


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    entries: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class EmbeddingCache:
    """Maps (fingerprint, text) to a vector. root=None keeps everything in memory."""

    def __init__(self, fingerprint: str, dim: int, root: Path | None = None, *, shard_size: int = 4096):
        self.fingerprint, self.dim, self.shard_size = fingerprint, dim, shard_size
        self._prefix = fingerprint.encode("utf-8") + b"\x00"
        self._vectors: dict[bytes, np.ndarray] = {}
        self.hits = self.misses = 0
        self.folder = None
        if root is not None:
            self.folder = Path(root) / hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / "fingerprint.txt").write_text(fingerprint + "\n")
            for shard in sorted(self.folder.glob("shard-*.npz")):
                self._load(shard)

    def key(self, text: str) -> bytes:
        return hashlib.sha256(self._prefix + text.encode("utf-8")).digest()

    def stats(self) -> CacheStats:
        return CacheStats(self.hits, self.misses, len(self._vectors))

    def reset_counters(self) -> None:
        self.hits = self.misses = 0

    def __contains__(self, text: str) -> bool:
        return self.key(text) in self._vectors

    def get_many(self, texts: Sequence[str], compute: Callable[[list[str]], np.ndarray]) -> np.ndarray:
        """Vectors for texts, computing only those not cached.

        A text seen earlier in the same call counts as a hit: it is computed once.
        """
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        missing: dict[bytes, list[int]] = {}
        for i, text in enumerate(texts):
            k = self.key(text)
            vec = self._vectors.get(k)
            if vec is not None:
                out[i] = vec
                self.hits += 1
            elif k in missing:
                missing[k].append(i)
                self.hits += 1
            else:
                missing[k] = [i]
                self.misses += 1
        todo = list(missing.items())
        for start in range(0, len(todo), self.shard_size):
            part = todo[start:start + self.shard_size]
            vecs = np.asarray(compute([texts[positions[0]] for _, positions in part]), dtype=np.float32)
            if vecs.shape != (len(part), self.dim):
                raise ValueError(f"compute returned shape {vecs.shape}, expected {(len(part), self.dim)}")
            keys = [k for k, _ in part]
            self._store(keys, vecs)
            for (_, positions), vec in zip(part, vecs):
                out[positions] = vec
        return out

    def _store(self, keys: list[bytes], vecs: np.ndarray) -> None:
        for k, vec in zip(keys, vecs):
            self._vectors[k] = vec
        if self.folder is None:
            return
        key_array = np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(len(keys), KEY_BYTES)
        fd, tmp_name = tempfile.mkstemp(dir=self.folder, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, keys=key_array, vectors=vecs)
                f.flush()
                os.fsync(f.fileno())
            final = self.folder / f"shard-{len(list(self.folder.glob('shard-*.npz'))):06d}-{keys[0].hex()[:8]}.npz"
            os.replace(tmp_name, final)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        _fsync_dir(self.folder)

    def _load(self, shard: Path) -> None:
        with np.load(shard, allow_pickle=False) as data:
            keys, vecs = data["keys"], data["vectors"]
        if keys.ndim != 2 or keys.shape[1] != KEY_BYTES or vecs.shape != (len(keys), self.dim):
            raise ValueError(f"{shard}: unexpected shapes {keys.shape} and {vecs.shape}")
        for row, vec in zip(keys, vecs.astype(np.float32, copy=False)):
            self._vectors[row.tobytes()] = vec


def _fsync_dir(folder: Path) -> None:
    """Make a rename durable. Not every platform can open a folder; skip where it can't."""
    try:
        fd = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
