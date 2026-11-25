"""What embedding costs, and what the cache saves.

    uv run python -m memret.bench_embed

1. Throughput: embed the same 1,000 FiQA documents cold (an empty cache) and
   warm (all cached), and report texts and tokens per second.
2. Hit rates, counted from the texts themselves, so they don't depend on the
   machine: duplicates inside each corpus (a cold run's only hits), and the
   share of chunks that are whole documents (already cached by a document run).
3. What a full cold run of each corpus would take on this machine, from the
   measured tokens per second.

Writes reports/phase2/cache.json and reports/phase2/cache.md.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from .cache import EmbeddingCache
from .chunking import chunk_corpus
from .datasets import DEFAULT_ROOT, SOURCES, load
from .embed import BGE_SMALL, Embedder
from .evaluate_dense import CHUNK_MAX_TOKENS, CHUNK_OVERLAP
from .experiment import machine, write_json


def throughput(embedder: Embedder, texts: list[str]) -> dict:
    tokens = int(sum(min(n, BGE_SMALL.max_length - 2) + 2 for n in embedder.token_counts(texts)))
    embedder.encode(texts[:32])  # warm-up: the first call pays one-off setup costs
    with tempfile.TemporaryDirectory() as tmp:
        cache = EmbeddingCache(BGE_SMALL.fingerprint, BGE_SMALL.dim, Path(tmp))
        start = time.perf_counter()
        cold = cache.get_many(texts, embedder.encode)
        cold_s = time.perf_counter() - start
        start = time.perf_counter()
        warm = cache.get_many(texts, embedder.encode)
        warm_s = time.perf_counter() - start
        reloaded_start = time.perf_counter()
        EmbeddingCache(BGE_SMALL.fingerprint, BGE_SMALL.dim, Path(tmp))
        reload_s = time.perf_counter() - reloaded_start
    assert np.array_equal(cold, warm)
    return {"texts": len(texts), "tokens": tokens, "cold_s": cold_s, "warm_s": warm_s, "reload_s": reload_s,
            "cold_texts_per_s": len(texts) / cold_s, "cold_tokens_per_s": tokens / cold_s,
            "warm_texts_per_s": len(texts) / warm_s, "speedup": cold_s / warm_s}


def corpus_counts(name: str, embedder: Embedder, tokens_per_s: float, root: Path) -> dict:
    docs = list(load(name, "test", root).docs)
    chunks = chunk_corpus(docs, embedder.token_spans, max_tokens=CHUNK_MAX_TOKENS, overlap=CHUNK_OVERLAP)
    doc_set = set(docs)
    whole = sum(c.text in doc_set for c in chunks)
    chunk_unique = {c.text for c in chunks}
    new_texts = chunk_unique - doc_set
    doc_tokens = sum(min(n, BGE_SMALL.max_length - 2) + 2 for n in embedder.token_counts(list(doc_set)))
    new_tokens = sum(min(n, BGE_SMALL.max_length - 2) + 2 for n in embedder.token_counts(sorted(new_texts)))
    return {
        "dataset": name, "documents": len(docs), "unique_documents": len(doc_set),
        "duplicate_hit_rate_cold": 1 - len(doc_set) / len(docs),
        "chunks": len(chunks), "chunks_that_are_whole_documents": whole,
        "chunk_hit_rate_after_document_run": (len(chunks) - len(new_texts)) / len(chunks),
        "tokens_document_run": doc_tokens, "tokens_new_in_chunk_run": new_tokens,
        "estimated_cold_document_run_s": doc_tokens / tokens_per_s,
        "estimated_chunk_run_after_document_run_s": new_tokens / tokens_per_s,
    }


def summary_markdown(r: dict) -> str:
    t = r["throughput"]
    lines = [
        "# Phase 2: embedding cost and the cache",
        "",
        f"Model `{BGE_SMALL.name}`, CPU only, {r['threads']} threads, batches of 32 sorted by length.",
        "",
        "## Throughput",
        "",
        f"The same {t['texts']:,} FiQA documents ({t['tokens']:,} tokens including [CLS] and [SEP]), embedded into "
        "an empty cache and then again from it.",
        "",
        "| Run | Seconds | Texts per second | Tokens per second |",
        "| --- | --- | --- | --- |",
        f"| Cold (computed) | {t['cold_s']:.1f} | {t['cold_texts_per_s']:.1f} | {t['cold_tokens_per_s']:,.0f} |",
        f"| Warm (cached) | {t['warm_s']:.3f} | {t['warm_texts_per_s']:,.0f} | |",
        "",
        f"A cache hit is about {t['speedup']:,.0f} times faster than computing. Loading the cache from disk "
        f"in a new process took {t['reload_s']:.3f} s for these vectors.",
        "",
        "## Hit rates and what a full run costs",
        "",
        "Counted from the texts, so they hold on any machine. A cold document run's only hits are duplicate "
        "documents. A chunk run after a document run hits on every document short enough to stay whole. "
        "Times are estimates from the throughput above.",
        "",
        "| Dataset | Documents | Duplicate documents | Chunks | Chunk hit rate after a document run "
        "| Cold document run | Chunk run after it |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in r["corpora"]:
        lines.append(f"| {c['dataset']} | {c['documents']:,} | {c['documents'] - c['unique_documents']:,} "
                     f"({c['duplicate_hit_rate_cold']:.2%}) | {c['chunks']:,} | {c['chunk_hit_rate_after_document_run']:.1%} "
                     f"| {c['estimated_cold_document_run_s'] / 60:.0f} min | "
                     f"{c['estimated_chunk_run_after_document_run_s'] / 60:.1f} min |")
    m = r["machine"]
    lines += ["", f"Machine: {m['platform']}, Python {m['python']}, NumPy {m['numpy']}, {m['cpus']} CPUs.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=Path("reports/phase2"))
    args = parser.parse_args()
    embedder = Embedder(BGE_SMALL)
    import torch

    docs = list(load("fiqa", "test", args.data_dir).docs)
    sample = [docs[i] for i in np.random.default_rng(0).choice(len(docs), args.sample, replace=False)]
    tp = throughput(embedder, sample)
    print(f"cold {tp['cold_texts_per_s']:.1f} texts/s ({tp['cold_tokens_per_s']:,.0f} tokens/s), "
          f"warm {tp['warm_texts_per_s']:,.0f} texts/s", flush=True)
    corpora = [corpus_counts(n, embedder, tp["cold_tokens_per_s"], args.data_dir) for n in SOURCES]
    result = {"threads": torch.get_num_threads(), "throughput": tp, "corpora": corpora, "machine": machine()}
    write_json(args.out / "cache.json", result)
    (args.out / "cache.md").write_text(summary_markdown(result))


if __name__ == "__main__":
    main()
