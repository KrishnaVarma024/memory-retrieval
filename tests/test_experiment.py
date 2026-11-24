import json

from memret.bm25 import BM25
from memret.datasets import Dataset
from memret.experiment import index_corpus, rounded, run_queries, write_trec_run
from memret.metrics import evaluate
from memret.text import DEFAULT


def tiny():
    return Dataset(
        name="tiny", split="test",
        doc_ids=["q1", "d2", "d3"],
        docs=["apple apple apple", "apple banana", "cherry"],
        queries={"q1": "apple", "q2": "cherry pie"},
        qrels={"q1": {"d2": 1}, "q2": {"d3": 1}},
    )


def test_end_to_end_on_a_tiny_collection():
    ds = tiny()
    index, timings = index_corpus(ds.docs, DEFAULT)
    run, latencies = run_queries(BM25(index), DEFAULT, ds)
    assert run == {"q1": ["d2"], "q2": ["d3"]}  # document "q1" is dropped from query q1's results
    assert len(latencies) == 2 and set(timings) == {"analyze_s", "index_s"}
    assert evaluate(run, ds.qrels).mean()["nDCG@10"] == 1.0


def test_keeping_the_query_id_document():
    ds = tiny()
    index, _ = index_corpus(ds.docs, DEFAULT)
    run, _ = run_queries(BM25(index), DEFAULT, ds, drop_query_id=False)
    assert run["q1"] == ["q1", "d2"]


def test_outputs_are_stable(tmp_path):
    assert rounded({"x": [0.1 + 0.2, 1], "y": 2.0000000001}) == {"x": [0.3, 1], "y": 2.0}
    write_trec_run(tmp_path / "run.trec", {"q1": ["d2", "d9"]}, "tag")
    assert (tmp_path / "run.trec").read_text() == "q1 Q0 d2 1 -1 tag\nq1 Q0 d9 2 -2 tag\n"
    json.dumps(rounded({"a": 1.5}))
