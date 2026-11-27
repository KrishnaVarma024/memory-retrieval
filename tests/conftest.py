"""Shared test setup.

On macOS, the torch and faiss-cpu wheels each bundle their own copy of the
OpenMP runtime (libomp.dylib), and the test suite loads both in one process:
test_embed.py imports torch, and the FAISS tests import faiss. Two things then
go wrong:

1. The second runtime refuses to start ("OMP: Error #15") and aborts the run.
   KMP_DUPLICATE_LIB_OK lets it start anyway.
2. With both running, worker threads started by FAISS's copy end up calling
   into torch's copy (a macOS crash report shows exactly that), and the process
   segfaults in the first multithreaded FAISS search. With one thread there
   are no worker threads, so the two copies never cross.

So on macOS the tests run with both settings, made here before torch or faiss
is imported. OMP_NUM_THREADS is forced rather than defaulted, because a value
inherited from the shell would bring the crash back. The FAISS tests compare
against our own exact search, so a wrong result would still fail them.

The commands don't need any of this: none of them loads both libraries, and
per-query FAISS search is single-threaded anyway (HNSWIndex defaults to one
thread). A future process that embeds queries with torch and searches with
FAISS must keep FAISS single-threaded for the same reason.
"""

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["OMP_NUM_THREADS"] = "1"
