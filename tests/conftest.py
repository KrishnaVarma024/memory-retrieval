"""Shared test setup.

On macOS, the torch and faiss-cpu wheels each bundle their own copy of the
OpenMP runtime (libomp.dylib). A process that loads both aborts with
"OMP: Error #15" the first time FAISS starts a parallel region. The commands
never load both (ann.py uses FAISS without torch; the embedding commands use
torch without FAISS), but one pytest process imports every test module. This
variable tells the runtime to tolerate the second copy. The tests that use
FAISS check its results against our own exact search, so a wrong answer
caused by the duplicate runtime would fail them rather than pass silently.
"""

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
