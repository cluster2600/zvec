"""Test configuration — provides a numpy-based faiss mock for macOS Tahoe.

The FAISS SWIG C extension segfaults on macOS 26 (Tahoe) due to binary
incompatibility.  This conftest installs a lightweight numpy-based mock
that provides enough of the FAISS API surface for our GPU index tests.

This must be loaded **before** any `import zvec` so that `detect.py`
picks up the mock instead of the broken C library.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np


# ---------------------------------------------------------------------------
# Minimal faiss mock (numpy-only, supports Flat indexes)
# ---------------------------------------------------------------------------


class _FaissIndexFlatL2:
    """Minimal IndexFlatL2 implemented in pure numpy."""

    def __init__(self, d: int):
        self.d = d
        self.ntotal = 0
        self.is_trained = True
        self._data: np.ndarray | None = None

    def add(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        if self._data is None:
            self._data = x.copy()
        else:
            self._data = np.vstack([self._data, x])
        self.ntotal = self._data.shape[0]

    def search(self, x: np.ndarray, k: int):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # Brute-force L2 search
        nq = x.shape[0]
        k = min(k, self.ntotal)
        distances = np.zeros((nq, k), dtype=np.float32)
        indices = np.zeros((nq, k), dtype=np.int64)
        for i in range(nq):
            dists = np.sum((self._data - x[i]) ** 2, axis=1)
            idx = np.argsort(dists)[:k]
            distances[i] = dists[idx]
            indices[i] = idx
        return distances, indices

    def reset(self) -> None:
        self._data = None
        self.ntotal = 0


class _FaissIndexFlatIP:
    """Minimal IndexFlatIP (inner product)."""

    def __init__(self, d: int):
        self.d = d
        self.ntotal = 0
        self.is_trained = True
        self._data: np.ndarray | None = None

    def add(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        if self._data is None:
            self._data = x.copy()
        else:
            self._data = np.vstack([self._data, x])
        self.ntotal = self._data.shape[0]

    def search(self, x: np.ndarray, k: int):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        nq = x.shape[0]
        k = min(k, self.ntotal)
        distances = np.zeros((nq, k), dtype=np.float32)
        indices = np.zeros((nq, k), dtype=np.int64)
        for i in range(nq):
            sims = x[i] @ self._data.T
            idx = np.argsort(-sims)[:k]  # descending for IP
            distances[i] = sims[idx]
            indices[i] = idx
        return distances, indices

    def reset(self) -> None:
        self._data = None
        self.ntotal = 0


def _mock_faiss_module():
    """Create a mock faiss module with numpy-backed implementations."""
    faiss = types.ModuleType("faiss")
    faiss.__version__ = "0.0.0-mock"
    faiss.__path__ = []

    faiss.IndexFlatL2 = _FaissIndexFlatL2
    faiss.IndexFlatIP = _FaissIndexFlatIP

    # Metric constants
    faiss.METRIC_L2 = 1
    faiss.METRIC_INNER_PRODUCT = 0

    # StandardGpuResources — raise to simulate no GPU
    def _no_gpu_resources():
        raise RuntimeError("Mock FAISS: no GPU available")

    faiss.StandardGpuResources = _no_gpu_resources

    # swigfaiss sub-module (needed by some import paths)
    swigfaiss = types.ModuleType("faiss.swigfaiss")
    faiss.swigfaiss = swigfaiss

    # loader sub-module
    loader = types.ModuleType("faiss.loader")
    faiss.loader = loader

    return faiss


# ---------------------------------------------------------------------------
# Install the mock BEFORE any zvec import
# ---------------------------------------------------------------------------

# Only install if real faiss would segfault (or isn't importable)
_need_mock = False
if "faiss" not in sys.modules:
    try:
        import faiss as _real_faiss  # noqa: F401
    except (ImportError, SystemError, OSError):
        _need_mock = True
    except Exception:
        # Segfault can't be caught, but any other failure → mock
        _need_mock = True

if _need_mock:
    _mock = _mock_faiss_module()
    sys.modules["faiss"] = _mock
    sys.modules["faiss.swigfaiss"] = _mock.swigfaiss
    sys.modules["faiss.loader"] = _mock.loader
