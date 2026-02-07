from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def detect_best(self, bgr: np.ndarray):
        ...

    def detect_all(self, bgr: np.ndarray):
        ...

    def embed_bgr(self, bgr: np.ndarray) -> np.ndarray:
        ...
