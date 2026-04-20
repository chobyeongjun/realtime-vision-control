"""CUDA Graph capture for fixed-shape infer + preproc.

On Orin NX the CPU launch cost of hundreds of tiny kernels dominates
small-batch inference. Once shapes are fixed, we capture the
preproc+infer sub-graph and replay it with a single launch.

If capture fails (dynamic shape, disallowed op, etc.) we fall back to
regular eager execution without affecting correctness.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Callable, Optional

import torch

LOGGER = logging.getLogger(__name__)


class GraphedStep:
    """Wraps a callable into a CUDA Graph on a specific stream."""

    def __init__(
        self,
        stream: torch.cuda.Stream,
        fn: Callable[[], None],
        warmup: int = 3,
    ) -> None:
        self.stream = stream
        self.fn = fn
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._warmup_iters = warmup
        self._captured = False
        self._capture_error: Optional[str] = None

    def try_capture(self) -> bool:
        """Warm up then capture. Returns True on success."""
        try:
            with torch.cuda.stream(self.stream):
                for _ in range(self._warmup_iters):
                    self.fn()
                self.stream.synchronize()

                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph, stream=self.stream):
                    self.fn()
            self._captured = True
            LOGGER.info("CUDA graph captured on stream %s", self.stream)
            return True
        except Exception as err:  # pragma: no cover — device-specific
            self._capture_error = str(err)
            self._graph = None
            self._captured = False
            LOGGER.warning("CUDA graph capture failed: %s — eager fallback", err)
            return False

    def replay(self) -> None:
        if self._captured and self._graph is not None:
            self._graph.replay()
        else:
            with torch.cuda.stream(self.stream):
                self.fn()

    @property
    def captured(self) -> bool:
        return self._captured

    @property
    def capture_error(self) -> Optional[str]:
        return self._capture_error


@contextlib.contextmanager
def disabled_if(cond: bool):
    """Tiny utility — lets the caller skip graph capture under a flag."""
    if cond:
        yield None
    else:
        yield True
