"""Wall-clock + peak-RSS measurement for blocking experiments.

Peak RAM is measured via OS-level process monitoring (psutil), not inferred
from Phase-1 audit precedent, per the Phase-2 mission spec's scale-budget
requirement. A background sampler thread polls RSS at a fixed interval
because a single before/after snapshot would miss a transient peak that
occurs and is freed mid-operation (e.g., inside a sparse matrix multiply).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import psutil


@dataclass
class ResourceReport:
    wall_seconds: float
    peak_rss_bytes: int
    start_rss_bytes: int
    end_rss_bytes: int

    @property
    def peak_rss_mb(self) -> float:
        return self.peak_rss_bytes / (1024 * 1024)

    def as_dict(self) -> dict:
        return {
            "wall_seconds": round(self.wall_seconds, 3),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "start_rss_mb": round(self.start_rss_bytes / (1024 * 1024), 1),
            "end_rss_mb": round(self.end_rss_bytes / (1024 * 1024), 1),
        }


class ResourceMonitor:
    """Context manager: `with ResourceMonitor(poll_interval=0.1) as m: ...`

    After the `with` block, `m.report` is a populated ResourceReport.
    """

    def __init__(self, poll_interval: float = 0.2):
        self.poll_interval = poll_interval
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._peak = 0
        self._thread: threading.Thread | None = None
        self.report: ResourceReport | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss
                if rss > self._peak:
                    self._peak = rss
            except Exception:
                pass
            self._stop.wait(self.poll_interval)

    def __enter__(self) -> "ResourceMonitor":
        self._start_rss = self._process.memory_info().rss
        self._peak = self._start_rss
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        end_rss = self._process.memory_info().rss
        if end_rss > self._peak:
            self._peak = end_rss
        wall = time.perf_counter() - self._start_time
        self.report = ResourceReport(
            wall_seconds=wall,
            peak_rss_bytes=self._peak,
            start_rss_bytes=self._start_rss,
            end_rss_bytes=end_rss,
        )
