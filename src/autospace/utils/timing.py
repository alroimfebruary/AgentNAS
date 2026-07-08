"""Training time enforcement using GPU time measurement."""

from __future__ import annotations

import time

import torch


class GPUTimer:
    """Tracks cumulative GPU compute time using per-step CUDA event pairs.

    Records a start and end event for EACH training step, measuring only the
    GPU compute (forward + backward + optimizer). Inter-step CPU time (data
    loading, augmentation, .to(device)) is excluded.

    Synchronizes and sums events periodically (every sync_interval steps) to
    avoid the overhead of per-step cuda.synchronize().

    Usage:
        timer = GPUTimer(limit_s=3600, device=device)
        for batch in loader:
            # CPU work here (data loading, augmentation) — NOT measured
            timer.step_start()
            # GPU work here (forward, backward, optimizer) — MEASURED
            timer.step_end()
            if timer.exceeded:
                break
    """

    def __init__(self, limit_s: float, device: torch.device | None = None, sync_interval: int = 50):
        self.limit_s = limit_s
        self._total_gpu_ms: float = 0.0
        self._step_count: int = 0
        self._sync_interval = sync_interval
        self._use_cuda = torch.cuda.is_available() and device is not None and device.type == "cuda"

        # Per-step event pairs (start, end) not yet synced
        self._pending_pairs: list[tuple] = []
        self._current_start = None

        if not self._use_cuda:
            self._wall_start: float = 0.0

    def step_start(self) -> None:
        """Call before each training step (forward + backward + optimizer)."""
        if self._use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._current_start = ev
        else:
            self._wall_start = time.monotonic()

    def step_end(self) -> None:
        """Call after each training step. Accumulates GPU time.

        Creates a per-step event pair. Only synchronizes every sync_interval
        steps to reduce overhead. Each pair measures ONLY that step's GPU work.
        """
        self._step_count += 1
        if self._use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._pending_pairs.append((self._current_start, ev))
            if self._step_count % self._sync_interval == 0:
                self._flush_pending()
        else:
            self._total_gpu_ms += (time.monotonic() - self._wall_start) * 1000.0

    def _flush_pending(self) -> None:
        """Synchronize and sum all pending event pairs."""
        if self._use_cuda and self._pending_pairs:
            torch.cuda.synchronize()
            for start, end in self._pending_pairs:
                self._total_gpu_ms += start.elapsed_time(end)
            self._pending_pairs.clear()

    def flush(self) -> None:
        """Flush any pending steps. Call at epoch end."""
        self._flush_pending()

    @property
    def elapsed_s(self) -> float:
        """Total accumulated GPU time in seconds."""
        return self._total_gpu_ms / 1000.0

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.limit_s - self.elapsed_s)

    @property
    def exceeded(self) -> bool:
        # Only sync when enough steps have passed to avoid per-batch CUDA sync (B-04 fix)
        # Uses stale elapsed_s between syncs — up to sync_interval steps of lag
        return self.elapsed_s >= self.limit_s

    def stop(self) -> None:
        """Flush remaining and finalize."""
        self._flush_pending()


class WallClockTimer:
    """Tracks wall-clock time with a hard limit. Kept for subprocess timeout."""

    def __init__(self, limit_s: float):
        self.limit_s = limit_s
        self._start: float | None = None
        self._end: float | None = None

    def start(self) -> None:
        self._start = time.monotonic()
        self._end = None

    def stop(self) -> None:
        self._end = time.monotonic()

    @property
    def elapsed_s(self) -> float:
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.monotonic()
        return end - self._start

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.limit_s - self.elapsed_s)

    @property
    def exceeded(self) -> bool:
        return self.elapsed_s >= self.limit_s
