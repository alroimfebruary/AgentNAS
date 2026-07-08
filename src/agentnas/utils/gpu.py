"""GPU discovery, allocation pool, and VRAM monitoring."""

from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Lock


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    index: int
    name: str
    total_memory_mb: int
    free_memory_mb: int


class GPUPool:
    """Manages allocation and release of GPU devices.

    Usage:
        pool = GPUPool(n_gpu=2)
        gpu_ids = pool.allocate(count=2)   # [0, 1]
        pool.release(gpu_ids)
    """

    def __init__(self, n_gpu: int | None = None):
        available = self._discover_gpus()
        if n_gpu is not None:
            if n_gpu > len(available):
                import logging as _log
                _log.getLogger(__name__).warning(
                    f"Requested {n_gpu} GPUs but only {len(available)} available. "
                    f"Using {len(available)}."
                )
            available = available[: n_gpu]
        self._all_gpus = available
        self._free: set[int] = {g.index for g in available}
        self._lock = Lock()

    @property
    def total_gpus(self) -> int:
        return len(self._all_gpus)

    @property
    def free_count(self) -> int:
        with self._lock:
            return len(self._free)

    def allocate(self, count: int = 1) -> list[int]:
        """Allocate `count` GPU indices. Raises RuntimeError if not enough free."""
        with self._lock:
            if len(self._free) < count:
                raise RuntimeError(
                    f"Requested {count} GPUs but only {len(self._free)} free "
                    f"(total: {self.total_gpus})"
                )
            allocated = sorted(self._free)[:count]
            self._free -= set(allocated)
            return allocated

    def release(self, gpu_ids: list[int] | int) -> None:
        """Return GPUs to the pool."""
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        with self._lock:
            self._free.update(gpu_ids)

    def allocate_one(self) -> int:
        """Allocate a single GPU index."""
        return self.allocate(1)[0]

    @staticmethod
    def _discover_gpus() -> list[GPUInfo]:
        """Discover available GPUs using torch.cuda."""
        try:
            import torch

            if not torch.cuda.is_available():
                return []
            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mb = props.total_memory // (1024 * 1024)
                # Free memory requires setting device
                free_mb = total_mb  # Approximate; actual free checked at runtime
                gpus.append(
                    GPUInfo(
                        index=i,
                        name=props.name,
                        total_memory_mb=total_mb,
                        free_memory_mb=free_mb,
                    )
                )
            return gpus
        except ImportError:
            return []

    def get_gpu_info(self, gpu_id: int) -> GPUInfo | None:
        """Get info for a specific GPU."""
        for g in self._all_gpus:
            if g.index == gpu_id:
                return g
        return None

    def __repr__(self) -> str:
        return (
            f"GPUPool(total={self.total_gpus}, free={self.free_count}, "
            f"gpus={[g.name for g in self._all_gpus]})"
        )


def get_gpu_env(gpu_id: int) -> dict[str, str]:
    """Return environment variables to restrict a subprocess to a single GPU.

    Resolves logical pool index to physical GPU ID when the parent process
    has a non-default CUDA_VISIBLE_DEVICES (e.g., on managed clusters). (C-05 fix)
    """
    env = os.environ.copy()
    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_visible:
        physical_ids = [x.strip() for x in parent_visible.split(",")]
        if gpu_id < len(physical_ids):
            env["CUDA_VISIBLE_DEVICES"] = physical_ids[gpu_id]
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


