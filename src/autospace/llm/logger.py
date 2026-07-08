"""Full LLM interaction logging to disk."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from threading import Lock


def _sanitize_for_json(obj):
    """Replace inf/-inf/NaN with JSON-safe values."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return 1e308 if obj > 0 else -1e308
        if math.isnan(obj):
            return None
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_sanitize_for_json(v) for v in obj)
    return obj


class LLMLogger:
    """Logs every LLM interaction to individual JSON files.

    Directory structure:
        log_dir/
            {agent}_{sequence:04d}.json
    """

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        # Resume counter from existing files to avoid overwriting (B1 fix)
        self._counter = 0
        import re
        for f in self.log_dir.glob("*_*.json"):
            m = re.search(r"_(\d+)\.json$", f.name)
            if m:
                self._counter = max(self._counter, int(m.group(1)))

    def log_interaction(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        response_content: str,
        input_tokens: int,
        output_tokens: int,
        latency_s: float,
        agent: str = "unknown",
        metadata: dict | None = None,
    ) -> Path:
        """Log a single LLM interaction to disk.

        Returns:
            Path to the written log file.
        """
        with self._lock:
            self._counter += 1
            seq = self._counter

            record = {
                "sequence": seq,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "model": model,
                "agent": agent,
                "system_prompt": system,
                "messages": messages,
                "response": response_content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency_s": round(latency_s, 3),
            }
            if metadata:
                record["metadata"] = metadata

            filename = f"{agent}_{seq:04d}.json"
            path = self.log_dir / filename
            path.write_text(json.dumps(
                _sanitize_for_json(record), indent=2, ensure_ascii=False,
            ))
        return path

    def set_agent(self, agent_name: str) -> "ScopedLogger":
        """Return a scoped logger that tags all interactions with the agent name."""
        return ScopedLogger(self, agent_name)


class ScopedLogger:
    """Logger wrapper that automatically tags interactions with an agent name."""

    def __init__(self, parent: LLMLogger, agent: str):
        self._parent = parent
        self.agent = agent

    def log_interaction(self, **kwargs) -> Path:
        kwargs.setdefault("agent", self.agent)
        return self._parent.log_interaction(**kwargs)
