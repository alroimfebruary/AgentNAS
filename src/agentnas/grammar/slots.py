"""Slot-based architecture representation using self-contained nn.Module alternatives.

Each slot in the scaffold instantiates a named module class (e.g., SLOT_block0).
Each alternative provides a complete nn.Module class definition for that name.
realize() concatenates the chosen module classes + scaffold → valid Python.

No template injection. No code fragments. Each piece is a standalone class.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_SLOT_CLASS_BLOCK_RE = re.compile(
    r"(^class\s+(SLOT_\w+)\s*\([^)]*\)\s*:\s*\n(?:[ \t]+.*\n|[ \t]*\n)+)",
    re.MULTILINE,
)


def _dedupe_slot_classes(code: str) -> str:
    """Remove duplicate top-level `class SLOT_X(...)` definitions, keeping the first.

    Two bug variants this protects against:
    1. Phase 2 scaffolds that include stub `class SLOT_X(nn.Module): def forward(self, x): return x`
       declarations at the top of the scaffold. When concatenated AFTER the chosen alternatives'
       implementations, the stubs override the real classes (Python keeps the last module-level
       definition).
    2. Individual slot alternatives whose module_code accidentally defines the same SLOT_X class
       twice in their own body.

    Both manifest as evolved architectures collapsing to identity-only forward passes.
    """
    seen: set[str] = set()

    def _keep_first(match: re.Match[str]) -> str:
        name = match.group(2)
        if name in seen:
            return ""
        seen.add(name)
        return match.group(1)

    return _SLOT_CLASS_BLOCK_RE.sub(_keep_first, code)


@dataclass
class SlotAlternative:
    """A self-contained nn.Module class implementing one slot option."""

    name: str  # Short identifier, e.g. "basic_block"
    description: str  # Natural language description
    module_code: str  # Complete nn.Module class definition

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "module_code": self.module_code}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SlotAlternative":
        return cls(**d)


@dataclass
class Slot:
    """A searchable position with its module class name and alternatives."""

    name: str  # Unique slot id, e.g. "block0"
    class_name: str  # Module class name used in scaffold, e.g. "SLOT_block0"
    description: str  # What this slot controls
    input_shape_at_slot: list[int]  # Tensor shape entering this slot (no batch)
    output_shape_at_slot: list[int]  # Tensor shape exiting this slot (no batch)
    alternatives: list[SlotAlternative]
    interface: str = "single"  # "single": forward(x), "dual": forward(x, residual)
    residual_shape: list[int] | None = None  # Shape of residual tensor for dual interface
    default_idx: int = 0

    @property
    def n_alternatives(self) -> int:
        return len(self.alternatives)

    @property
    def is_dual(self) -> bool:
        return self.interface == "dual"

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name, "class_name": self.class_name,
            "description": self.description,
            "input_shape_at_slot": self.input_shape_at_slot,
            "output_shape_at_slot": self.output_shape_at_slot,
            "interface": self.interface,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "default_idx": self.default_idx,
        }
        if self.residual_shape is not None:  # M-14 fix: [] and [0,...] are valid shapes
            d["residual_shape"] = self.residual_shape
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Slot":
        return cls(
            name=d["name"], class_name=d["class_name"],
            description=d["description"],
            input_shape_at_slot=d["input_shape_at_slot"],
            output_shape_at_slot=d["output_shape_at_slot"],
            interface=d.get("interface", "single"),
            residual_shape=d.get("residual_shape"),
            alternatives=[SlotAlternative.from_dict(a) for a in d["alternatives"]],
            default_idx=d.get("default_idx", 0),
        )


@dataclass
class SlottedArchitecture:
    """Architecture with swappable nn.Module slots.

    scaffold_code: The main Architecture class that references slot modules
                   by class name (e.g., `self.block0 = SLOT_block0()`).
    slots: Named slots, each with validated module alternatives.
    realize(choices): Concatenates chosen module classes + scaffold → complete code.
    """

    scaffold_code: str  # Architecture class referencing SLOT_xxx() classes
    slots: dict[str, Slot]
    training_config: dict[str, Any]
    seed_choices: dict[str, int]
    original_seed_code: str = ""  # Verbatim Phase 1 seed code (preserved exactly)

    def realize(self, choices: dict[str, int]) -> str:
        """Generate complete model code from slot choices.

        Concatenates: imports + chosen module classes + scaffold, then deduplicates
        any repeated top-level `class SLOT_X` definitions (keeping the first). The
        dedup pass protects against scaffolds that include stub `class SLOT_X` forward
        declarations and against alternatives that accidentally redefine their class.
        Without it, Python's last-definition-wins semantics silently replaces the real
        slot implementations with identity stubs.

        Virtual slots (empty module_code, e.g., lr_mult) are skipped in code generation.
        Phase 3 reads their choices separately via `lr_mult_for()` / similar helpers.
        """
        parts = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
        ]

        # Add chosen module class for each slot
        for slot_name, slot in self.slots.items():
            alt_idx = choices.get(slot_name, slot.default_idx)
            alt = slot.alternatives[alt_idx]
            # Virtual slot: no code to append (e.g., lr_mult)
            if not alt.module_code.strip():
                continue
            parts.append(f"# Slot: {slot_name} — {alt.description}")
            parts.append(alt.module_code)
            parts.append("")

        # Add the scaffold (main Architecture class)
        parts.append(self.scaffold_code)

        return _dedupe_slot_classes("\n".join(parts))

    def realize_seed(self) -> str:
        return self.realize(self.seed_choices)

    def lr_mult_for(self, choices: dict[str, int]) -> float:
        """Return the learning-rate multiplier for the given choices.

        If the grammar has a 'lr_mult' slot, parse its description (e.g. "0.5" -> 0.5).
        Otherwise returns 1.0 (no scaling).
        """
        if "lr_mult" not in self.slots:
            return 1.0
        slot = self.slots["lr_mult"]
        idx = choices.get("lr_mult", slot.default_idx)
        try:
            return float(slot.alternatives[idx].description)
        except (ValueError, IndexError):
            return 1.0

    @property
    def slot_names(self) -> list[str]:
        return list(self.slots.keys())

    @property
    def search_space_size(self) -> int:
        size = 1
        for slot in self.slots.values():
            size *= slot.n_alternatives
        return size

    def to_dict(self) -> dict[str, Any]:
        return {
            "scaffold_code": self.scaffold_code,
            "slots": {k: v.to_dict() for k, v in self.slots.items()},
            "training_config": self.training_config,
            "seed_choices": self.seed_choices,
            "original_seed_code": self.original_seed_code,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SlottedArchitecture":
        return cls(
            scaffold_code=d["scaffold_code"],
            slots={k: Slot.from_dict(v) for k, v in d["slots"].items()},
            training_config=d.get("training_config", {}),
            seed_choices=d.get("seed_choices", {}),
            original_seed_code=d.get("original_seed_code", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "SlottedArchitecture":
        return cls.from_dict(json.loads(s))

    def save(self, path: str) -> None:
        from pathlib import Path
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> "SlottedArchitecture":
        from pathlib import Path
        return cls.from_json(Path(path).read_text())
