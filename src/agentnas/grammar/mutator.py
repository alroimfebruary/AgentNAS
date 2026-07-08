"""Slot-based mutations for evolutionary search.

Simple and guaranteed valid: swap a slot's alternative index.
No code generation at mutation time.

Coarse-effect slots (e.g., lr_mult) get up-weighted selection probability because
they affect training more coarsely than per-slot architectural choices.
"""

from __future__ import annotations

import random

from agentnas.grammar.slots import SlottedArchitecture


# Slots whose mutation is weighted higher because a single change has large
# fitness impact relative to swapping one architectural op.
COARSE_SLOTS: dict[str, float] = {
    "lr_mult": 3.0,
}


def _slot_probs(slot_names: list[str]) -> list[float]:
    """Build per-slot selection probabilities with COARSE_SLOTS up-weighted."""
    weights = [COARSE_SLOTS.get(name, 1.0) for name in slot_names]
    total = sum(weights)
    return [w / total for w in weights]


def _pick_slot(slot_names: list[str]) -> str:
    """Pick one slot respecting COARSE_SLOTS up-weighting."""
    probs = _slot_probs(slot_names)
    return random.choices(slot_names, weights=probs, k=1)[0]


def _pick_slots(slot_names: list[str], n: int) -> list[str]:
    """Pick n distinct slots respecting COARSE_SLOTS up-weighting."""
    n = min(n, len(slot_names))
    remaining = list(slot_names)
    picked: list[str] = []
    for _ in range(n):
        if not remaining:
            break
        probs = _slot_probs(remaining)
        idx = random.choices(range(len(remaining)), weights=probs, k=1)[0]
        picked.append(remaining.pop(idx))
    return picked


def slot_mutation(
    choices: dict[str, int],
    slotted_arch: SlottedArchitecture,
) -> dict[str, int]:
    """Mutate by changing one random slot to a different alternative.

    Returns a new choices dict (does not modify the input).
    """
    new_choices = dict(choices)
    slot_names = slotted_arch.slot_names

    if not slot_names:
        return new_choices

    # Pick a random slot (up-weights COARSE_SLOTS like lr_mult)
    slot_name = _pick_slot(slot_names)
    slot = slotted_arch.slots[slot_name]

    if slot.n_alternatives <= 1:
        return new_choices

    # Pick a different alternative
    current = new_choices.get(slot_name, slot.default_idx)
    candidates = [i for i in range(slot.n_alternatives) if i != current]
    new_choices[slot_name] = random.choice(candidates)

    return new_choices


def multi_slot_mutation(
    choices: dict[str, int],
    slotted_arch: SlottedArchitecture,
    n_slots: int = 2,
) -> dict[str, int]:
    """Mutate by changing multiple DISTINCT random slots (H4 fix)."""
    new_choices = dict(choices)
    slot_names = slotted_arch.slot_names

    # Pick n_slots distinct slots (up-weights COARSE_SLOTS like lr_mult)
    selected_slots = _pick_slots(slot_names, n_slots)

    for slot_name in selected_slots:
        slot = slotted_arch.slots[slot_name]
        if slot.n_alternatives <= 1:
            continue
        current = new_choices.get(slot_name, slot.default_idx)
        candidates = [i for i in range(slot.n_alternatives) if i != current]
        new_choices[slot_name] = random.choice(candidates)

    return new_choices


def random_choices(slotted_arch: SlottedArchitecture) -> dict[str, int]:
    """Generate completely random slot choices."""
    return {
        name: random.randint(0, slot.n_alternatives - 1)
        for name, slot in slotted_arch.slots.items()
    }


def crossover(
    parent1: dict[str, int],
    parent2: dict[str, int],
    slotted_arch: SlottedArchitecture,
) -> dict[str, int]:
    """Per-slot uniform crossover between two parents.

    For each slot, randomly inherit the alternative index from either parent
    (independent 50/50 per slot). Used in 'crossover' search mode.

    Returns a new choices dict; does not modify inputs.
    """
    new_choices = {}
    for slot_name in slotted_arch.slot_names:
        p1_alt = parent1.get(slot_name, slotted_arch.slots[slot_name].default_idx)
        p2_alt = parent2.get(slot_name, slotted_arch.slots[slot_name].default_idx)
        new_choices[slot_name] = p1_alt if random.random() < 0.5 else p2_alt
    return new_choices


def choices_signature(choices: dict[str, int]) -> str:
    """Hashable string representation of slot choices for diversity tracking."""
    return "|".join(f"{k}={v}" for k, v in sorted(choices.items()))
