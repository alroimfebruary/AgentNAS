"""GDAS supernet construction over autospace's slot-based grammar.

Each searchable slot in a `SlottedArchitecture` is replaced by a `MixedSlot`
that holds all alternatives instantiated as nn.Modules plus a learnable
architecture parameter `alpha`. Forward sampling uses Gumbel-softmax with
straight-through estimation: forward executes one alternative (selected by
argmax of the Gumbel sample), backward propagates the soft Gumbel gradient
into `alpha`. Only one branch's activations are materialized per forward —
this is what makes the supernet fit on a single 2080ti.

Public API:
- `MixedSlot(slot)` — supernet block for one slot.
- `build_supernet(slotted) -> (Architecture, dict[name, MixedSlot])`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from autospace.grammar.slots import Slot, SlotAlternative, SlottedArchitecture


def _instantiate_alternative(alt: SlotAlternative, class_name: str) -> nn.Module:
    """Exec the alternative's module_code and instantiate the SLOT_xxx class."""
    ns: dict = {}
    exec(alt.module_code, ns)
    cls = ns.get(class_name)
    if cls is None:
        raise RuntimeError(
            f"alternative '{alt.name}' does not define class {class_name!r} "
            f"(namespace keys: {sorted(k for k in ns if not k.startswith('_'))})"
        )
    return cls()


class MixedSlot(nn.Module):
    """One supernet slot: alpha-weighted Gumbel-STE mixture over N alternatives."""

    def __init__(self, slot: Slot):
        super().__init__()
        self.slot_name = slot.name
        self.is_dual = slot.is_dual
        self.ops = nn.ModuleList(
            _instantiate_alternative(a, slot.class_name) for a in slot.alternatives
        )
        self.alpha = nn.Parameter(torch.zeros(len(slot.alternatives)))
        # Set per-batch by the search loop. Buffer (not Parameter) so it travels
        # with .to(device) but does not appear in optimizer param lists.
        self.register_buffer("tau", torch.tensor(1.0))

    def set_tau(self, tau: float) -> None:
        self.tau.fill_(float(tau))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        # gumbel_softmax(hard=True) returns a one-hot vector forward, but
        # autograd flows through the soft (relaxed) sample to `alpha`.
        weights = F.gumbel_softmax(self.alpha, tau=float(self.tau.item()), hard=True, dim=-1)
        k = int(weights.argmax().item())
        if self.is_dual:
            y = self.ops[k](x, residual)
        else:
            y = self.ops[k](x)
        # Multiply by weights[k] (==1.0 forward, soft-grad backward) so dL/dα is non-zero.
        return weights[k].to(y.dtype) * y


def _bind_slot_factory(ns: dict, class_name: str, instance: nn.Module) -> None:
    """Bind a SLOT_xxx name in `ns` to a 0-arg factory returning the prebuilt instance.

    The default-arg `inst=instance` trick captures the current loop value (avoids
    the late-binding gotcha that all closures share the last loop value).
    """
    ns[class_name] = (lambda inst=instance: lambda: inst)()


def build_supernet(
    slotted: SlottedArchitecture,
) -> tuple[nn.Module, dict[str, MixedSlot]]:
    """Build the GDAS supernet from a SlottedArchitecture.

    Returns (architecture_instance, mixed_slots) where mixed_slots maps the
    slot name → its MixedSlot. Slots with `interface == "special"` (e.g.
    lr_mult) and slots with empty module_code are skipped — they aren't in
    the scaffold's namespace anyway.

    Single-alternative slots (n_alternatives == 1) are instantiated directly
    instead of wrapped in a MixedSlot, so they don't appear in mixed_slots
    (no alpha to learn).
    """
    mixed: dict[str, MixedSlot] = {}
    singletons: dict[str, nn.Module] = {}

    for name, slot in slotted.slots.items():
        if slot.interface == "special":
            continue
        if not slot.alternatives or not slot.alternatives[0].module_code.strip():
            continue
        if slot.n_alternatives == 1:
            singletons[name] = _instantiate_alternative(slot.alternatives[0], slot.class_name)
        else:
            mixed[name] = MixedSlot(slot)

    ns: dict = {"torch": torch, "nn": nn, "F": F}
    exec("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n", ns)

    # Bind every searchable slot's class name to a 0-arg factory.
    for name, slot in slotted.slots.items():
        if slot.interface == "special":
            continue
        if not slot.alternatives or not slot.alternatives[0].module_code.strip():
            continue
        if name in mixed:
            _bind_slot_factory(ns, slot.class_name, mixed[name])
        elif name in singletons:
            _bind_slot_factory(ns, slot.class_name, singletons[name])

    exec(slotted.scaffold_code, ns)
    arch_cls = ns.get("Architecture")
    if arch_cls is None:
        raise RuntimeError("scaffold_code did not define class Architecture")

    # Some scaffolds' `_init_weights` walk `self.modules()` and unconditionally
    # call `nn.init.zeros_(m.bias)` / `nn.init.constant_(m.bias, 0)` on every
    # Conv/Linear. Alternatives with `bias=False` (e.g. depth-wise conv blocks)
    # have `m.bias is None` and crash the cast. Temporarily make those init
    # routines None-tolerant for the duration of arch_cls() construction.
    import torch.nn.init as _init
    _orig_zeros, _orig_constant, _orig_normal = _init.zeros_, _init.constant_, _init.normal_

    def _tolerant_zeros(t):
        return None if t is None else _orig_zeros(t)

    def _tolerant_constant(t, v):
        return None if t is None else _orig_constant(t, v)

    def _tolerant_normal(t, mean=0.0, std=1.0):
        return None if t is None else _orig_normal(t, mean, std)

    _init.zeros_ = _tolerant_zeros
    _init.constant_ = _tolerant_constant
    _init.normal_ = _tolerant_normal
    try:
        architecture = arch_cls()
    finally:
        _init.zeros_, _init.constant_, _init.normal_ = _orig_zeros, _orig_constant, _orig_normal
    return architecture, mixed


def supernet_choices(mixed: dict[str, MixedSlot], slotted: SlottedArchitecture) -> dict[str, int]:
    """Discretize the supernet by argmax(alpha) per MixedSlot.

    Slots not in `mixed` (singletons, lr_mult, etc.) get their default_idx —
    which for lr_mult is the seed choice (lr_mult_for collapses to 1.0).
    """
    choices: dict[str, int] = {}
    for name, ms in mixed.items():
        choices[name] = int(ms.alpha.argmax().item())
    for name, slot in slotted.slots.items():
        choices.setdefault(name, slot.default_idx)
    return choices
