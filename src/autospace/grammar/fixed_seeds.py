"""Fixed seed architectures for the Phase-1 ablation.

When ``config.fixed_seed_arch`` is set, Phase 1 skips its LLM iterative loop
and seeds the pipeline with a hardcoded architecture instead — ResNet-18 for
Unseen-NAS tasks, WideResNet-16-4 for NAS-Bench-360 tasks (replicating
``NAS-Bench-360/backbone/backbone_pt.py`` for classification and
``NAS-Bench-360/backbone/backbone_grid_wrn.py`` for DarcyFlow regression).

The module emits standalone ``model.py`` source strings that the existing
training worker consumes unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

TASK_TO_FAMILY: dict[str, str] = {
    "language": "resnet18",
    "cifartile": "resnet18",
    "chesseract": "resnet18",
    "spherical": "wideresnet_16_4",
    "ninapro": "wideresnet_16_4",
    "darcyflow": "wideresnet_16_4",
    "fsd50k": "wideresnet_16_4",
    "ecg": "wideresnet_16_4",
}

UNSEEN_TASKS: set[str] = {"language", "cifartile", "chesseract"}

_VALID_FAMILIES = {"resnet18", "wideresnet_16_4"}


def resolve_family(task: str, requested: str | None) -> str:
    if not requested:
        raise ValueError("fixed_seed_arch is empty")
    if requested == "auto":
        if task not in TASK_TO_FAMILY:
            raise ValueError(
                f"Task '{task}' has no fixed-seed family mapping. "
                f"Supported: {sorted(TASK_TO_FAMILY)}"
            )
        return TASK_TO_FAMILY[task]
    if requested not in _VALID_FAMILIES:
        raise ValueError(
            f"Unknown fixed_seed_arch '{requested}'. "
            f"Valid: 'auto', {sorted(_VALID_FAMILIES)}"
        )
    return requested


def source_explorer_path(task: str, project_root: Path) -> Path:
    """Path to the seed42 explorer report we reuse instead of re-running exploration."""
    suffix = "_blind" if task in UNSEEN_TASKS else ""
    return (
        project_root
        / "experiments"
        / f"{task}_claude-sonnet-4-6_seed42{suffix}"
        / "round_01"
        / "phase1"
        / "explorer_report.md"
    )


def build_fixed_seed_code(
    family: str,
    input_shape: tuple[int, ...],
    output_spec: dict[str, Any],
    task: str,
) -> tuple[str, dict[str, Any]]:
    """Return ``(model_py_source, training_config_dict)`` for the chosen family."""
    if family == "resnet18":
        if "num_classes" not in output_spec:
            raise ValueError(
                f"resnet18 family expects classification output_spec with 'num_classes', "
                f"got {output_spec}"
            )
        in_channels = input_shape[0]
        num_classes = output_spec["num_classes"]
        return _resnet18_classification(in_channels, num_classes), _resnet18_train_config()

    if family == "wideresnet_16_4":
        if "num_classes" in output_spec:
            in_channels = input_shape[0]
            num_classes = output_spec["num_classes"]
            # 2D input (C,H,W) → 2D conv body; 1D input (C,L) → 1D conv body.
            if len(input_shape) == 3:
                code = _wrn_16_4_classification(in_channels, num_classes)
            elif len(input_shape) == 2:
                code = _wrn_16_4_1d_classification(in_channels, num_classes)
            else:
                raise ValueError(
                    f"wideresnet_16_4 classification expects 2D or 3D input_shape, "
                    f"got {input_shape}"
                )
            return code, _wrn_classification_train_config(task)
        if "output_shape" in output_spec:
            in_channels = input_shape[0]
            out_channels = output_spec["output_shape"][0]
            return (
                _wrn_16_4_dense_regression(in_channels, out_channels),
                _wrn_dense_train_config(),
            )
        raise ValueError(
            f"wideresnet_16_4 family needs 'num_classes' or 'output_shape' "
            f"in output_spec, got {output_spec}"
        )

    raise ValueError(f"Unknown family '{family}'")


def _resnet18_classification(in_channels: int, num_classes: int) -> str:
    """ResNet-18 with 3x3 stride-1 stem (small-image variant for Unseen-NAS).

    [2,2,2,2] BasicBlocks, channels [64,128,256,512], strides [1,2,2,2].
    """
    return f"""\
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


def _make_stage(in_planes, planes, num_blocks, stride):
    layers = [BasicBlock(in_planes, planes, stride)]
    for _ in range(num_blocks - 1):
        layers.append(BasicBlock(planes, planes, 1))
    return nn.Sequential(*layers)


class Architecture(nn.Module):
    \"\"\"ResNet-18 (small-image stem). Fixed seed for Unseen-NAS ablation.\"\"\"

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d({in_channels}, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _make_stage(64, 64, 2, stride=1)
        self.stage2 = _make_stage(64, 128, 2, stride=2)
        self.stage3 = _make_stage(128, 256, 2, stride=2)
        self.stage4 = _make_stage(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, {num_classes})

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)
"""


def _wrn_basic_block_src() -> str:
    """Pre-activation BasicBlock used by both WRN variants (NB360 backbone style)."""
    return """\
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes and stride == 1)
        if not self.equalInOut:
            self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                          stride=stride, padding=0, bias=False)
        else:
            self.convShortcut = None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out = x
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        shortcut = x if self.equalInOut else self.convShortcut(x)
        return out + shortcut


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        layers = []
        for i in range(int(nb_layers)):
            in_p = in_planes if i == 0 else out_planes
            s = stride if i == 0 else 1
            layers.append(BasicBlock(in_p, out_planes, s, dropRate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
"""


def _wrn_16_4_classification(in_channels: int, num_classes: int) -> str:
    """WideResNet-16-4 classification — replicates NB360 backbone_pt.Backbone_Pt.

    depth=16, widen_factor=4 → channels [16,64,128,256], n=2 blocks/stage,
    strides [1,2,2]. Final: BN+ReLU+AdaptiveAvgPool+Linear(256,num_classes).
    """
    block = _wrn_basic_block_src()
    return f"""\
import torch
import torch.nn as nn
import torch.nn.functional as F


{block}

class Architecture(nn.Module):
    \"\"\"WideResNet-16-4 (depth=16, width=4) classification head.

    Mirrors NAS-Bench-360/backbone/backbone_pt.py:Backbone_Pt for the NB360
    classification ablation (Spherical, Ninapro).
    \"\"\"

    def __init__(self):
        super().__init__()
        nChannels = [16, 16 * 4, 32 * 4, 64 * 4]  # [16,64,128,256]
        n = (16 - 4) // 6  # 2 blocks per stage
        self.conv1 = nn.Conv2d({in_channels}, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], stride=1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], stride=2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], stride=2)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nChannels[3], {num_classes})

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out).flatten(1)
        return self.fc(out)
"""


def _wrn_basic_block_1d_src() -> str:
    """1D pre-activation BasicBlock — Conv1d/BatchNorm1d analog of the 2D version."""
    return """\
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes and stride == 1)
        if not self.equalInOut:
            self.convShortcut = nn.Conv1d(in_planes, out_planes, kernel_size=1,
                                          stride=stride, padding=0, bias=False)
        else:
            self.convShortcut = None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out = x
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        shortcut = x if self.equalInOut else self.convShortcut(x)
        return out + shortcut


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        layers = []
        for i in range(int(nb_layers)):
            in_p = in_planes if i == 0 else out_planes
            s = stride if i == 0 else 1
            layers.append(BasicBlock(in_p, out_planes, s, dropRate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
"""


def _wrn_16_4_1d_classification(in_channels: int, num_classes: int) -> str:
    """WideResNet-16-4 1D classification — 1D analog of Backbone_Pt for ECG.

    depth=16, widen_factor=4 → channels [16,64,128,256], n=2 blocks/stage,
    strides [1,2,2]. Final: BN+ReLU+AdaptiveAvgPool1d+Linear(256,num_classes).

    Used when input rank is 2 (C, L) — 1D signal classification (e.g., ECG).
    """
    block = _wrn_basic_block_1d_src()
    return f"""\
import torch
import torch.nn as nn
import torch.nn.functional as F


{block}

class Architecture(nn.Module):
    \"\"\"WideResNet-16-4 1D classification head for 1D-signal NB360 tasks (ECG).\"\"\"

    def __init__(self):
        super().__init__()
        nChannels = [16, 16 * 4, 32 * 4, 64 * 4]  # [16,64,128,256]
        n = (16 - 4) // 6  # 2 blocks per stage
        self.conv1 = nn.Conv1d({in_channels}, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], stride=1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], stride=2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], stride=2)
        self.bn1 = nn.BatchNorm1d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nChannels[3], {num_classes})

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out).flatten(1)
        return self.fc(out)
"""


def _wrn_16_4_dense_regression(in_channels: int, out_channels: int) -> str:
    """WideResNet-16-4 dense regression — replicates NB360 backbone_grid_wrn.Backbone.

    All three stages stride=1 (preserve spatial resolution). Per-pixel Linear
    head: BN+ReLU → permute(0,2,3,1) → Linear(256, out_channels) → permute(0,3,1,2).
    """
    block = _wrn_basic_block_src()
    return f"""\
import torch
import torch.nn as nn
import torch.nn.functional as F


{block}

class Architecture(nn.Module):
    \"\"\"WideResNet-16-4 dense head for DarcyFlow regression.

    Mirrors NAS-Bench-360/backbone/backbone_grid_wrn.py:Backbone — all blocks
    stride=1 to preserve spatial resolution; per-pixel Linear projection.
    \"\"\"

    def __init__(self):
        super().__init__()
        nChannels = [16, 16 * 4, 32 * 4, 64 * 4]  # [16,64,128,256]
        n = (16 - 4) // 6  # 2 blocks per stage
        self.conv1 = nn.Conv2d({in_channels}, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], stride=1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], stride=1)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], stride=1)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], {out_channels})

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.fc(out)
        return out.permute(0, 3, 1, 2).contiguous()
"""


def _resnet18_train_config() -> dict[str, Any]:
    """Hyperparameters for Unseen-NAS ResNet-18 (per user 2026-04-28)."""
    return {
        "optimizer": "sgd",
        "lr": 0.04,
        "weight_decay": 3e-4,
        "momentum": 0.9,
        "scheduler": "cosine",
        "scheduler_params": {},
        "warmup_epochs": 0,
        "batch_size": 256,
        "epochs": 64,
        "augmentations": [],
        "augmentation_params": {},
        "grad_clip": 1.0,
        "label_smoothing": 0.0,
    }


def _wrn_classification_train_config(task: str) -> dict[str, Any]:
    """NB360 classification hyperparameters.

    Spherical and Ninapro use bs=128 per the NB360 baseline yamls
    (`scifar100.yaml`, `ninapro.yaml`). FSD50K and ECG use bs=256 per
    `audio.yaml` and `ECG.yaml`. All share lr=0.1, wd=5e-4, momentum=0.9,
    epochs=200, cosine schedule.
    """
    bs = 256 if task in {"fsd50k", "ecg"} else 128
    return {
        "optimizer": "sgd",
        "lr": 0.1,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "scheduler": "cosine",
        "scheduler_params": {},
        "warmup_epochs": 0,
        "batch_size": bs,
        "epochs": 200,
        "augmentations": [],
        "augmentation_params": {},
        "grad_clip": 1.0,
        "label_smoothing": 0.0,
    }


def _wrn_dense_train_config() -> dict[str, Any]:
    """DarcyFlow regression hyperparameters."""
    return {
        "optimizer": "sgd",
        "lr": 0.001,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "scheduler": "cosine",
        "scheduler_params": {},
        "warmup_epochs": 0,
        "batch_size": 4,
        "epochs": 200,
        "augmentations": [],
        "augmentation_params": {},
        "grad_clip": 1.0,
        "label_smoothing": 0.0,
    }
