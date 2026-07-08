"""Smoke tests for benchmark data loaders."""

import sys
from pathlib import Path

NB360_ROOT = str(Path(__file__).parent.parent / "NAS-Bench-360-Benchmarks")


def test_cifar100():
    from agentnas.benchmarks.cifar100 import CIFAR100Benchmark
    b = CIFAR100Benchmark(NB360_ROOT)

    tl, vl = b.get_search_loaders(4, num_workers=0)
    assert len(tl.dataset) == 40000, f"Expected 40000 train, got {len(tl.dataset)}"
    assert len(vl.dataset) == 10000, f"Expected 10000 val, got {len(vl.dataset)}"

    batch = next(iter(tl))
    x, y = batch[0], batch[1]
    assert x.shape == (4, 3, 32, 32), f"Bad shape: {x.shape}"
    assert y.min() >= 0 and y.max() <= 99, f"Bad labels: [{y.min()}, {y.max()}]"

    tvl, tel = b.get_final_loaders(4, num_workers=0)
    assert len(tvl.dataset) == 50000, f"Expected 50000 trainval, got {len(tvl.dataset)}"
    assert len(tel.dataset) == 10000, f"Expected 10000 test, got {len(tel.dataset)}"
    print("CIFAR-100: PASS")


def test_ecg():
    from agentnas.benchmarks.ecg import ECGBenchmark
    b = ECGBenchmark(NB360_ROOT)

    tl, vl = b.get_search_loaders(4, num_workers=0)
    batch = next(iter(tl))
    x, y = batch[0], batch[1]
    assert x.shape[1:] == (1, 1000), f"Bad shape: {x.shape}"
    assert y.min() >= 0 and y.max() <= 3, f"Bad labels: [{y.min()}, {y.max()}]"
    print(f"ECG search: train={len(tl.dataset)}, val={len(vl.dataset)}")
    print("ECG: PASS")


def test_darcyflow():
    from agentnas.benchmarks.darcyflow import DarcyFlowBenchmark
    b = DarcyFlowBenchmark(NB360_ROOT)

    tl, vl = b.get_search_loaders(4, num_workers=0)
    assert len(tl.dataset) == 900, f"Expected 900 train, got {len(tl.dataset)}"
    assert len(vl.dataset) == 100, f"Expected 100 val, got {len(vl.dataset)}"

    batch = next(iter(tl))
    x, y = batch[0], batch[1]
    assert x.shape == (4, 3, 85, 85), f"Bad input shape: {x.shape}"
    assert y.shape == (4, 1, 85, 85), f"Bad output shape: {y.shape}"
    print("DarcyFlow: PASS")


def test_fsd50k():
    from agentnas.benchmarks.fsd50k import FSD50KBenchmark
    b = FSD50KBenchmark(NB360_ROOT)

    tl, vl = b.get_search_loaders(4, num_workers=0)
    batch = next(iter(tl))  # FSD50K returns (mel, target, clip_id)
    x, y = batch[0], batch[1]
    assert x.shape[1:] == (1, 96, 101), f"Bad shape: {x.shape}"
    assert y.shape == (4, 200), f"Bad label shape: {y.shape}"
    print(f"FSD50K search: train={len(tl.dataset)}, val={len(vl.dataset)}")
    print("FSD50K: PASS")


if __name__ == "__main__":
    tests = {
        "cifar100": test_cifar100,
        "ecg": test_ecg,
        "darcyflow": test_darcyflow,
        "fsd50k": test_fsd50k,
    }

    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "all":
        for name, fn in tests.items():
            try:
                fn()
            except Exception as e:
                print(f"{name}: FAIL - {e}")
    elif target in tests:
        tests[target]()
    else:
        print(f"Unknown test: {target}. Available: {list(tests.keys())}")
