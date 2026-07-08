"""Worker subprocess entry point: loads model code + config, runs training on one GPU.

This script is invoked as a subprocess with CUDA_VISIBLE_DEVICES set to a single GPU.
It communicates via files:
  Input:  {work_dir}/model.py       — nn.Module class definition
          {work_dir}/config.json    — TrainingConfig as JSON
          {work_dir}/args.json      — task name, mode, time limit, etc.
  Output: {work_dir}/results.json   — TrainResult as JSON
"""

from __future__ import annotations

import importlib.util
import json
import logging


def _sanitize_for_json(obj):
    """Replace inf/-inf/NaN with JSON-safe values (2.4 fix)."""
    if isinstance(obj, float):
        if obj == float("inf"): return 1e308
        if obj == float("-inf"): return -1e308
        if obj != obj: return None  # NaN
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_sanitize_for_json(v) for v in obj)
    return obj
import sys
import traceback
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model_from_file(model_path: Path) -> torch.nn.Module:
    """Dynamically import a model.py file and instantiate Architecture class."""
    spec = importlib.util.spec_from_file_location("dynamic_model", str(model_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Architecture"):
        raise AttributeError(
            f"model.py must define a class named 'Architecture', "
            f"found: {[n for n in dir(module) if not n.startswith('_')]}"
        )
    return module.Architecture()


def run_worker(work_dir: str) -> None:
    """Main worker entry point."""
    work_dir = Path(work_dir)
    results_path = work_dir / "results.json"

    try:
        # Load args
        with open(work_dir / "args.json") as f:
            args = json.load(f)

        task_name = args["task"]
        mode = args["mode"]  # "search" or "final"
        time_limit_s = args["time_limit_s"]
        data_root = args["data_root"]
        seed = args.get("seed", 42)
        proxy_epoch_fraction = args.get("proxy_epoch_fraction", 0.3)

        # Set seeds for reproducibility (all random sources)
        import random as _random
        _random.seed(seed)  # Python random (used by SpecAugment) — N9 fix
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        import numpy as np
        np.random.seed(seed)

        # cuDNN settings: benchmark=True for search (speed), deterministic for final (reproducibility)
        is_final = (mode == "final")
        torch.backends.cudnn.deterministic = is_final
        torch.backends.cudnn.benchmark = not is_final

        # Load model
        logger.info("Loading model...")
        model = load_model_from_file(work_dir / "model.py")

        # Load training config
        from agentnas.training.config import TrainingConfig
        with open(work_dir / "config.json") as f:
            config = TrainingConfig.from_dict(json.load(f))

        # Load benchmark
        from agentnas.benchmarks.registry import get_benchmark
        benchmark = get_benchmark(task_name, data_root=data_root)

        # Select device (CUDA_VISIBLE_DEVICES already set, so device 0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")

        # Validate model: forward pass with dummy input
        logger.info("Validating model (dummy forward pass)...")
        model_test = model.to(device)
        model_test.eval()
        dummy = torch.randn(1, *benchmark.input_shape(), device=device)
        with torch.no_grad():
            out = model_test(dummy)
        model_test.train()  # Restore training mode for actual training
        logger.info(f"Model output shape: {out.shape}")

        # Validate output shape matches benchmark spec
        spec = benchmark.output_spec()
        if "num_classes" in spec:
            expected = (1, spec["num_classes"])
        else:
            expected = (1, *spec["output_shape"])
        if tuple(out.shape) != expected:
            raise ValueError(
                f"Output shape mismatch: model produces {tuple(out.shape)}, "
                f"expected {expected}"
            )

        # M-03: VRAM dry run — check if forward pass VRAM is reasonable
        if torch.cuda.is_available():
            forward_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            logger.info(f"Forward pass VRAM: {forward_vram_mb:.0f} MB")
            # Rough estimate: backward uses ~2x forward memory
            estimated_training_vram = forward_vram_mb * 2.5
            max_vram = args.get("max_vram_mb", 0)
            if max_vram > 0 and estimated_training_vram > max_vram:
                raise ValueError(
                    f"Estimated training VRAM ({estimated_training_vram:.0f} MB) "
                    f"exceeds limit ({max_vram} MB). Forward-only: {forward_vram_mb:.0f} MB."
                )

        model_test = model_test.cpu()
        del dummy, out
        torch.cuda.empty_cache()

        # Run training
        from agentnas.training.harness import train_search, train_final
        if mode == "search":
            result = train_search(
                model=model, config=config, benchmark=benchmark,
                device=device, time_limit_s=time_limit_s,
                save_dir=work_dir,
                proxy_epoch_fraction=proxy_epoch_fraction,
            )
        else:
            result = train_final(
                model=model, config=config, benchmark=benchmark,
                device=device, time_limit_s=time_limit_s,
                save_dir=work_dir,
            )

        # Write results
        result_dict = {
            "metric": result.metric,
            "wall_time_s": result.wall_time_s,
            "peak_vram_mb": result.peak_vram_mb,
            "params": result.params,
            "flops": result.flops,
            "epochs_completed": result.epochs_completed,
            "status": result.status,
            "error_message": result.error_message,
            "train_losses": result.train_losses,
            "val_metrics": result.val_metrics,
            "best_epoch": result.best_epoch,
        }
        with open(results_path, "w") as f:
            json.dump(_sanitize_for_json(result_dict), f, indent=2)
        logger.info(f"Done: metric={result.metric:.4f}, status={result.status}")

    except Exception as e:
        logger.error(f"Worker failed: {e}")
        logger.error(traceback.format_exc())
        # Write error result
        error_result = {
            "metric": float("inf"),
            "wall_time_s": 0,
            "peak_vram_mb": 0,
            "params": 0,
            "flops": 0,
            "epochs_completed": 0,
            "status": "error",
            "error_message": f"{type(e).__name__}: {e}",
            "train_losses": [],
            "val_metrics": [],
            "best_epoch": -1,
        }
        with open(results_path, "w") as f:
            json.dump(_sanitize_for_json(error_result), f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <work_dir>")
        sys.exit(1)
    run_worker(sys.argv[1])
