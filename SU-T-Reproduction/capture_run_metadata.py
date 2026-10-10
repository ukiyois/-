"""Capture environment and input-file provenance before an inference run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--reid", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "free_memory_bytes_at_capture": free,
            "cuda_runtime": torch.version.cuda,
        }

    packages = {}
    for name in (
        "numpy", "torch", "torchvision", "opencv-python", "pycocotools",
        "motmetrics", "loguru", "lap", "cython-bbox", "filterpy",
        "faiss-cpu", "xmltodict", "trackeval",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "packages": packages,
        "repository": {"path": str(repo), "commit": commit, "working_tree_status": status},
        "inputs": {
            "mft25_train_archive": file_info(args.archive),
            "detector_checkpoint": file_info(args.detector),
            "reid_checkpoint": file_info(args.reid),
            "dataset_manifest": str(args.dataset_manifest.resolve()),
        },
        "evaluation": {
            "split": "MFT25 author-provided val_half, 8 training sequences",
            "training": False,
            "batch_size": 1,
            "fp16": True,
            "fuse": False,
            "input_size": [800, 1440],
            "test_conf": 0.1,
            "nms_threshold": 0.7,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "commit": commit, "gpu": gpu}, indent=2))


if __name__ == "__main__":
    main()
