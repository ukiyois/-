"""Run the frozen author detector once and cache its post-NMS val_half boxes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "SU-T"
CACHE = ROOT / "data" / "detection_cache_A05"
CHECKPOINT = ROOT.parent / "Pretrained" / "Checkpoint.pth.tar"
CONFIG = ROOT / "configs" / "SU-T-ReID-ValHalf.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def main() -> None:
    if CACHE.exists():
        raise FileExistsError(f"Refusing to mix with existing detector cache: {CACHE}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from yolox.utils import postprocess

    spec = importlib.util.spec_from_file_location("sut_valhalf_exp", CONFIG)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    exp = module.Exp()
    loader = exp.get_eval_loader(1, False, run_tracking=True)

    model = exp.get_model()
    model.cuda(0).eval().half()
    checkpoint = torch.load(CHECKPOINT, map_location="cuda:0")
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    torch.backends.cudnn.benchmark = True

    frames: dict[str, list[np.ndarray]] = defaultdict(list)
    filenames: dict[str, list[str]] = defaultdict(list)
    frame_ids: dict[str, list[int]] = defaultdict(list)
    dimensions: dict[str, tuple[int, int]] = {}
    total = len(loader)
    start = time.time()

    for index, (imgs, _, info_imgs, _, _) in enumerate(loader, start=1):
        h = int(scalar(info_imgs[0]))
        w = int(scalar(info_imgs[1]))
        frame = int(scalar(info_imgs[2]))
        filename = str(info_imgs[4][0]).replace("\\", "/")
        parts = filename.split("/")
        if len(parts) < 3:
            raise ValueError(f"Unexpected MFT25 file name: {filename}")
        sequence = parts[0]
        if sequence in dimensions and dimensions[sequence] != (h, w):
            raise ValueError(f"Inconsistent image dimensions in {sequence}")
        dimensions[sequence] = (h, w)

        with torch.inference_mode():
            outputs = model(imgs.cuda(non_blocking=True).half())
            outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)[0]

        if outputs is None:
            detections = np.empty((0, 5), dtype=np.float32)
        else:
            values = outputs.detach().float().cpu().numpy()
            scale = min(exp.test_size[0] / float(h), exp.test_size[1] / float(w))
            boxes = values[:, :4] / scale
            scores = values[:, 4] * values[:, 5]
            detections = np.concatenate((boxes, scores[:, None]), axis=1).astype(np.float32)

        frames[sequence].append(detections)
        filenames[sequence].append(filename)
        frame_ids[sequence].append(frame)
        if index % 100 == 0 or index == total:
            elapsed = time.time() - start
            print(f"CACHE_PROGRESS={index}/{total} elapsed_seconds={elapsed:.1f}", flush=True)

    manifest = json.loads((ROOT / "data" / "dataset_manifest.json").read_text(encoding="utf-8"))
    expected = {item["sequence"]: int(item["val_half_frames"]) for item in manifest["sequences"]}
    if set(frames) != set(expected):
        raise RuntimeError(f"Sequence mismatch: cached={sorted(frames)} expected={sorted(expected)}")

    CACHE.mkdir(parents=True)
    seq_manifest = []
    for sequence in sorted(expected):
        if len(frames[sequence]) != expected[sequence]:
            raise RuntimeError(f"{sequence}: cached {len(frames[sequence])}, expected {expected[sequence]} frames")
        if frame_ids[sequence] != list(range(1, expected[sequence] + 1)):
            raise RuntimeError(f"Non-contiguous val_half frame IDs for {sequence}")
        offsets = [0]
        for item in frames[sequence]:
            offsets.append(offsets[-1] + len(item))
        all_dets = np.concatenate(frames[sequence], axis=0) if offsets[-1] else np.empty((0, 5), np.float32)
        output = CACHE / f"{sequence}.npz"
        np.savez_compressed(output, detections=all_dets, offsets=np.asarray(offsets, np.int64))
        seq_manifest.append({
            "sequence": sequence,
            "frames": expected[sequence],
            "height": dimensions[sequence][0],
            "width": dimensions[sequence][1],
            "detections": int(len(all_dets)),
            "image_paths": filenames[sequence],
            "frame_ids": frame_ids[sequence],
            "cache_file": output.name,
            "cache_sha256": sha256(output),
        })

    metadata = {
        "purpose": "fixed-detection local validation ablation; not the paper test-set reproduction",
        "split": "author-provided MFT25 train val_half",
        "sequences": seq_manifest,
        "frames": sum(expected.values()),
        "detections": sum(item["detections"] for item in seq_manifest),
        "detector_checkpoint": str(CHECKPOINT),
        "detector_checkpoint_sha256": sha256(CHECKPOINT),
        "input_size": list(exp.test_size),
        "test_conf": float(exp.test_conf),
        "nms_threshold": float(exp.nmsthre),
        "cache_coordinates": "original-image xyxy plus objectness*class confidence, after author YOLOX postprocess/NMS",
        "python": sys.version,
        "torch": torch.__version__,
    }
    (CACHE / "cache_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"CACHE_COMPLETE={CACHE}", flush=True)
    print(f"CACHE_FRAMES={metadata['frames']} CACHE_DETECTIONS={metadata['detections']}", flush=True)


if __name__ == "__main__":
    main()
