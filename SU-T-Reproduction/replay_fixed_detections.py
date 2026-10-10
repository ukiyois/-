"""Replay one frozen post-NMS detection cache through the planned trackers."""

from __future__ import annotations

import configparser
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "SU-T"
CACHE = ROOT / "data" / "detection_cache_A05"
OUTPUT_ROOT = ROOT / "runs" / "fixed_detection_A05"
DATA_ROOT = REPO / "datasets" / "mft25"


def sequence_info(sequence: str) -> tuple[int, int, int]:
    parser = configparser.ConfigParser()
    parser.read(ROOT / "data" / "ground_truth" / sequence / "seqinfo.ini", encoding="utf-8-sig")
    section = parser["Sequence"]
    return int(section["imHeight"]), int(section["imWidth"]), int(section["seqLength"])


def save_mot(path: Path, rows: list[tuple[int, int, float, float, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for frame, track_id, x, y, w, h in rows:
            handle.write(f"{frame},{track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},1,-1,-1,-1\n")


def make_args(motion_filter: str = "ukf", asso: str = "iou") -> SimpleNamespace:
    return SimpleNamespace(
        motion_filter=motion_filter,
        track_thresh=0.6,
        track_buffer=30,
        min_hits=3,
        iou_thresh=0.25,
        inertia=0.05,
        deltat=3,
        min_box_area=10.0,
        asso=asso,
        use_byte=True,
        TCM_first_step=True,
        TCM_byte_step=True,
        TCM_first_step_weight=1.0,
        TCM_byte_step_weight=1.0,
        low_thresh=0.1,
        with_reid=False,
        with_fastreid=True,
        fast_reid_config="fast_reid/configs/SBS_S101.yml",
        fast_reid_weights=str(ROOT.parent / "Pretrained" / "ReID-Checkpoint.pth"),
        EG_weight_high_score=1.3,
        EG_weight_low_score=1.2,
        with_longterm_reid=False,
        longterm_reid_weight=0.0,
        with_longterm_reid_correction=True,
        longterm_reid_correction_thresh=0.4,
        longterm_reid_correction_thresh_low=0.4,
        longterm_bank_length=30,
        dataset="mft25",
        ECC=False,
        adapfs=False,
        alpha=0.8,
        high_score_matching_thresh=0.8,
        low_score_matching_thresh=0.5,
        mot20=False,
        match_thresh=0.9,
    )


def load_cache(sequence: str):
    cache = np.load(CACHE / f"{sequence}.npz")
    meta = next(item for item in json.loads((CACHE / "cache_manifest.json").read_text(encoding="utf-8"))["sequences"] if item["sequence"] == sequence)
    detections = cache["detections"]
    offsets = cache["offsets"]
    if len(offsets) != meta["frames"] + 1:
        raise RuntimeError(f"Bad cache offsets for {sequence}")
    return meta, detections, offsets


def detect_outputs(sequence: str, detections: np.ndarray, offsets: np.ndarray, frame_index: int) -> np.ndarray:
    return detections[offsets[frame_index]:offsets[frame_index + 1]].copy()


def mot_rows_from_xyxy(frame: int, outputs: np.ndarray, min_box_area: float = 10.0):
    rows = []
    for item in outputs:
        x1, y1, x2, y2, track_id = map(float, item[:5])
        w, h = x2 - x1, y2 - y1
        if w * h > min_box_area:
            rows.append((frame, int(track_id), x1, y1, w, h))
    return rows


def run_sut(name: str, sequences: list[str], motion: str, association: str, reid: bool = False) -> Path:
    if reid:
        from fast_reid.fast_reid_interfece import FastReIDInterface
        from trackers.SUT_reid_UKF import SU_T_ReID
    else:
        from trackers.SUT_UKF import SU_T

    args = make_args(motion, association)
    args.with_reid = reid
    if reid:
        encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, "cuda")
    outdir = OUTPUT_ROOT / name
    outdir.mkdir(parents=True, exist_ok=False)
    processed = 0

    for sequence in sequences:
        meta, detections, offsets = load_cache(sequence)
        height, width, expected = sequence_info(sequence)
        if expected != meta["frames"] or (height, width) != (meta["height"], meta["width"]):
            raise RuntimeError(f"Dataset/cache metadata mismatch in {sequence}")
        tracker = (
            SU_T_ReID(args, det_thresh=args.track_thresh, iou_threshold=args.iou_thresh,
                      asso_func=association, delta_t=args.deltat, inertia=args.inertia)
            if reid else
            SU_T(args, det_thresh=args.track_thresh, iou_threshold=args.iou_thresh,
                 asso_func=association, delta_t=args.deltat, inertia=args.inertia,
                 use_byte=args.use_byte)
        )
        rows = []
        for idx, frame in enumerate(meta["frame_ids"]):
            det = detect_outputs(sequence, detections, offsets, idx)
            features = None
            if reid:
                if len(det):
                    relative = meta["image_paths"][idx]
                    image_path = DATA_ROOT / "train" / Path(relative)
                    raw_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if raw_image is None:
                        raise FileNotFoundError(f"Could not decode image: {image_path}")
                    features = encoder.inference(raw_image, det[:, :4])
                else:
                    features = np.empty((0, 2048), dtype=np.float32)
            # Cached boxes are in original-image coordinates, so both shape arguments
            # are original dimensions; tracker scaling is exactly 1 for every method.
            online = (
                tracker.update(det, (height, width), (height, width), id_feature=features)
                if reid else tracker.update(det, (height, width), (height, width))
            )
            if len(online):
                rows.extend(mot_rows_from_xyxy(frame, online))
            processed += 1
            if processed % 500 == 0:
                print(f"REPLAY_PROGRESS={name} frames={processed}", flush=True)
        save_mot(outdir / f"{sequence}.txt", rows)
        print(f"REPLAY_SEQUENCE_COMPLETE={name}/{sequence} tracks={len(rows)}", flush=True)
    return outdir


def run_external(name: str, sequences: list[str], kind: str) -> Path:
    from trackers.byte_tracker.byte_tracker import BYTETracker
    from trackers.ocsort_tracker.ocsort import OCSort

    args = make_args("kf", "iou")
    args.asso = "iou"
    outdir = OUTPUT_ROOT / name
    outdir.mkdir(parents=True, exist_ok=False)
    processed = 0

    for sequence in sequences:
        meta, detections, offsets = load_cache(sequence)
        height, width, expected = sequence_info(sequence)
        if expected != meta["frames"] or (height, width) != (meta["height"], meta["width"]):
            raise RuntimeError(f"Dataset/cache metadata mismatch in {sequence}")
        tracker = (
            BYTETracker(args, frame_rate=30)
            if kind == "bytetrack" else
            OCSort(det_thresh=args.track_thresh, max_age=args.track_buffer, min_hits=args.min_hits,
                   iou_threshold=args.iou_thresh, delta_t=args.deltat,
                   asso_func="iou", inertia=args.inertia, use_byte=False)
        )
        rows = []
        for idx, frame in enumerate(meta["frame_ids"]):
            det = detect_outputs(sequence, detections, offsets, idx)
            online = tracker.update(det, (height, width), (height, width))
            if kind == "bytetrack":
                frame_outputs = [np.r_[track.tlbr, track.track_id] for track in online]
            else:
                frame_outputs = online
            if len(frame_outputs):
                rows.extend(mot_rows_from_xyxy(frame, frame_outputs))
            processed += 1
            if processed % 500 == 0:
                print(f"REPLAY_PROGRESS={name} frames={processed}", flush=True)
        save_mot(outdir / f"{sequence}.txt", rows)
        print(f"REPLAY_SEQUENCE_COMPLETE={name}/{sequence} tracks={len(rows)}", flush=True)
    return outdir


def main() -> None:
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    if not CACHE.is_dir():
        raise FileNotFoundError(f"Run cache_detector_valhalf.py first: {CACHE}")
    sequences = [item["sequence"] for item in json.loads((CACHE / "cache_manifest.json").read_text(encoding="utf-8"))["sequences"]]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    variants = [
        ("KF_IoU", "kf", "iou", False),
        ("UKF_IoU", "ukf", "iou", False),
        ("KF_FishIoU", "kf", "fishiou", False),
        ("UKF_FishIoU", "ukf", "fishiou", False),
        ("UKF_FishIoU_ReID", "ukf", "fishiou", True),
    ]
    for name, motion, association, reid in variants:
        run_sut(name, sequences, motion, association, reid)
    run_external("ByteTrack", sequences, "bytetrack")
    run_external("OCSort", sequences, "ocsort")

    summary = {
        "status": "PREDICTIONS_COMPLETE",
        "dataset": "MFT25 author-provided train val_half",
        "frames": json.loads((CACHE / "cache_manifest.json").read_text(encoding="utf-8"))["frames"],
        "detector_cache": str(CACHE),
        "prediction_root": str(OUTPUT_ROOT),
        "variants": [name for name, *_ in variants] + ["ByteTrack", "OCSort"],
        "claim_scope": "local controlled validation ablation only; not a paper test-set reproduction",
    }
    (OUTPUT_ROOT / "replay_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"REPLAY_COMPLETE={OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
