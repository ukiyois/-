"""Validate completed author-format predictions and export compact run summaries."""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RUN_ID = "SU-T-ReID-ValHalf-A03"
RECORD = ROOT / "data" / "run_metadata.json"
MANIFEST = ROOT / "data" / "dataset_manifest.json"
TRACK_SUMMARY = ROOT / "metrics" / "trackeval_summary.json"
RUN_LOG = ROOT / "logs" / "sut_reid_valhalf_attempt03_console.log"
RUN_ROOT = ROOT / "runs" / RUN_ID


def percent(value: float) -> float:
    return round(float(value) * 100.0, 4)


metadata = json.loads(RECORD.read_text(encoding="utf-8"))
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
trackeval_summary = json.loads(TRACK_SUMMARY.read_text(encoding="utf-8"))
run_log_bytes = RUN_LOG.read_bytes()
if run_log_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
    run_log = run_log_bytes.decode("utf-16")
else:
    run_log = run_log_bytes.decode("utf-8-sig")

assert "RUN_STATUS=COMPLETED" in run_log
assert "Completed" in run_log
assert trackeval_summary["tracker"] == RUN_ID

sequence_lengths = {
    item["sequence"]: int(item["val_half_frames"])
    for item in manifest["sequences"]
}
result_dirs = [
    path for path in RUN_ROOT.iterdir()
    if path.is_dir() and any(path.glob("*.txt"))
]
if len(result_dirs) != 1:
    raise RuntimeError(f"Expected one prediction directory, found {len(result_dirs)}")
prediction_dir = result_dirs[0]

prediction_files = {path.stem: path for path in prediction_dir.glob("*.txt")}
if set(prediction_files) != set(sequence_lengths):
    raise RuntimeError(
        f"Prediction sequence mismatch; missing={sorted(set(sequence_lengths) - set(prediction_files))}, "
        f"extra={sorted(set(prediction_files) - set(sequence_lengths))}"
    )

per_sequence = trackeval_summary["per_sequence"]
if set(per_sequence) != set(sequence_lengths):
    raise RuntimeError("TrackEval sequence list does not match the local val_half manifest")

rows: list[dict[str, object]] = []
for sequence, length in sequence_lengths.items():
    path = prediction_files[sequence]
    detections = 0
    seen_frame_ids: set[tuple[int, int]] = set()
    frames: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != 10:
            raise ValueError(f"Expected 10 columns at {path}:{line_number}, got {len(fields)}")
        frame, track_id = int(float(fields[0])), int(float(fields[1]))
        if frame < 1 or frame > length:
            raise ValueError(f"Frame {frame} outside 1..{length} at {path}:{line_number}")
        if (frame, track_id) in seen_frame_ids:
            raise ValueError(f"Duplicate frame/track pair at {path}:{line_number}")
        seen_frame_ids.add((frame, track_id))
        frames.add(frame)
        detections += 1
    if not frames or max(frames) != length:
        raise ValueError(f"{sequence} result does not reach its final val_half frame {length}")

    scores = per_sequence[sequence]
    hota = scores["HOTA"]
    clear = scores["CLEAR"]
    identity = scores["Identity"]
    count = scores["Count"]
    rows.append(
        {
            "sequence": sequence,
            "frames": length,
            "prediction_rows": detections,
            "HOTA_pct": percent(np.mean(hota["HOTA"])),
            "DetA_pct": percent(np.mean(hota["DetA"])),
            "AssA_pct": percent(np.mean(hota["AssA"])),
            "MOTA_pct": percent(clear["MOTA"]),
            "IDF1_pct": percent(identity["IDF1"]),
            "IDSW": int(clear["IDSW"]),
            "GT_detections": int(count["GT_Dets"]),
            "GT_IDs": int(count["GT_IDs"]),
        }
    )

combined = trackeval_summary["combined"]
combined_hota = combined["HOTA"]
combined_clear = combined["CLEAR"]
combined_identity = combined["Identity"]
combined_count = combined["Count"]
summary_row = {
    "sequence": "COMBINED",
    "frames": int(trackeval_summary["frames"]),
    "prediction_rows": int(combined_count["Dets"]),
    "HOTA_pct": percent(np.mean(combined_hota["HOTA"])),
    "DetA_pct": percent(np.mean(combined_hota["DetA"])),
    "AssA_pct": percent(np.mean(combined_hota["AssA"])),
    "MOTA_pct": percent(combined_clear["MOTA"]),
    "IDF1_pct": percent(combined_identity["IDF1"]),
    "IDSW": int(combined_clear["IDSW"]),
    "GT_detections": int(combined_count["GT_Dets"]),
    "GT_IDs": int(combined_count["GT_IDs"]),
}

timing_log = (RUN_ROOT / "val_log.txt").read_text(encoding="utf-8-sig")
timed_lines = []
for line in timing_log.splitlines():
    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})", line)
    if match:
        timed_lines.append((datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f"), line))
if not timed_lines:
    raise RuntimeError("No timestamped lines found in the run log")
start_time = timed_lines[0][0]
completed_lines = [item for item in timed_lines if "Completed" in item[1]]
if not completed_lines:
    raise RuntimeError("Completion timestamp not found in the run log")
end_time = completed_lines[-1][0]
duration_seconds = round((end_time - start_time).total_seconds(), 3)

def find_float(pattern: str) -> float:
    match = re.search(pattern, run_log)
    if not match:
        raise RuntimeError(f"Could not parse run metric with pattern: {pattern}")
    return float(match.group(1))


timing_match = re.search(
    r"Average forward time: ([0-9.]+) ms, Average track time: ([0-9.]+) ms, "
    r"Average inference time: ([0-9.]+) ms",
    run_log,
)
if not timing_match:
    raise RuntimeError("Could not parse average inference timing")

ap_metrics = {
    "AP_50_95": find_float(r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s*all .*?= ([0-9.]+)"),
    "AP50": find_float(r"Average Precision\s+\(AP\) @\[ IoU=0\.50\s+\| area=\s*all .*?= ([0-9.]+)"),
    "AP75": find_float(r"Average Precision\s+\(AP\) @\[ IoU=0\.75\s+\| area=\s*all .*?= ([0-9.]+)"),
}

result = {
    "run_id": RUN_ID,
    "status": "COMPLETED",
    "start_time_local": start_time.isoformat(sep=" "),
    "end_time_local": end_time.isoformat(sep=" "),
    "duration_seconds": duration_seconds,
    "split": "MFT25 author-provided train val_half",
    "sequences": list(sequence_lengths),
    "frames": sum(sequence_lengths.values()),
    "ground_truth_detections": sum(
        sum(1 for _ in (ROOT / "data" / "ground_truth" / sequence / "gt" / "gt.txt").open(encoding="utf-8"))
        for sequence in sequence_lengths
    ),
    "hardware": metadata["gpu"],
    "python_executable": sys.executable,
    "resource_observations": {
        "gpu_memory_used_sampled_mib_range": [3304, 3433],
        "gpu_temperature_sampled_c_range": [73, 86],
        "sampling_note": "Periodic nvidia-smi snapshots during the completed run; not a hardware peak recorder.",
    },
    "python": metadata["python"],
    "packages": metadata["packages"],
    "weights": {
        name: metadata["inputs"][name]
        for name in ("detector_checkpoint", "reid_checkpoint")
    },
    "evaluation_config": metadata["evaluation"],
    "author_entry_metrics": {
        "AP": ap_metrics,
        "average_forward_ms": float(timing_match.group(1)),
        "average_track_ms": float(timing_match.group(2)),
        "average_inference_ms": float(timing_match.group(3)),
    },
    "trackeval": {
        "version": trackeval_summary["trackeval_version"],
        "HOTA_pct": summary_row["HOTA_pct"],
        "DetA_pct": summary_row["DetA_pct"],
        "AssA_pct": summary_row["AssA_pct"],
        "MOTA_pct": summary_row["MOTA_pct"],
        "IDF1_pct": summary_row["IDF1_pct"],
        "IDSW": summary_row["IDSW"],
        "prediction_detections": summary_row["prediction_rows"],
        "ground_truth_detections": summary_row["GT_detections"],
        "ground_truth_ids": summary_row["GT_IDs"],
        "raw_summary": "trackeval_summary.json",
    },
    "per_sequence_csv": "sequence_summary.csv",
    "predictions": str(prediction_dir.relative_to(ROOT)),
    "run_log": "../logs/sut_reid_valhalf_attempt03_console.log",
    "interpretation": (
        "Local val_half verification only; this is the latter half of training videos and "
        "is not directly comparable with the hidden MFT25 test leaderboard."
    ),
}

fields = list(summary_row)
with (ROOT / "metrics" / "sequence_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerow(summary_row)
    writer.writerows(rows)

(ROOT / "metrics" / "run_summary.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"validated_sequences={len(rows)} frames={result['frames']}")
print(f"summary={ROOT / 'metrics' / 'run_summary.json'}")
print(f"sequence_csv={ROOT / 'metrics' / 'sequence_summary.csv'}")
