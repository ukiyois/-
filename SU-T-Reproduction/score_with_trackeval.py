"""Score author-format SU-T results against the MFT25 val_half labels."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trackeval


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


parser = argparse.ArgumentParser()
parser.add_argument("--record-root", type=Path, required=True)
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--tracker-name", required=True)
args = parser.parse_args()

record_root = args.record_root.resolve()
prediction_dir = args.predictions.resolve()
manifest = json.loads((record_root / "data" / "dataset_manifest.json").read_text(encoding="utf-8"))
seq_lengths = {item["sequence"]: item["val_half_frames"] for item in manifest["sequences"]}

trackers_root = record_root / "runs" / "trackeval_input"
tracker_data = trackers_root / args.tracker_name / "data"
tracker_data.mkdir(parents=True, exist_ok=True)
for seq in seq_lengths:
    source = prediction_dir / f"{seq}.txt"
    if not source.is_file():
        raise FileNotFoundError(f"Missing tracker result: {source}")
    rows = [line.strip() for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    for line_number, line in enumerate(rows, start=1):
        fields = line.split(",")
        if len(fields) != 10:
            raise ValueError(f"Expected author MOT output with 10 fields: {source}:{line_number}")
        frame = int(float(fields[0]))
        if not 1 <= frame <= seq_lengths[seq]:
            raise ValueError(f"Frame {frame} is outside {seq} val_half length {seq_lengths[seq]}")
    shutil.copy2(source, tracker_data / source.name)

dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
dataset_config.update(
    {
        "GT_FOLDER": str(record_root / "data" / "ground_truth"),
        "TRACKERS_FOLDER": str(trackers_root),
        "OUTPUT_FOLDER": str(record_root / "metrics" / "trackeval"),
        "TRACKERS_TO_EVAL": [args.tracker_name],
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "MFT25",
        "SPLIT_TO_EVAL": "val_half",
        "DO_PREPROC": False,
        "SEQ_INFO": seq_lengths,
        "SKIP_SPLIT_FOL": True,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "PRINT_CONFIG": False,
        "PLOT_CURVES": False,
    }
)
eval_config = trackeval.Evaluator.get_default_eval_config()
eval_config.update(
    {
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "PRINT_ONLY_COMBINED": False,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
        "LOG_ON_ERROR": str(record_root / "logs" / "trackeval_error.log"),
    }
)

dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
results, messages = trackeval.Evaluator(eval_config).evaluate([dataset], metrics)
dataset_name = dataset.get_name()
result = results[dataset_name][args.tracker_name]
if messages[dataset_name][args.tracker_name] != "Success" or result is None:
    raise RuntimeError(f"TrackEval failed: {messages[dataset_name][args.tracker_name]}")

summary = {
    "dataset": "MFT25 author-provided train val_half",
    "tracker": args.tracker_name,
    "sequences": list(seq_lengths),
    "frames": sum(seq_lengths.values()),
    "trackeval_version": __import__("importlib.metadata", fromlist=["version"]).version("trackeval"),
    "combined": jsonable(result["COMBINED_SEQ"]["pedestrian"]),
    "per_sequence": {
        seq: jsonable(result[seq]["pedestrian"])
        for seq in seq_lengths
    },
}
output = record_root / "metrics" / "trackeval_summary.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"summary={output}")
print(f"sequences={len(seq_lengths)} frames={summary['frames']}")
