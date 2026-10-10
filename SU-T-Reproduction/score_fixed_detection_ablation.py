"""Score every fixed-detection ablation through the same TrackEval instance."""

from __future__ import annotations

import csv
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import trackeval


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "runs" / "fixed_detection_A05"
METRICS = ROOT / "metrics" / "fixed_detection_A05"
TRACKERS = [
    "KF_IoU", "UKF_IoU", "KF_FishIoU", "UKF_FishIoU", "UKF_FishIoU_ReID",
    "ByteTrack", "OCSort",
]


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


def compact(raw: dict) -> dict[str, float | int]:
    hota = raw["HOTA"]
    clear = raw["CLEAR"]
    identity = raw["Identity"]
    return {
        "HOTA_pct": round(float(np.mean(hota["HOTA"]) * 100), 4),
        "DetA_pct": round(float(np.mean(hota["DetA"]) * 100), 4),
        "AssA_pct": round(float(np.mean(hota["AssA"]) * 100), 4),
        "MOTA_pct": round(float(clear["MOTA"] * 100), 4),
        "IDF1_pct": round(float(identity["IDF1"] * 100), 4),
        "IDSW": int(clear["IDSW"]),
    }


def main() -> None:
    replay_manifest = json.loads((PREDICTIONS / "replay_manifest.json").read_text(encoding="utf-8"))
    if replay_manifest["status"] != "PREDICTIONS_COMPLETE":
        raise RuntimeError("Prediction replay is not marked complete")
    cache_manifest = json.loads((ROOT / "data" / "detection_cache_A05" / "cache_manifest.json").read_text(encoding="utf-8"))
    seq_lengths = {item["sequence"]: int(item["frames"]) for item in cache_manifest["sequences"]}

    trackers_root = ROOT / "runs" / "trackeval_input_A05"
    trackers_root.mkdir(parents=True, exist_ok=True)
    for tracker in TRACKERS:
        source_dir = PREDICTIONS / tracker
        tracker_data = trackers_root / tracker / "data"
        tracker_data.mkdir(parents=True, exist_ok=True)
        actual = {path.stem for path in source_dir.glob("*.txt")}
        if actual != set(seq_lengths):
            raise RuntimeError(f"{tracker}: sequence mismatch missing={set(seq_lengths)-actual} extra={actual-set(seq_lengths)}")
        for sequence, length in seq_lengths.items():
            source = source_dir / f"{sequence}.txt"
            lines = [line.strip() for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            for line_no, line in enumerate(lines, 1):
                fields = line.split(",")
                if len(fields) != 10:
                    raise ValueError(f"Expected 10 MOT fields: {source}:{line_no}")
                frame = int(float(fields[0]))
                if not 1 <= frame <= length:
                    raise ValueError(f"Frame outside val_half range: {source}:{line_no}")
            shutil.copy2(source, tracker_data / source.name)

    METRICS.mkdir(parents=True, exist_ok=True)
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(ROOT / "data" / "ground_truth"),
        "TRACKERS_FOLDER": str(trackers_root),
        "OUTPUT_FOLDER": str(METRICS / "trackeval"),
        "TRACKERS_TO_EVAL": [],
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
    })
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "PRINT_ONLY_COMBINED": False,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
        "LOG_ON_ERROR": str(METRICS / "trackeval_error.log"),
    })
    summaries = {}
    detailed = {}
    for tracker in TRACKERS:
        print(f"TRACKEVAL_START={tracker}", flush=True)
        dataset_config["TRACKERS_TO_EVAL"] = [tracker]
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
        metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
        results, messages = trackeval.Evaluator(eval_config).evaluate([dataset], metrics)
        dataset_name = dataset.get_name()
        result = results[dataset_name][tracker]
        if messages[dataset_name][tracker] != "Success" or result is None:
            raise RuntimeError(f"TrackEval failed for {tracker}: {messages[dataset_name][tracker]}")
        combined = result["COMBINED_SEQ"]["pedestrian"]
        per_sequence = {seq: result[seq]["pedestrian"] for seq in seq_lengths}
        summaries[tracker] = compact(combined)
        detailed[tracker] = {
            "combined": jsonable(combined),
            "per_sequence": {seq: jsonable(value) for seq, value in per_sequence.items()},
            "per_sequence_compact": {seq: compact(value) for seq, value in per_sequence.items()},
            "macro_sequence_average": {
                key: round(float(np.mean([compact(value)[key] for value in per_sequence.values()])), 4)
                for key in ("HOTA_pct", "DetA_pct", "AssA_pct", "MOTA_pct", "IDF1_pct")
            },
            "macro_IDSW_per_sequence": round(float(np.mean([compact(value)["IDSW"] for value in per_sequence.values()])), 4),
        }
        del results, messages, result, dataset, metrics
        gc.collect()
        print(f"TRACKEVAL_COMPLETE={tracker}", flush=True)

    pooled_keys = ("HOTA_pct", "DetA_pct", "AssA_pct", "MOTA_pct", "IDF1_pct", "IDSW")
    macro_keys = ("HOTA_pct", "DetA_pct", "AssA_pct", "MOTA_pct", "IDF1_pct")

    def delta(reference: str, comparison: str) -> dict:
        pooled = {key: round(summaries[comparison][key] - summaries[reference][key], 4) for key in pooled_keys}
        macro = {
            key: round(
                detailed[comparison]["macro_sequence_average"][key]
                - detailed[reference]["macro_sequence_average"][key], 4
            )
            for key in macro_keys
        }
        macro["IDSW_per_sequence"] = round(
            detailed[comparison]["macro_IDSW_per_sequence"]
            - detailed[reference]["macro_IDSW_per_sequence"], 4
        )
        return {"pooled": pooled, "macro_sequence_average": macro}

    deltas = {
        "UKF_effect_at_IoU": delta("KF_IoU", "UKF_IoU"),
        "FishIoU_effect_with_KF": delta("KF_IoU", "KF_FishIoU"),
        "FishIoU_effect_with_UKF": delta("UKF_IoU", "UKF_FishIoU"),
        "UKF_effect_with_IoU": delta("KF_IoU", "UKF_IoU"),
        "UKF_effect_with_FishIoU": delta("KF_FishIoU", "UKF_FishIoU"),
        "UKF_plus_FishIoU_vs_KF_IoU": delta("KF_IoU", "UKF_FishIoU"),
        "UKF_FishIoU_interaction": {
            "pooled": {
                key: round(
                    summaries["UKF_FishIoU"][key] - summaries["UKF_IoU"][key]
                    - summaries["KF_FishIoU"][key] + summaries["KF_IoU"][key], 4
                )
                for key in pooled_keys
            },
            "macro_sequence_average": {
                key: round(
                    detailed["UKF_FishIoU"]["macro_sequence_average"][key]
                    - detailed["UKF_IoU"]["macro_sequence_average"][key]
                    - detailed["KF_FishIoU"]["macro_sequence_average"][key]
                    + detailed["KF_IoU"]["macro_sequence_average"][key], 4
                )
                for key in macro_keys
            },
        },
        "ReID_effect_on_UKF_FishIoU": delta("UKF_FishIoU", "UKF_FishIoU_ReID"),
    }

    payload = {
        "dataset": "MFT25 author-provided train val_half",
        "frames": sum(seq_lengths.values()),
        "sequences": list(seq_lengths),
        "detection_cache": str(ROOT / "data" / "detection_cache_A05"),
        "detector_cache_sha256": {item["sequence"]: item["cache_sha256"] for item in cache_manifest["sequences"]},
        "trackeval_version": __import__("importlib.metadata", fromlist=["version"]).version("trackeval"),
        "interpretation": "same cached post-NMS detector boxes and same TrackEval code; local val_half ablation, not paper test-set reproduction",
        "pooled_summary": summaries,
        "per_sequence_and_raw_metrics": detailed,
        "deltas": deltas,
    }
    (METRICS / "ablation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (METRICS / "ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        columns = ["tracker", "HOTA_pct", "DetA_pct", "AssA_pct", "MOTA_pct", "IDF1_pct", "IDSW",
                   "macro_HOTA_pct", "macro_DetA_pct", "macro_AssA_pct", "macro_MOTA_pct", "macro_IDF1_pct",
                   "macro_IDSW_per_sequence"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for tracker in TRACKERS:
            writer.writerow({
                "tracker": tracker,
                **summaries[tracker],
                **{f"macro_{key}": value for key, value in detailed[tracker]["macro_sequence_average"].items()},
                "macro_IDSW_per_sequence": detailed[tracker]["macro_IDSW_per_sequence"],
            })
    print(f"SCORED={METRICS / 'ablation_results.json'}")
    print(json.dumps({"pooled_summary": summaries, "deltas": deltas}, indent=2))


if __name__ == "__main__":
    main()
