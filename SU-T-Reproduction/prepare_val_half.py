"""Prepare an auditable COCO view of the MFT25 train-half validation split."""

from __future__ import annotations

import argparse
import configparser
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


def read_mot_rows(raw: bytes) -> dict[int, list[tuple[float, ...]]]:
    rows: dict[int, list[tuple[float, ...]]] = defaultdict(list)
    for line in raw.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        values = tuple(float(value) for value in line.split(","))
        rows[int(values[0])].append(values)
    return rows


def row_signature(rows: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    return sorted(tuple(row[1:9]) for row in rows)


def seqinfo(archive: zipfile.ZipFile, seq: str) -> dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read_string(archive.read(f"{seq}/seqinfo.ini").decode("utf-8-sig"))
    return dict(parser["Sequence"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    record_root = args.record_root.resolve()
    json_dir = dataset_root / "annotations"
    gt_root = record_root / "data" / "ground_truth"
    json_dir.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    annotations: list[dict] = []
    seqmap: list[str] = []
    sequences: list[dict] = []
    next_image_id = 1
    next_annotation_id = 1

    with zipfile.ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        seqs = sorted(
            name[:-len("/seqinfo.ini")]
            for name in names
            if name.endswith("/seqinfo.ini")
        )
        for video_id, seq in enumerate(seqs, start=1):
            info = seqinfo(archive, seq)
            length = int(info["seqlength"])
            width = int(info["imwidth"])
            height = int(info["imheight"])
            extension = info["imext"]

            train_rows = read_mot_rows(archive.read(f"{seq}/gt/gt_train_half.txt"))
            val_rows = read_mot_rows(archive.read(f"{seq}/gt/gt_val_half.txt"))
            full_rows = read_mot_rows(archive.read(f"{seq}/gt/gt.txt"))
            train_end = max(train_rows)
            val_length = length - train_end
            if val_length < 1 or max(val_rows, default=0) > val_length:
                raise ValueError(f"Invalid val_half frame range for {seq}")

            # The author-provided val_half frame numbers are local to the held-out
            # segment. Verify each labelled frame against the full annotation.
            for local_frame, rows in val_rows.items():
                source_frame = train_end + local_frame
                if row_signature(rows) != row_signature(full_rows.get(source_frame, [])):
                    raise ValueError(
                        f"val_half labels do not map to {seq} source frame {source_frame}"
                    )

            for local_frame in range(1, val_length + 1):
                source_frame = train_end + local_frame
                file_name = f"{seq}/img1/{source_frame:06d}{extension}"
                if f"{seq}/img1/{source_frame:06d}{extension}" not in names:
                    raise FileNotFoundError(f"Missing archive image: {file_name}")
                images.append(
                    {
                        "id": next_image_id,
                        "file_name": file_name,
                        "width": width,
                        "height": height,
                        "frame_id": local_frame,
                        "video_id": video_id,
                    }
                )
                for row in val_rows.get(local_frame, []):
                    frame, track_id, x, y, box_w, box_h, mark, category, visibility = row
                    if mark != 1 or category != 1:
                        raise ValueError(f"Unexpected non-fish annotation in {seq}")
                    annotations.append(
                        {
                            "id": next_annotation_id,
                            "image_id": next_image_id,
                            "category_id": 1,
                            "bbox": [x, y, box_w, box_h],
                            "area": box_w * box_h,
                            "iscrowd": 0,
                            "track_id": int(track_id),
                            "visibility": visibility,
                        }
                    )
                    next_annotation_id += 1
                next_image_id += 1

            seq_dir = gt_root / seq
            gt_dir = seq_dir / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)
            val_bytes = archive.read(f"{seq}/gt/gt_val_half.txt")
            (gt_dir / "gt.txt").write_bytes(val_bytes)
            (seq_dir / "seqinfo.ini").write_text(
                "[Sequence]\n"
                f"name={seq}\n"
                "imDir=img1\n"
                f"frameRate={info['framerate']}\n"
                f"seqLength={val_length}\n"
                f"imWidth={width}\n"
                f"imHeight={height}\n"
                f"imExt={extension}\n",
                encoding="utf-8",
            )
            seqmap.append(seq)
            sequences.append(
                {
                    "sequence": seq,
                    "full_frames": length,
                    "train_half_end_frame": train_end,
                    "val_half_frames": val_length,
                    "val_half_labeled_frames": len(val_rows),
                    "val_half_ground_truth_boxes": sum(map(len, val_rows.values())),
                    "verified_against_full_gt": True,
                }
            )

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "fish", "supercategory": "fish"}],
    }
    output_json = json_dir / "val_half.json"
    output_json.write_text(json.dumps(coco, separators=(",", ":")), encoding="utf-8")
    record_json = record_root / "data" / "annotations" / "val_half.json"
    shutil.copy2(output_json, record_json)
    (record_root / "data" / "seqmap.txt").write_text(
        "name\n" + "\n".join(seqmap) + "\n", encoding="utf-8"
    )
    manifest = {
        "source_archive": str(args.archive.resolve()),
        "dataset_root": str(dataset_root),
        "annotation_file": str(output_json),
        "split": "author-provided val_half from MFT25-train.zip",
        "frame_mapping": "val_half local frame f maps to source img1 frame train_half_end_frame + f",
        "num_sequences": len(sequences),
        "num_images": len(images),
        "num_ground_truth_boxes": len(annotations),
        "sequences": sequences,
    }
    (record_root / "data" / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("num_sequences", "num_images", "num_ground_truth_boxes")}, indent=2))
    print(f"annotation_file={output_json}")
    print(f"ground_truth={gt_root}")


if __name__ == "__main__":
    main()
