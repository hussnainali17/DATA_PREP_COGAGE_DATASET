"""Prepare JSON-selected activities for every subject without reusing class names."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from . import prepare_dataset as pipeline
except ImportError:
    import prepare_dataset as pipeline


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "Sample"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "processed_datasets_from_activity_config"
ACTIVITY_SELECTION_FILE = SCRIPT_DIR / "subject_activity_selection.json"
DEFAULT_ACTIVITIES_PER_SUBJECT = 4
SUBJECT_FOLDER_PATTERN = re.compile(r"^(S\d+)_(.+)_([12])_extracted$")


def discover_subject_recordings(data_root: Path) -> dict[str, dict[str, dict[str, Path]]]:
    """Discover canonical subject/activity folders and preserve folder-based splits."""
    subjects: dict[str, dict[str, dict[str, Path]]] = {}
    for folder in sorted(data_root.iterdir(), key=lambda item: item.name.casefold()):
        if not folder.is_dir():
            continue
        match = SUBJECT_FOLDER_PATTERN.fullmatch(folder.name)
        if match is None:
            continue
        subject, activity, recording = match.groups()
        split = "train" if recording == "1" else "test"
        subjects.setdefault(subject, {}).setdefault(activity, {})[split] = folder
    return subjects


def build_inventory(
    subjects: dict[str, dict[str, dict[str, Path]]],
    selected_activities: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows: list[dict[str, Any]] = []
    paired_activities: dict[str, list[str]] = {}
    for subject in sorted(subjects):
        paired_activities[subject] = []
        for activity in selected_activities:
            recordings = subjects[subject].get(activity, {})
            train_folder = recordings.get("train")
            test_folder = recordings.get("test")
            if train_folder and test_folder:
                status = "paired"
                paired_activities[subject].append(activity)
            elif train_folder:
                status = "missing_test"
            elif test_folder:
                status = "missing_train"
            else:
                status = "missing_both"
            rows.append(
                {
                    "subject": subject,
                    "activity": activity,
                    "train_folder": train_folder.name if train_folder else "",
                    "test_folder": test_folder.name if test_folder else "",
                    "status": status,
                }
            )
    return pd.DataFrame(rows), paired_activities


def load_activity_selection(
    selection_file: Path,
    subjects: dict[str, dict[str, dict[str, Path]]],
    expected_count: int,
) -> dict[str, list[str]]:
    """Load exact per-subject choices, canonicalize case, and require training folders."""
    config = json.loads(selection_file.read_text(encoding="utf-8"))
    if config.get("activities_per_subject") != expected_count:
        raise ValueError(
            f"{selection_file.name} must specify activities_per_subject={expected_count}."
        )
    requested = config.get("activities_by_subject")
    if not isinstance(requested, dict):
        raise ValueError("Activity config must contain an activities_by_subject object.")

    missing_subjects = sorted(set(subjects) - set(requested))
    unknown_subjects = sorted(set(requested) - set(subjects))
    if missing_subjects or unknown_subjects:
        raise ValueError(
            f"Activity config subject mismatch; missing={missing_subjects}, unknown={unknown_subjects}."
        )

    normalized_owner: dict[str, str] = {}
    selection: dict[str, list[str]] = {}
    for subject in sorted(subjects):
        choices = requested[subject]
        if not isinstance(choices, list) or len(choices) != expected_count:
            raise ValueError(f"{subject} must list exactly {expected_count} activity names.")
        if len({str(name).casefold() for name in choices}) != expected_count:
            raise ValueError(f"{subject} has duplicate activity names (ignoring case).")

        canonical_choices: list[str] = []
        for choice in choices:
            matches = [name for name in subjects[subject] if name.casefold() == str(choice).casefold()]
            if len(matches) != 1:
                raise ValueError(
                    f"{subject} activity {choice!r} matched {len(matches)} folders; "
                    "use the exact activity represented in the folder inventory."
                )
            activity = matches[0]
            if "train" not in subjects[subject][activity]:
                raise ValueError(f"{subject} activity {activity} has no _1_extracted training folder.")
            key = activity.casefold()
            previous_subject = normalized_owner.get(key)
            if previous_subject is not None:
                raise ValueError(
                    f"Activity {activity!r} is assigned to both {previous_subject} and {subject}; "
                    "choose different activities for each subject."
                )
            normalized_owner[key] = subject
            canonical_choices.append(activity)
        selection[subject] = canonical_choices
    return selection


def _subject_output_complete(output_dir: Path) -> bool:
    required_files = (
        "X_train.npy",
        "y_train.npy",
        "X_test.npy",
        "y_test.npy",
        "train_metadata.csv",
        "test_metadata.csv",
        "feature_names.json",
        "class_mapping.json",
        "scaler.joblib",
        "dataset_summary.txt",
    )
    return output_dir.is_dir() and all((output_dir / filename).is_file() for filename in required_files)


def _save_subject_dataset(
    output_dir: Path,
    subject: str,
    paired_activities: list[str],
    class_mapping: dict[str, int],
    event_records: list[dict[str, Any]],
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    train_metadata: list[dict[str, Any]],
    x_test_raw: np.ndarray,
    y_test: np.ndarray,
    test_metadata: list[dict[str, Any]],
    feature_names: list[str],
    audit_frame: pd.DataFrame,
    cleaning_stats: Counter[str],
    scaler: StandardScaler,
    missing_test_activities: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    x_train = scaler.transform(x_train_raw.reshape(-1, x_train_raw.shape[-1])).reshape(x_train_raw.shape)
    x_test = (
        scaler.transform(x_test_raw.reshape(-1, x_test_raw.shape[-1])).reshape(x_test_raw.shape)
        if len(x_test_raw)
        else x_test_raw.copy()
    )
    x_train = x_train.astype(np.float32, copy=False)
    x_test = x_test.astype(np.float32, copy=False)

    pipeline.save_dataset(
        output_dir,
        x_train,
        y_train,
        x_test,
        y_test,
        train_metadata,
        test_metadata,
        feature_names,
        class_mapping,
        scaler,
        x_train_raw,
        Counter(y_test.tolist()),
    )
    if not test_metadata:
        pd.DataFrame(
            columns=[
                "subject",
                "activity",
                "train_or_test",
                "activity_instance",
                "event",
                "window_id",
                "label",
                "original_folder",
                "split",
            ]
        ).to_csv(output_dir / "test_metadata.csv", index=False)
    audit_frame.to_csv(output_dir / "raw_file_audit.csv", index=False)

    train_counts = Counter(y_train.tolist())
    test_counts = Counter(y_test.tolist())
    rows = [
        f"CogAge Atomic Behaviour dataset preparation: {subject}",
        "",
        f"Subject: {subject}",
        f"Selected activities: {', '.join(paired_activities)}",
        f"Activities without original test folders: {', '.join(missing_test_activities) or 'none'}",
        f"Subject-specific class mapping: {json.dumps(class_mapping, sort_keys=True)}",
        f"Resample frequency: {pipeline.RESAMPLE_MS}",
        f"Window size: {pipeline.WINDOW_SIZE}",
        f"Feature count: {len(feature_names)}",
        f"X_train shape: {x_train.shape}",
        f"X_test shape: {x_test.shape}",
        "Split policy: original _1_extracted folders feed train; original _2_extracted folders feed test when present.",
        "Scaler: fitted only on this subject's training windows; then applied to that subject's train and test windows.",
        f"Raw sensor files audited: {len(audit_frame)}",
        f"Malformed rows: {int(audit_frame['malformed_rows'].sum()) if not audit_frame.empty else 0}",
        f"Missing raw measurement values: {int(audit_frame['missing_values'].sum()) if not audit_frame.empty else 0}",
        f"Resampled NaN cells interpolated or dropped: {cleaning_stats['resampled_nan_cells_before_interpolation']}",
        f"Rows removed after interpolation/synchronization: {cleaning_stats['rows_removed_after_interpolation_or_sync']}",
        f"Incomplete train windows discarded: {cleaning_stats['discarded_incomplete_windows_train']} ({cleaning_stats['discarded_tail_rows_train']} tail rows)",
        f"Incomplete test windows discarded: {cleaning_stats['discarded_incomplete_windows_test']} ({cleaning_stats['discarded_tail_rows_test']} tail rows)",
        "",
        "Window counts per subject-specific class ID:",
    ]
    if not test_metadata:
        rows.append(
            "TEST UNAVAILABLE: this subject has no _2_extracted source folders. "
            "X_test/y_test are empty; training Events are not reused as test data."
        )
    for activity, label in sorted(class_mapping.items(), key=lambda item: item[1]):
        rows.append(
            f"  Class {label} ({activity}): train={train_counts[label]}, test={test_counts[label]}"
        )

    rows.extend(["", "Original folders used:"])
    used_folders = sorted({record["original_folder"] for record in event_records})
    rows.extend(f"  {folder}" for folder in used_folders)
    rows.extend(
        [
            "",
            "Label IDs 0-3 are local to this subject and correspond to this subject-specific class mapping.",
            "No _1_extracted or _extracted_r training data is relabeled as test. Missing original test data remains unavailable.",
        ]
    )
    (output_dir / "dataset_summary.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def process_subject(
    subject: str,
    subject_recordings: dict[str, dict[str, Path]],
    paired_activities: list[str],
    class_mapping: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    audit_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    cleaning_stats: Counter[str] = Counter()
    missing_test_activities: list[str] = []

    for activity in paired_activities:
        label = class_mapping[activity]
        train_folder = subject_recordings[activity].get("train")
        test_folder = subject_recordings[activity].get("test")
        if train_folder is None:
            raise ValueError(f"Selected activity {activity} for {subject} has no original training folder.")
        for split, folder in (("train", train_folder), ("test", test_folder)):
            if folder is None:
                missing_test_activities.append(activity)
                continue
            print(f"  Processing {folder.name} -> {split.upper()}")
            records = pipeline.process_activity(
                activity,
                folder,
                label,
                split,
                audit_records,
                cleaning_stats,
            )
            for record in records:
                record["subject"] = subject
            event_records.extend(records)

    training_events = [record for record in event_records if record["split"] == "train"]
    feature_names = pipeline._feature_intersection(training_events)
    if not feature_names:
        return {
            "subject": subject,
            "status": "skipped_no_training_features",
            "selected_activities": ",".join(paired_activities),
            "missing_test_activities": ",".join(sorted(set(missing_test_activities))),
            "train_windows": 0,
            "test_windows": 0,
            "output_dir": "",
        }

    x_train_raw, y_train, train_metadata = pipeline.build_train_dataset(
        event_records, feature_names, cleaning_stats
    )
    x_test_raw, y_test, test_metadata = pipeline.build_test_dataset(
        event_records, feature_names, cleaning_stats
    )
    if not len(x_train_raw):
        return {
            "subject": subject,
            "status": "skipped_no_complete_training_windows",
            "selected_activities": ",".join(paired_activities),
            "missing_test_activities": ",".join(sorted(set(missing_test_activities))),
            "train_windows": 0,
            "test_windows": len(x_test_raw),
            "output_dir": "",
        }

    scaler = StandardScaler()
    scaler.fit(x_train_raw.reshape(-1, x_train_raw.shape[-1]))
    _save_subject_dataset(
        output_dir,
        subject,
        paired_activities,
        class_mapping,
        event_records,
        x_train_raw,
        y_train,
        train_metadata,
        x_test_raw,
        y_test,
        test_metadata,
        feature_names,
        pd.DataFrame(audit_records),
        cleaning_stats,
        scaler,
        sorted(set(missing_test_activities)),
    )
    status = "prepared" if not missing_test_activities else "prepared_training_only_no_test_data"
    return {
        "subject": subject,
        "status": status,
        "selected_activities": ",".join(paired_activities),
        "missing_test_activities": ",".join(sorted(set(missing_test_activities))),
        "train_windows": len(x_train_raw),
        "test_windows": len(x_test_raw),
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for a new, selection-hash output folder.",
    )
    parser.add_argument(
        "--activity-config",
        type=Path,
        default=ACTIVITY_SELECTION_FILE,
        help="JSON file listing exactly four unique activities for every subject.",
    )
    parser.add_argument(
        "--activities-per-subject",
        type=int,
        default=DEFAULT_ACTIVITIES_PER_SUBJECT,
        help="Require exactly four activity names in the JSON config for each subject.",
    )
    parser.add_argument(
        "--subjects",
        default="",
        help="Optional comma-separated subject IDs, e.g. S2,S3. Default: discover every subject.",
    )
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_parent = args.output_root.expanduser().resolve()
    selection_file = args.activity_config.expanduser().resolve()
    requested_subjects = [value.strip().upper() for value in args.subjects.split(",") if value.strip()]

    if args.activities_per_subject != 4:
        raise ValueError("This assignment requires exactly four activities per subject.")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")

    discovered = discover_subject_recordings(data_root)
    if requested_subjects:
        unknown = sorted(set(requested_subjects) - set(discovered))
        if unknown:
            raise ValueError(f"Requested subjects not found: {unknown}")

    activity_assignments = load_activity_selection(
        selection_file,
        discovered,
        args.activities_per_subject,
    )
    print(f"Loading subject activity choices from: {selection_file}")
    class_mappings = {
        subject: {activity: index for index, activity in enumerate(activities)}
        for subject, activities in activity_assignments.items()
    }
    selection_json = json.dumps(activity_assignments, sort_keys=True, separators=(",", ":"))
    selection_hash = hashlib.sha256(selection_json.encode("utf-8")).hexdigest()[:12]
    output_root = output_parent / f"selection_{selection_hash}"

    all_activities = sorted(
        {activity for recordings in discovered.values() for activity in recordings},
        key=str.casefold,
    )
    inventory, _ = build_inventory(discovered, all_activities)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "activity_selection_manifest.json"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("activities_by_subject") != activity_assignments:
            raise ValueError(
                "The output folder manifest does not match the selected activity JSON. "
                "Use a different output root or restore the matching activity config."
            )
    else:
        existing_subject_dirs = [
            subject for subject in discovered if (output_root / subject).exists()
        ]
        if existing_subject_dirs:
            raise FileExistsError(
                f"Subject output folders already exist without a selection manifest: {existing_subject_dirs}. "
                "Choose a new --output-root rather than risking a class-map mismatch."
            )
        manifest = {
            "activities_per_subject": args.activities_per_subject,
            "activity_config_file": str(selection_file),
            "selection_hash": selection_hash,
            "selection_policy": "four configured activity names per subject; activity names cannot be reused across subjects ignoring case",
            "activities_by_subject": activity_assignments,
            "class_mapping_by_subject": class_mappings,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if requested_subjects:
        subjects_to_process = requested_subjects
    else:
        subjects_to_process = sorted(discovered)

    selected_rows: list[dict[str, Any]] = []
    for subject in sorted(discovered):
        chosen = activity_assignments.get(subject, [])
        labels = {activity: label for label, activity in enumerate(chosen)}
        for row in inventory.loc[inventory["subject"] == subject].to_dict("records"):
            row["selected_for_subject"] = row["activity"] in labels
            row["subject_label"] = labels.get(row["activity"], "")
            selected_rows.append(row)
    inventory = pd.DataFrame(selected_rows)
    inventory.to_csv(output_root / "subject_activity_inventory.csv", index=False)
    (output_root / "class_mappings_by_subject.json").write_text(
        json.dumps(manifest.get("class_mapping_by_subject", {}), indent=2),
        encoding="utf-8",
    )

    print("Original split rule: _1_extracted -> TRAIN; _2_extracted -> TEST")
    print(f"Activity choices loaded from: {selection_file}")
    print("The JSON config must give each subject exactly four unique names; names cannot repeat across subjects ignoring case.")
    print("A selected activity without _2_extracted stays train-only; `_extracted_r` is never treated as test.")
    print("\nFull folder inventory (chosen rows marked selected_for_subject):")
    print(inventory.to_string(index=False))

    results: list[dict[str, Any]] = []
    for subject in subjects_to_process:
        available_pairs = activity_assignments.get(subject, [])
        if not available_pairs:
            result = {
                "subject": subject,
                "status": "skipped_fewer_than_four_configured_activities",
                "selected_activities": "",
                "train_windows": 0,
                "test_windows": 0,
                "output_dir": "",
            }
            print(
                f"\n{subject}: skipped; fewer than four unique training activities were available "
                "after assigning distinct activities across subjects."
            )
            results.append(result)
            continue

        subject_output = output_root / subject
        if subject_output.exists():
            if _subject_output_complete(subject_output):
                result = {
                    "subject": subject,
                    "status": "skipped_existing_dataset",
                    "selected_activities": ",".join(available_pairs),
                    "missing_test_activities": ",".join(
                        activity for activity in available_pairs
                        if "test" not in discovered[subject].get(activity, {})
                    ),
                    "train_windows": len(np.load(subject_output / "y_train.npy")),
                    "test_windows": len(np.load(subject_output / "y_test.npy")),
                    "output_dir": str(subject_output),
                }
                print(f"\n{subject}: existing complete dataset found; skipping without rewriting it.")
            else:
                result = {
                    "subject": subject,
                    "status": "skipped_existing_incomplete_output",
                    "selected_activities": ",".join(available_pairs),
                    "missing_test_activities": ",".join(
                        activity for activity in available_pairs
                        if "test" not in discovered[subject].get(activity, {})
                    ),
                    "train_windows": 0,
                    "test_windows": 0,
                    "output_dir": str(subject_output),
                }
                print(f"\n{subject}: output folder exists but is incomplete; leaving it untouched.")
            results.append(result)
            continue

        class_mapping = {activity: index for index, activity in enumerate(available_pairs)}
        print(f"\nPreparing {subject}; selected classes and labels: {class_mapping}")
        result = process_subject(
            subject,
            discovered[subject],
            available_pairs,
            class_mapping,
            subject_output,
        )
        results.append(result)
        print(
            f"{subject}: {result['status']}; train windows={result['train_windows']}; "
            f"test windows={result['test_windows']}"
        )

    results_frame = pd.DataFrame(results)
    results_path = output_root / "subject_preparation_summary.csv"
    if results_path.is_file():
        previous = pd.read_csv(results_path).set_index("subject")
        current = results_frame.set_index("subject")
        merged = []
        for subject in sorted(set(previous.index) | set(current.index)):
            if subject not in current.index:
                merged.append(previous.loc[subject].to_dict() | {"subject": subject})
                continue
            new_result = current.loc[subject].to_dict() | {"subject": subject}
            old_result = previous.loc[subject].to_dict() | {"subject": subject} if subject in previous.index else None
            if (
                old_result is not None
                and new_result["status"] == "skipped_existing_dataset"
                and str(old_result.get("status", "")).startswith("prepared")
            ):
                old_result["last_run_action"] = "skipped_existing_dataset"
                merged.append(old_result)
            else:
                merged.append(new_result)
        results_frame = pd.DataFrame(merged)
    results_frame.to_csv(results_path, index=False)
    print(f"\nPer-subject datasets and inventory saved under: {output_root}")
    print("Subject-level summary:")
    print(results_frame.to_string(index=False))


if __name__ == "__main__":
    main()
