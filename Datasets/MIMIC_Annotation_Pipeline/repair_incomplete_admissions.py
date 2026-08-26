#!/usr/bin/env python3
"""
Repair Incomplete Admissions in MIMIC Splits

Problem:
    Stage 0 (mimic_data_extraction_restructured.py) exports {hadm_id}-medication.csv
    even when the admission has zero prescription rows (headers-only file).
    Stage 1 (preprocess_split_mimic.py) skips empty medication tables, so those
    admissions end up with only a diagnosis example in the v2 JSON — no medication
    example. generate_prompts.py then flags them as "[WARNING] Incomplete admission".

Fix strategy:
    1. Scan mimic_split/{train,val,test} for admissions where {hadm_id}-medication.csv
       has 0 data rows.
    2. (--fix) Remove those admission folders from mimic_split/.
    3. (--fix) Remove patients left with 0 admissions from mimic_split/ and from
       split_manifest.json patient ID lists.
    4. (--fix) Filter mimic_data/{split}_row_level_v2.json to remove all examples
       (both diagnosis and medication) that belong to dropped admissions.
    5. Update split_manifest.json statistics.

Usage:
    # Dry-run: report only
    python repair_incomplete_admissions.py

    # Apply fix
    python repair_incomplete_admissions.py --fix

    # Custom paths
    python repair_incomplete_admissions.py --split_dir mimic_split --data_dir mimic_data --manifest split_manifest.json --fix
"""

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple


# --------------------------------------------------------------------------
# Scanning helpers
# --------------------------------------------------------------------------

def count_data_rows(csv_path: Path) -> int:
    """Return number of data rows (excluding header) in a CSV file."""
    if not csv_path.exists():
        return -1  # missing entirely
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # rows[0] is the header (if present); rest are data rows
    return max(len(rows) - 1, 0)


def scan_split(split_dir: Path, split_name: str) -> Dict[str, List[str]]:
    """
    Scan a split directory and return a dict mapping subject_id ->
    list of hadm_ids that have empty medication CSVs.
    """
    incomplete: Dict[str, List[str]] = {}

    if not split_dir.exists():
        print(f"  [SKIP] {split_dir} does not exist")
        return incomplete

    for patient_folder in sorted(split_dir.iterdir()):
        if not patient_folder.is_dir():
            continue
        subject_id = patient_folder.name

        for hadm_folder in sorted(patient_folder.iterdir()):
            if not hadm_folder.is_dir():
                continue
            hadm_id = hadm_folder.name

            med_path = hadm_folder / f"{hadm_id}-medication.csv"
            n_rows = count_data_rows(med_path)

            if n_rows <= 0:
                incomplete.setdefault(subject_id, []).append(hadm_id)

    return incomplete


def all_admissions_for_patient(patient_folder: Path) -> List[str]:
    """Return all hadm_id subdirectory names under a patient folder."""
    return [d.name for d in patient_folder.iterdir() if d.is_dir()]


# --------------------------------------------------------------------------
# v2 JSON filtering
# --------------------------------------------------------------------------

def filter_v2_json(
    v2_path: Path,
    drop_keys: Set[str],  # set of "{subject_id}-{hadm_id}" keys to drop
) -> Tuple[int, int]:
    """
    Load v2 JSON, remove all examples whose anchor_metadata starts with a
    dropped admission key, and write back in-place.

    Returns (original_count, removed_count).
    """
    if not v2_path.exists():
        return 0, 0

    with open(v2_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_count = len(data)
    kept = []
    for ex in data:
        meta = ex.get("anchor_metadata", "")
        # anchor_metadata format: "{subject_id}-{hadm_id}-{table_type}"
        parts = meta.split("-")
        if len(parts) >= 2:
            adm_key = f"{parts[0]}-{parts[1]}"
        else:
            adm_key = meta
        if adm_key not in drop_keys:
            kept.append(ex)

    removed_count = original_count - len(kept)

    if removed_count > 0:
        with open(v2_path, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)

    return original_count, removed_count


# --------------------------------------------------------------------------
# Manifest update
# --------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest_path: Path, data: dict) -> None:
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find and optionally remove incomplete admissions from MIMIC splits."
    )
    parser.add_argument("--split_dir", default="mimic_split",
                        help="Root split directory (default: mimic_split)")
    parser.add_argument("--data_dir", default="mimic_data",
                        help="Directory containing *_row_level_v2.json files (default: mimic_data)")
    parser.add_argument("--manifest", default="split_manifest.json",
                        help="Path to split_manifest.json (default: split_manifest.json)")
    parser.add_argument("--fix", action="store_true",
                        help="Actually remove incomplete admissions and update derived files")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    data_dir = Path(args.data_dir)
    manifest_path = Path(args.manifest)

    split_names = ["train", "val", "test"]

    # ------------------------------------------------------------------
    # Step 1: Scan all splits
    # ------------------------------------------------------------------
    print("=" * 60)
    print("SCANNING FOR INCOMPLETE ADMISSIONS")
    print("=" * 60)

    # incomplete[split_name] = {subject_id: [hadm_id, ...]}
    incomplete: Dict[str, Dict[str, List[str]]] = {}
    total_incomplete_admissions = 0
    total_incomplete_patients = 0

    for split_name in split_names:
        split_path = split_dir / split_name
        result = scan_split(split_path, split_name)
        incomplete[split_name] = result
        n_admissions = sum(len(v) for v in result.values())
        n_patients = len(result)
        total_incomplete_admissions += n_admissions
        print(f"\n  [{split_name.upper()}] {n_patients} patients / {n_admissions} admissions have empty medication.csv")
        if result:
            for sid, hadm_ids in sorted(result.items()):
                print(f"    Patient {sid}: admissions {hadm_ids}")

    print(f"\nTOTAL: {total_incomplete_admissions} incomplete admissions across all splits")

    if total_incomplete_admissions == 0:
        print("\nNo incomplete admissions found. Nothing to do.")
        return

    if not args.fix:
        print("\n[DRY RUN] Pass --fix to apply the following changes:")
        print("  1. Remove incomplete admission folders from mimic_split/")
        print("  2. Remove empty patient folders from mimic_split/")
        print("  3. Filter *_row_level_v2.json to drop those admissions' examples")
        print("  4. Update split_manifest.json patient IDs and statistics")
        return

    # ------------------------------------------------------------------
    # Step 2: Remove admission folders
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("REMOVING INCOMPLETE ADMISSION FOLDERS")
    print("=" * 60)

    removed_patients_by_split: Dict[str, Set[str]] = {s: set() for s in split_names}

    for split_name in split_names:
        split_path = split_dir / split_name
        for subject_id, hadm_ids in incomplete[split_name].items():
            patient_folder = split_path / subject_id
            for hadm_id in hadm_ids:
                hadm_folder = patient_folder / hadm_id
                if hadm_folder.exists():
                    shutil.rmtree(hadm_folder)
                    print(f"  Removed {split_name}/{subject_id}/{hadm_id}/")

            # Remove patient folder if now empty
            remaining = all_admissions_for_patient(patient_folder)
            if not remaining and patient_folder.exists():
                shutil.rmtree(patient_folder)
                removed_patients_by_split[split_name].add(subject_id)
                print(f"  Removed empty patient folder {split_name}/{subject_id}/")

    # ------------------------------------------------------------------
    # Step 3: Filter v2 JSON files
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FILTERING V2 JSON FILES")
    print("=" * 60)

    for split_name in split_names:
        v2_path = data_dir / f"{split_name}_row_level_v2.json"
        # Build set of "subject_id-hadm_id" keys to drop
        drop_keys: Set[str] = set()
        for subject_id, hadm_ids in incomplete[split_name].items():
            for hadm_id in hadm_ids:
                drop_keys.add(f"{subject_id}-{hadm_id}")

        if not drop_keys:
            print(f"  [{split_name}] No examples to remove.")
            continue

        orig, removed = filter_v2_json(v2_path, drop_keys)
        print(f"  [{split_name}] {v2_path}: {orig} → {orig - removed} examples ({removed} removed)")

    # ------------------------------------------------------------------
    # Step 4: Update split_manifest.json
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("UPDATING SPLIT_MANIFEST.JSON")
    print("=" * 60)

    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        splits_block = manifest.get("splits", {})

        for split_name in split_names:
            key = f"{split_name}_patient_ids"
            if key in splits_block and removed_patients_by_split[split_name]:
                old_ids = splits_block[key]
                new_ids = [pid for pid in old_ids
                           if pid not in removed_patients_by_split[split_name]]
                removed_count = len(old_ids) - len(new_ids)
                splits_block[key] = new_ids
                print(f"  [{split_name}] Removed {removed_count} patients from manifest "
                      f"({len(old_ids)} → {len(new_ids)})")

        # Recompute statistics from actual folder counts
        stats = manifest.get("statistics", {})
        for split_name in split_names:
            split_path = split_dir / split_name
            if split_path.exists():
                n_patients = sum(1 for p in split_path.iterdir() if p.is_dir())
                n_admissions = sum(
                    1 for p in split_path.iterdir() if p.is_dir()
                    for h in p.iterdir() if h.is_dir()
                )
                stats[f"{split_name}_patients"] = n_patients
                stats[f"{split_name}_examples"] = n_admissions * 2  # diagnosis + medication per admission
                print(f"  [{split_name}] {n_patients} patients, {n_admissions} admissions remaining")

        stats["total_patients"] = (
            stats.get("train_patients", 0)
            + stats.get("val_patients", 0)
            + stats.get("test_patients", 0)
        )
        manifest["statistics"] = stats
        manifest["last_modified"] = datetime.utcnow().isoformat() + "Z"

        mods = manifest.get("modifications", [])
        mods.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": "repair_incomplete_admissions",
            "detail": (
                f"Removed {total_incomplete_admissions} admissions with empty medication.csv "
                f"across {sum(len(v) for v in removed_patients_by_split.values())} now-empty patients"
            )
        })
        manifest["modifications"] = mods

        save_manifest(manifest_path, manifest)
        print(f"\n  split_manifest.json updated.")
    else:
        print(f"  [SKIP] {manifest_path} not found — manifest not updated.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    total_removed_patients = sum(len(v) for v in removed_patients_by_split.values())
    print(f"  Removed {total_incomplete_admissions} incomplete admission folders")
    print(f"  Removed {total_removed_patients} now-empty patient folders")
    print(f"\n  Next step: re-run preprocess_split_mimic.py to regenerate clean v2 JSON files")
    print(f"  (or the v2 files were already filtered in-place by this script)")


if __name__ == "__main__":
    main()
