"""
MIMIC Folder Splitting Script

This script separates patient folders from a source MIMIC directory into 
train/val/test splits based on an existing split_manifest.json file.

This is useful when you have already processed the data and have a split_manifest.json
but need to physically separate the raw patient folders for different purposes
(e.g., annotation on test set only).

Usage:
    # Copy folders to split directories (preserves original)
    python split_mimic_folders.py --source_dir ./mimic_6617 --output_dir ./mimic_split --manifest ./split_manifest.json
    
    # Move folders instead of copying (faster, but modifies source)
    python split_mimic_folders.py --source_dir ./mimic_6617 --output_dir ./mimic_split --manifest ./split_manifest.json --move
    
    # Dry run to see what would happen
    python split_mimic_folders.py --source_dir ./mimic_6617 --output_dir ./mimic_split --manifest ./split_manifest.json --dry_run

Output Structure:
    mimic_split/
        train/
            <patient_id>/
                <hadm_id>/
                    <hadm_id>-diagnosis.csv
                    <hadm_id>-medication.csv
                    <hadm_id>-notes.txt
        val/
            <patient_id>/
                ...
        test/
            <patient_id>/
                ...
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set
from tqdm import tqdm
from datetime import datetime


def load_manifest(manifest_path: str) -> Dict:
    """Load and validate the split manifest file."""
    manifest_path = Path(manifest_path)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    # Validate required fields
    required_fields = ['splits']
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"Manifest missing required field: {field}")
    
    splits = manifest['splits']
    for split_name in ['train_patient_ids', 'val_patient_ids', 'test_patient_ids']:
        if split_name not in splits:
            raise ValueError(f"Manifest missing required split: {split_name}")
    
    return manifest


def get_available_patients(source_dir: Path) -> Set[str]:
    """Get set of patient IDs available in the source directory."""
    return {f.name for f in source_dir.iterdir() if f.is_dir()}


def validate_splits(manifest: Dict, available_patients: Set[str]) -> Dict[str, List[str]]:
    """
    Validate and filter split patient IDs against available patients.
    Returns a dictionary mapping split names to lists of valid patient IDs.
    """
    splits = manifest['splits']
    
    result = {}
    missing_patients = {}
    
    for split_name, manifest_key in [
        ('train', 'train_patient_ids'),
        ('val', 'val_patient_ids'),
        ('test', 'test_patient_ids')
    ]:
        patient_ids = splits[manifest_key]
        valid_ids = [pid for pid in patient_ids if pid in available_patients]
        missing_ids = [pid for pid in patient_ids if pid not in available_patients]
        
        result[split_name] = valid_ids
        
        if missing_ids:
            missing_patients[split_name] = missing_ids
    
    # Report missing patients
    if missing_patients:
        print("\nWarning: Some patients from manifest not found in source directory:")
        for split_name, missing in missing_patients.items():
            print(f"  {split_name}: {len(missing)} patients missing")
            if len(missing) <= 10:
                print(f"    IDs: {missing}")
            else:
                print(f"    First 10 IDs: {missing[:10]}...")
        print()
    
    return result


def split_folders(
    source_dir: str,
    output_dir: str,
    manifest_path: str,
    move: bool = False,
    dry_run: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Split patient folders from source directory into train/val/test directories.
    
    Args:
        source_dir: Path to source MIMIC directory with patient folders
        output_dir: Path to output directory for split folders
        manifest_path: Path to split_manifest.json file
        move: If True, move folders instead of copying (faster but modifies source)
        dry_run: If True, only print what would be done without actually doing it
        verbose: If True, print progress information
    
    Returns:
        Dictionary with statistics about the operation
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Load manifest
    if verbose:
        print("=" * 60)
        print("MIMIC Folder Splitting")
        print("=" * 60)
        print(f"Source: {source_dir}")
        print(f"Output: {output_dir}")
        print(f"Manifest: {manifest_path}")
        print(f"Mode: {'MOVE' if move else 'COPY'}")
        print(f"Dry run: {dry_run}")
        print("=" * 60)
    
    manifest = load_manifest(manifest_path)
    
    if verbose:
        print(f"\nManifest loaded (generated: {manifest.get('generated_at', 'unknown')})")
        stats = manifest.get('statistics', {})
        print(f"  Expected patients - Train: {stats.get('train_patients', '?')}, "
              f"Val: {stats.get('val_patients', '?')}, Test: {stats.get('test_patients', '?')}")
    
    # Get available patients
    available_patients = get_available_patients(source_dir)
    if verbose:
        print(f"\nFound {len(available_patients)} patient folders in source directory")
    
    # Validate splits
    split_patient_ids = validate_splits(manifest, available_patients)
    
    if verbose:
        print("Valid patients per split:")
        for split_name, patient_ids in split_patient_ids.items():
            print(f"  {split_name}: {len(patient_ids)} patients")
    
    # Check for overlap
    all_assigned = set()
    for split_name, patient_ids in split_patient_ids.items():
        overlap = all_assigned & set(patient_ids)
        if overlap:
            print(f"\nWarning: {len(overlap)} patients appear in multiple splits!")
        all_assigned.update(patient_ids)
    
    # Check for unassigned patients
    unassigned = available_patients - all_assigned
    if unassigned and verbose:
        print(f"\nNote: {len(unassigned)} patients in source not assigned to any split")
    
    # Create output directories
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ['train', 'val', 'test']:
            (output_dir / split_name).mkdir(exist_ok=True)
    
    # Process each split
    operation_stats = {
        'total_processed': 0,
        'total_skipped': 0,
        'by_split': {}
    }
    
    operation = shutil.move if move else shutil.copytree
    operation_name = "Moving" if move else "Copying"
    
    for split_name, patient_ids in split_patient_ids.items():
        if verbose:
            print(f"\n{operation_name} {split_name} patients...")
        
        split_output = output_dir / split_name
        processed = 0
        skipped = 0
        
        for patient_id in tqdm(patient_ids, desc=f"  {split_name}", disable=not verbose):
            src_path = source_dir / patient_id
            dst_path = split_output / patient_id
            
            if not src_path.exists():
                skipped += 1
                continue
            
            if dst_path.exists():
                if verbose:
                    tqdm.write(f"    Skipping {patient_id} (already exists in destination)")
                skipped += 1
                continue
            
            if dry_run:
                if verbose:
                    tqdm.write(f"    Would {operation_name.lower()[:-3]} {patient_id}")
            else:
                try:
                    if move:
                        shutil.move(str(src_path), str(dst_path))
                    else:
                        shutil.copytree(str(src_path), str(dst_path))
                    processed += 1
                except Exception as e:
                    tqdm.write(f"    Error processing {patient_id}: {e}")
                    skipped += 1
                    continue
            
            processed += 1
        
        operation_stats['by_split'][split_name] = {
            'processed': processed,
            'skipped': skipped,
            'total': len(patient_ids)
        }
        operation_stats['total_processed'] += processed
        operation_stats['total_skipped'] += skipped
    
    # Create a summary file in output directory
    if not dry_run:
        summary = {
            'created_at': datetime.now().isoformat(),
            'source_dir': str(source_dir),
            'manifest_path': str(manifest_path),
            'operation': 'move' if move else 'copy',
            'statistics': operation_stats,
            'manifest_statistics': manifest.get('statistics', {})
        }
        
        summary_path = output_dir / 'split_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        if verbose:
            print(f"\nSummary written to: {summary_path}")
    
    # Print final summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for split_name, stats in operation_stats['by_split'].items():
            print(f"  {split_name}: {stats['processed']}/{stats['total']} patients "
                  f"({'moved' if move else 'copied'}), {stats['skipped']} skipped")
        print(f"\nTotal: {operation_stats['total_processed']} patients processed, "
              f"{operation_stats['total_skipped']} skipped")
        
        if dry_run:
            print("\n[DRY RUN - No files were actually modified]")
    
    return operation_stats


def main():
    parser = argparse.ArgumentParser(
        description="Split MIMIC patient folders into train/val/test directories based on manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Copy folders to split directories (preserves original)
    python split_mimic_folders.py --source_dir ./mimic_6617 --output_dir ./mimic_split --manifest ./split_manifest.json
    
    # Move folders instead of copying (faster, but modifies source)
    python split_mimic_folders.py --source_dir ./mimic_6617 --output_dir ./mimic_split --manifest ./split_manifest.json --move
    
    # Dry run to see what would happen
    python split_mimic_folders.py --source_dir ./mimic_6617 --output_dir ./mimic_split --manifest ./split_manifest.json --dry_run
"""
    )
    
    parser.add_argument(
        '--source_dir',
        type=str,
        default='./mimic_admissions',
        help='Path to source MIMIC directory containing patient folders (default: ./mimic_admissions)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./mimic_split',
        help='Path to output directory for split folders (default: ./mimic_split)'
    )
    
    parser.add_argument(
        '--manifest',
        type=str,
        default='./split_manifest.json',
        help='Path to split_manifest.json file (default: ./split_manifest.json)'
    )
    
    parser.add_argument(
        '--move',
        action='store_true',
        default=True,
        help='Move folders instead of copying (default: True)'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print what would be done without actually doing it'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    try:
        split_folders(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            move=args.move,
            dry_run=args.dry_run,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
