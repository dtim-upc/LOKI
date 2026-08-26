import json
import os
import hashlib
from pathlib import Path
import re

def _generate_stable_id(input_str: str) -> int:
    """
    Generate a stable, numeric ID from a string using SHA256 hashing.
    Matches the logic in preprocess_split_mimic.py.
    """
    hash_bytes = hashlib.sha256(input_str.encode('utf-8')).digest()
    numeric_id = int.from_bytes(hash_bytes[:8], byteorder='big')
    # Ensure positive (clear the sign bit for 64-bit)
    return numeric_id & ((1 << 63) - 1)

def patch_file(file_path: Path, patient_id: str, admission_id: str):
    """Patch a single JSON file with correct metadata."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        updated = False
        
        # Calculate expected IDs
        diag_id = _generate_stable_id(f"{patient_id}_{admission_id}_diagnosis")
        med_id = _generate_stable_id(f"{patient_id}_{admission_id}_medication")
        
        # Check and update fields
        if data.get('patient_id') != patient_id:
            data['patient_id'] = patient_id
            updated = True
        
        if data.get('admission_id') != admission_id:
            data['admission_id'] = admission_id
            updated = True
            
        if data.get('diagnosis_anchor_id') != diag_id:
            data['diagnosis_anchor_id'] = diag_id
            updated = True
            
        if data.get('medication_anchor_id') != med_id:
            data['medication_anchor_id'] = med_id
            updated = True
            
        if updated:
            print(f"  [PATCHED] {file_path.name}")
            # Reorder fields for readability (metadata first)
            new_data = {
                "patient_id": data["patient_id"],
                "admission_id": data["admission_id"],
                "diagnosis_anchor_id": data["diagnosis_anchor_id"],
                "medication_anchor_id": data["medication_anchor_id"],
                "row_grounding": data.get("row_grounding", {}),
                "relationships": data.get("relationships", []),
                "multi_relationship_flags": data.get("multi_relationship_flags", []),
                "negative_relationships": data.get("negative_relationships", []),
                "quality_notes": data.get("quality_notes", "")
            }
            # Keep any other keys
            for k, v in data.items():
                if k not in new_data:
                    new_data[k] = v
                    
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=2)
            return True
        return False
    except Exception as e:
        print(f"Error patching {file_path}: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify and patch annotation metadata.")
    parser.add_argument('--input_dir', '-i', type=str, 
                        default='Annotations/Individual',
                        help='Directory containing annotator folders (default: Annotations/Individual)')
    args = parser.parse_args()

    # Resolve paths relative to the script location
    script_dir = Path(__file__).parent
    root_dir = script_dir / args.input_dir
    
    if not root_dir.exists():
        print(f"Error: Input directory not found: {root_dir}")
        return

    # Dynamically find annotator subdirectories
    annotators = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
    
    if not annotators:
        print(f"No annotator subdirectories found in {root_dir}")
        return

    # Load manifest for root-level lookups (if needed)
    manifest_path = script_dir / "split_manifest.json"
    
    for ann in annotators:
        ann_path = root_dir / ann
        print(f"Processing {ann}...")
        patched_count = 0
        verified_count = 0
        total_count = 0
        
        # 1. Process subdirectories (Patient folders)
        for patient_dir in ann_path.iterdir():
            if patient_dir.is_dir():
                patient_id = patient_dir.name
                for json_file in patient_dir.glob("*.json"):
                    total_count += 1
                    filename = json_file.name
                    admission_id = re.search(r'(\d+)', filename)
                    if admission_id:
                        if patch_file(json_file, patient_id, admission_id.group(1)):
                            patched_count += 1
                        else:
                            verified_count += 1
                            
        # 2. Process root-level files (Legacy files)
        for json_file in ann_path.glob("*.json"):
            total_count += 1
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                patient_id = data.get('patient_id')
                admission_id = data.get('admission_id', json_file.stem)
                
                if patient_id and admission_id:
                    if patch_file(json_file, patient_id, admission_id):
                        patched_count += 1
                    else:
                        verified_count += 1
                else:
                    print(f"Warning: Root file {json_file.name} is missing metadata.")
            except:
                pass

        print(f"Finished {ann}: Patched {patched_count}, Verified {verified_count} / {total_count} files.")

if __name__ == "__main__":
    main()
