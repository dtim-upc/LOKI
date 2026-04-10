#!/usr/bin/env python3
"""
Finalize Annotations

Takes the voted ground truth JSON and injects it back into the master blank template.
"""

import json
import argparse
from pathlib import Path
import sys

# Import core merging logic from existing script
try:
    from merge_annotations import merge_single_annotation, update_statistics
except ImportError:
    print("Error: merge_annotations.py not found in the same directory.")
    sys.exit(1)

def load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: dict, path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Inject voted ground truth into the master template.")
    parser.add_argument('--voted_file', '-v', type=str,
                        default='Annotations/Voting/merged_annotations_all.json',
                        help='Path to the merged voted ground truth (default: Annotations/Voting/merged_annotations_all.json)')
    parser.add_argument('--template_file', '-t', type=str,
                        default='mimic_data/annotations/test_annotations.json',
                        help='Path to the blank master template (default: mimic_data/annotations/test_annotations.json)')
    parser.add_argument('--output_file', '-o', type=str,
                        default='mimic_data/annotations/test_annotations.json',
                        help='Output file to write the merged template (default: overrides template_file)')
    
    args = parser.parse_args()
    
    voted_path = Path(args.voted_file)
    template_path = Path(args.template_file)
    output_path = Path(args.output_file)
    
    if not voted_path.exists():
        print(f"Error: Voted file not found: {voted_path}")
        return 1
        
    if not template_path.exists():
        print(f"Error: Template file not found: {template_path}")
        return 1
        
    print(f"Loading voted ground truth from: {voted_path}")
    voted_data = load_json(voted_path)
    
    print(f"Loading master template from: {template_path}")
    template_data = load_json(template_path)
    
    # Process each merged admission
    success_count = 0
    fail_count = 0
    
    for admission_id, annotation_payload in voted_data.items():
        if "admission_id" not in annotation_payload:
            annotation_payload["admission_id"] = admission_id
            
        if not annotation_payload.get("diagnosis_anchor_id") or not annotation_payload.get("medication_anchor_id"):
            for entry in template_data.get("annotations", []):
                anchor_meta = entry.get("anchor_metadata", "")
                parts = anchor_meta.split("-")
                
                if len(parts) >= 3 and parts[1] == admission_id:
                    if "patient_id" not in annotation_payload:
                        annotation_payload["patient_id"] = parts[0]
                        
                    if parts[2] == "diagnosis" and not annotation_payload.get("diagnosis_anchor_id"):
                        annotation_payload["diagnosis_anchor_id"] = entry.get("anchor_id")
                    elif parts[2] == "medication" and not annotation_payload.get("medication_anchor_id"):
                        annotation_payload["medication_anchor_id"] = entry.get("anchor_id")
                        
        # merge_single_annotation handles finding the proper anchors inside the template
        if merge_single_annotation(template_data, annotation_payload, annotator="Merged_Voting"):
            # Note: because each admission matches TWO anchors (med & diag), 
            # merge_single_annotation correctly reports "Merged to 2 entries"
            success_count += 1
        else:
            print(f"[ERROR] Failed to merge admission {admission_id}")
            fail_count += 1
            
    print(f"Processed {len(voted_data)} admissions: {success_count} successful, {fail_count} failed.")
    
    print("Updating global statistics...")
    update_statistics(template_data)
    
    print(f"Saving final dataset to: {output_path}")
    save_json(template_data, output_path)
    print("Done!")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
