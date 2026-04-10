"""
Merge LLM annotation outputs back into the test_annotations.json file.

This script takes JSON annotation outputs from LLM responses and merges them
into the master annotations file.

Usage:
    python merge_annotations.py
    python merge_annotations.py --input annotations_response.json
    python merge_annotations.py --input_dir llm_outputs/
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def load_annotations_file(path: str = "mimic_data/annotations/test_annotations.json") -> Dict:
    """Load the master annotations file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_annotations_file(data: Dict, path: str = "mimic_data/annotations/test_annotations.json") -> None:
    """Save the master annotations file."""
    # Update last_modified
    data["last_modified"] = datetime.now().strftime("%Y-%m-%d")
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved annotations to: {path}")


def validate_annotation(annotation: Dict) -> List[str]:
    """Validate an annotation structure and return list of issues."""
    issues = []
    
    # Check required fields
    required = ["anchor_id", "row_grounding", "relationships"]
    for field in required:
        if field not in annotation:
            issues.append(f"Missing required field: {field}")
    
    # Validate relationships
    if "relationships" in annotation:
        for i, rel in enumerate(annotation["relationships"]):
            if "drug_row" not in rel:
                issues.append(f"Relationship {i}: missing drug_row")
            if "diagnosis_row" not in rel:
                issues.append(f"Relationship {i}: missing diagnosis_row")
            if "relationship_type" not in rel:
                issues.append(f"Relationship {i}: missing relationship_type")
            elif rel["relationship_type"] not in ["TREATS", "ADVERSE_EFFECT", "CONTRAINDICATED", "DISCONTINUED"]:
                issues.append(f"Relationship {i}: invalid relationship_type: {rel['relationship_type']}")
    
    return issues


def merge_single_annotation(
    master: Dict,
    new_annotation: Dict,
    annotator: str = "llm_claude"
) -> bool:
    """
    Merge a single annotation into the master file.
    Handles both single-anchor and combined (dual-anchor) annotations.
    
    Returns True if successful, False otherwise.
    """
    # Check for combined annotation with two anchor IDs
    diag_anchor_id = new_annotation.get("diagnosis_anchor_id")
    med_anchor_id = new_annotation.get("medication_anchor_id")
    single_anchor_id = new_annotation.get("anchor_id")
    
    # Determine which anchor IDs to use
    anchor_ids = []
    if diag_anchor_id is not None:
        anchor_ids.append(diag_anchor_id)
    if med_anchor_id is not None:
        anchor_ids.append(med_anchor_id)
    if single_anchor_id is not None and not anchor_ids:
        anchor_ids.append(single_anchor_id)
    
    if not anchor_ids:
        print(f"[ERROR] No anchor_id in annotation")
        return False
    
    # Find matching entries in master
    target_indices = []
    for anchor_id in anchor_ids:
        for i, entry in enumerate(master["annotations"]):
            if entry.get("anchor_id") == anchor_id:
                target_indices.append(i)
                break
    
    if not target_indices:
        print(f"[ERROR] No matching anchor_ids found: {anchor_ids}")
        return False
    
    # Validate (skip anchor_id check for combined annotations)
    issues = []
    if "row_grounding" not in new_annotation:
        issues.append("Missing required field: row_grounding")
    if "relationships" not in new_annotation:
        issues.append("Missing required field: relationships")
    
    if "relationships" in new_annotation:
        for i, rel in enumerate(new_annotation["relationships"]):
            if "drug_row" not in rel:
                issues.append(f"Relationship {i}: missing drug_row")
            if "diagnosis_row" not in rel:
                issues.append(f"Relationship {i}: missing diagnosis_row")
            if "relationship_type" not in rel:
                issues.append(f"Relationship {i}: missing relationship_type")
            elif rel["relationship_type"] not in ["TREATS", "ADVERSE_EFFECT", "CONTRAINDICATED", "DISCONTINUED"]:
                issues.append(f"Relationship {i}: invalid relationship_type: {rel['relationship_type']}")
    
    if issues:
        print(f"[WARNING] Validation issues for {anchor_ids}:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Merge to all matching entries
    for target_idx in target_indices:
        target = master["annotations"][target_idx]
        
        target["status"] = "completed"
        target["annotator"] = annotator
        target["timestamp"] = datetime.now().isoformat()
        
        if "row_grounding" in new_annotation:
            target["row_grounding"] = new_annotation["row_grounding"]
        
        if "relationships" in new_annotation:
            target["relationships"] = new_annotation["relationships"]
        
        if "multi_relationship_flags" in new_annotation:
            target["multi_relationship_flags"] = new_annotation["multi_relationship_flags"]
        
        if "negative_relationships" in new_annotation:
            target["negative_relationships"] = new_annotation.get("negative_relationships", [])
        
        if "quality_notes" in new_annotation:
            target["quality_notes"] = new_annotation["quality_notes"]
    
    print(f"[OK] Merged to {len(target_indices)} entries")
    return True


def update_statistics(master: Dict) -> None:
    """Update statistics in the master file based on annotations."""
    stats = master["statistics"]
    
    annotated = 0
    pending = 0
    total_rels = 0
    by_type = {"TREATS": 0, "ADVERSE_EFFECT": 0, "CONTRAINDICATED": 0, "DISCONTINUED": 0}
    multi_rels = 0
    
    for entry in master["annotations"]:
        if entry.get("status") == "completed":
            annotated += 1
        else:
            pending += 1
        
        for rel in entry.get("relationships", []):
            total_rels += 1
            rel_type = rel.get("relationship_type", "")
            if rel_type in by_type:
                by_type[rel_type] += 1
        
        multi_rels += len(entry.get("multi_relationship_flags", []))
    
    stats["annotated"] = annotated
    stats["pending"] = pending
    stats["total_relationships"] = total_rels
    stats["by_type"] = by_type
    stats["multi_relationship_cases"] = multi_rels


def merge_file_into_master(input_path: str, master: Dict, annotator: str) -> Dict[str, int]:
    """Merge annotations from a single file into the provided master dictionary."""
    print(f"  [MERGE] Loading: {Path(input_path).name}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to extract JSON from the content
    # LLM responses might have markdown code blocks
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        content = content[start:end].strip()
    elif "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        content = content[start:end].strip()
    
    try:
        new_annotation = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Failed to parse JSON: {e}")
        return {"merged": 0, "failed": 1}
    
    # Handle single annotation or list
    if isinstance(new_annotation, list):
        annotations = new_annotation
    else:
        annotations = [new_annotation]
        
    # Inject missing metadata by matching the filename to the master template
    admission_id = Path(input_path).stem
    if admission_id.startswith("annotation_"):
        admission_id = admission_id.replace("annotation_", "")
        
    for ann in annotations:
        if "admission_id" not in ann:
            ann["admission_id"] = admission_id
            
        if "diagnosis_anchor_id" not in ann or "medication_anchor_id" not in ann:
            for entry in master.get("annotations", []):
                anchor_meta = entry.get("anchor_metadata", "")
                parts = anchor_meta.split("-")
                
                if len(parts) >= 3 and parts[1] == admission_id:
                    # Inject patient_id if missing
                    if "patient_id" not in ann:
                        ann["patient_id"] = parts[0]
                        
                    # Inject anchor IDs based on table_type
                    if parts[2] == "diagnosis" and "diagnosis_anchor_id" not in ann:
                        ann["diagnosis_anchor_id"] = entry.get("anchor_id")
                    elif parts[2] == "medication" and "medication_anchor_id" not in ann:
                        ann["medication_anchor_id"] = entry.get("anchor_id")
    
    merged = 0
    failed = 0
    
    for ann in annotations:
        if merge_single_annotation(master, ann, annotator):
            merged += 1
        else:
            failed += 1
            
    return {"merged": merged, "failed": failed}


def process_annotator_directory(annotator_dir: Path, master_path: str, output_dir: Path) -> Dict[str, int]:
    """Process all annotation files for a single annotator directory."""
    annotator_name = annotator_dir.name
    print(f"\nProcessing Annotator: {annotator_name}")
    
    # Load pristine master
    master = load_annotations_file(master_path)
    
    total_merged = 0
    total_failed = 0
    
    for json_file in annotator_dir.rglob("*.json"):
        result = merge_file_into_master(str(json_file), master, annotator_name)
        total_merged += result["merged"]
        total_failed += result["failed"]
        
    # Update statistics
    update_statistics(master)
    
    # Save isolated output
    output_path = output_dir / f"{annotator_name}.json"
    save_annotations_file(master, str(output_path))
    
    return {"merged": total_merged, "failed": total_failed}


def main():
    parser = argparse.ArgumentParser(
        description="Merge LLM annotations into per-annotator master output files"
    )
    parser.add_argument(
        "--annotators_dir", type=str, default="Annotations/Individual",
        help="Path to directory containing annotator output folders"
    )
    parser.add_argument(
        "--output_dir", type=str, default="Annotations/Merged_Per_Annotator",
        help="Path to save merged per-annotator JSON files"
    )
    parser.add_argument(
        "--master", type=str, default="mimic_data/annotations/test_annotations.json",
        help="Path to master template annotations file"
    )
    
    args = parser.parse_args()
    
    annotators_dir = Path(args.annotators_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("LOKI Batch Annotation Merger")
    print("=" * 60)
    
    if not annotators_dir.exists() or not annotators_dir.is_dir():
        print(f"[ERROR] Annotators directory not found: {args.annotators_dir}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
        
    annotator_folders = sorted([d for d in annotators_dir.iterdir() if d.is_dir()])
    
    if not annotator_folders:
        print(f"[ERROR] No annotator folders found in {args.annotators_dir}")
        return
        
    for annotator_dir in annotator_folders:
        result = process_annotator_directory(annotator_dir, args.master, output_dir)
        print(f"  [DONE {annotator_dir.name}] Merged: {result['merged']}, Failed: {result['failed']}")


if __name__ == "__main__":
    main()
