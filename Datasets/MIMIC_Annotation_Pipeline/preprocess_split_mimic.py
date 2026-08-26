"""
MIMIC Preprocessing for Pre-Split Folders

This script processes MIMIC data that has already been split into separate
train/val/test folders using split_mimic_folders.py.

Usage:
    # Process all splits from mimic_split/
    python preprocess_split_mimic.py
    
    # Custom paths
    python preprocess_split_mimic.py --split_dir ./mimic_split --output_dir ./mimic_data

    # Process only test set (for annotation)
    python preprocess_split_mimic.py --test_only

Input Structure:
    mimic_split/
        train/
            <patient_id>/
                <hadm_id>/
                    <hadm_id>-diagnosis.csv
                    <hadm_id>-medication.csv
                    <hadm_id>-notes.txt
        val/
            <patient_id>/...
        test/
            <patient_id>/...

Output Structure:
    mimic_data/
        train_row_level_v2.json
        val_row_level_v2.json
        test_row_level_v2.json
        annotations/
            test_annotations.json
"""

import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import OrderedDict
from tqdm import tqdm
import csv
import re
import pandas as pd
import random
from collections import defaultdict

# ============================================================================
# V2 TRANSFORMATION (from transform_mimic_data.py)
# ============================================================================

SECTION_PATTERNS = [
    # Administrative sections
    (r"^[\s]*Name:\s*Unit No:", "Header", "administrative"),
    (r"^[\s]*Admission Date:", "Header", "administrative"),
    (r"^[\s]*Date of Birth:", "Header", "administrative"),
    (r"^[\s]*Service:", "Service", "administrative"),
    (r"^[\s]*Attending:", "Attending", "administrative"),
    
    # Clinical sections
    (r"^[\s]*Allergies:", "Allergies", "clinical"),
    (r"^[\s]*Chief Complaint:", "Chief Complaint", "clinical"),
    (r"^[\s]*Major Surgical", "Procedures", "clinical"),
    (r"^[\s]*History of Present Illness:", "History of Present Illness", "clinical"),
    (r"^[\s]*Past Medical History:", "Past Medical History", "clinical"),
    (r"^[\s]*Social History:", "Social History", "clinical"),
    (r"^[\s]*Family History:", "Family History", "clinical"),
    (r"^[\s]*Physical Exam:", "Physical Exam", "clinical"),
    (r"^[\s]*Pertinent Results:", "Pertinent Results", "clinical"),
    (r"^[\s]*Brief Hospital Course:", "Hospital Course", "clinical"),
    (r"^[\s]*Medications on Admission:", "Medications on Admission", "clinical"),
    (r"^[\s]*Discharge Medications:", "Discharge Medications", "clinical"),
    (r"^[\s]*Discharge Disposition:", "Discharge Disposition", "administrative"),
    (r"^[\s]*Discharge Diagnosis:", "Discharge Diagnosis", "clinical"),
    (r"^[\s]*Discharge Condition:", "Discharge Condition", "clinical"),
    (r"^[\s]*Discharge Instructions:", "Discharge Instructions", "clinical"),
    (r"^[\s]*Followup Instructions:", "Followup Instructions", "clinical"),
    
    # Numbered clinical sections
    (r"^[\s]*#\s+[A-Z]", "Clinical Section", "clinical"),
]

def _generate_stable_id(input_str: str) -> int:
    """
    Generate a stable, numeric ID from a string using SHA256 hashing.
    Returns a positive 64-bit integer suitable for JSON.
    """
    hash_bytes = hashlib.sha256(input_str.encode('utf-8')).digest()
    # Use first 8 bytes to get a 64-bit integer
    numeric_id = int.from_bytes(hash_bytes[:8], byteorder='big')
    # Ensure positive (clear the sign bit for 64-bit)
    return numeric_id & ((1 << 63) - 1)

def _simple_sentence_tokenize(text: str) -> List[str]:
    """
    Simple sentence tokenization that handles clinical text patterns.
    """
    if not text or not text.strip():
        return []
    
    import re
    
    # First, normalize whitespace
    text = ' '.join(text.split())
    
    # Split on common sentence boundaries
    # Match: period/exclamation/question followed by space and capital letter
    # Also handle newlines as sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Further split on newlines if they exist
    final_sentences = []
    for sent in sentences:
        # Split on multiple newlines (paragraph breaks)
        parts = re.split(r'\n\s*\n', sent)
        for part in parts:
            part = part.strip()
            if part and len(part) > 3:  # Skip very short fragments
                final_sentences.append(part)
    
    return final_sentences

def load_mimic_admission_data(admission_folder: Path, hidden_cols: Set[str] = None) -> Dict[str, Any]:
    """
    Load all data for a single MIMIC hospital admission from its folder.
    Structure: mimic_100/<subject_id>/<hadm_id>/
    """
    if hidden_cols is None:
        hidden_cols = {'subject_id', 'hadm_id'}
    
    admission_folder = Path(admission_folder)
    hadm_id = admission_folder.name
    subject_id = admission_folder.parent.name
    
    result = {
        'subject_id': subject_id,
        'hadm_id': hadm_id,
        'admission_id': f"{subject_id}-{hadm_id}",
        'diagnosis_table': None,
        'medication_table': None,
        'notes_text': None
    }
    
    # Load diagnosis CSV
    diagnosis_path = admission_folder / f"{hadm_id}-diagnosis.csv"
    if diagnosis_path.exists():
        try:
            df = pd.read_csv(diagnosis_path)
            result['diagnosis_table'] = df
        except Exception as e:
            print(f"Warning: Could not load diagnosis for {subject_id}-{hadm_id}: {e}")
    
    # Load medication CSV
    medication_path = admission_folder / f"{hadm_id}-medication.csv"
    if medication_path.exists():
        try:
            df = pd.read_csv(medication_path)
            result['medication_table'] = df
        except Exception as e:
            print(f"Warning: Could not load medication for {subject_id}-{hadm_id}: {e}")
    
    # Load combined notes
    notes_path = admission_folder / f"{hadm_id}-notes.txt"
    if notes_path.exists():
        try:
            with open(notes_path, 'r', encoding='utf-8-sig') as f:
                notes_text = f.read().strip()
                if notes_text:
                    result['notes_text'] = notes_text
        except Exception as e:
            print(f"Warning: Could not load notes for {subject_id}-{hadm_id}: {e}")
    
    return result

def format_mimic_table_row(row_dict: Dict[str, Any], headers: List[str], 
                           hidden_cols: Set[str]) -> Dict[str, Any]:
    """Format a single MIMIC table row, hiding PHI columns."""
    content = []
    formatted_parts = []
    
    for header in headers:
        value = row_dict.get(header, "")
        if pd.isna(value):
            value = ""
        else:
            value = str(value)
        
        if header in hidden_cols:
            content.append("")
        else:
            content.append(value)
            if value:
                formatted_parts.append(f"{header}: {value}")
    
    formatted = "; ".join(formatted_parts) + "."
    
    return {
        "content": content,
        "formatted": formatted
    }

def convert_mimic_to_protrix_format(patient_admissions: Dict[str, List[Dict[str, Any]]], 
                                   hidden_cols: Set[str] = None,
                                   seed: int = 42) -> List[Dict[str, Any]]:
    """
    Convert MIMIC admission data to protrix-compatible format (admission-centric).
    Creates TWO examples per admission (one for diagnosis table, one for medication table).
    
    REPRODUCIBILITY GUARANTEES:
    - All patient/admission iterations are sorted for deterministic order
    - Negative sampling uses per-example deterministic seeds derived from main seed
    - Hash-based IDs (anchor_id, context_id) are deterministic via SHA256
    """
    if hidden_cols is None:
        hidden_cols = {'subject_id', 'hadm_id'}
    
    examples = []
    
    # Build mapping of all admission notes for negative sampling
    # Sort notes_list by hadm_id for deterministic order
    all_patient_notes = {}
    for subject_id in sorted(patient_admissions.keys()):
        admissions = patient_admissions[subject_id]
        notes_list = []
        for admission in admissions:
            if admission['notes_text']:
                notes_list.append((admission['hadm_id'], admission['notes_text']))
        if notes_list:
            # Sort by hadm_id for deterministic order
            notes_list.sort(key=lambda x: x[0])
            all_patient_notes[subject_id] = notes_list
    
    # CRITICAL: Sort patient IDs for deterministic iteration order
    all_patient_ids = sorted(all_patient_notes.keys())
    
    print(f"Processing {len(patient_admissions)} patients with admissions...")
    
    # CRITICAL: Iterate in sorted order for reproducibility
    for subject_id in tqdm(sorted(patient_admissions.keys()), desc="Converting to protrix format"):
        admissions = patient_admissions[subject_id]
        for admission_idx, admission in enumerate(admissions):
            subject_id = admission['subject_id']
            hadm_id = admission['hadm_id']
            admission_id = admission['admission_id']
            notes_text = admission['notes_text']
            
            if not notes_text:
                continue
            
            primary_sentences = _simple_sentence_tokenize(notes_text)
            if not primary_sentences:
                continue
            
            # Additional positives: other admissions from SAME patient
            additional_positives = []
            for other_idx, other_admission in enumerate(admissions):
                if other_idx == admission_idx:
                    continue
                
                other_hadm_id = other_admission['hadm_id']
                other_notes = other_admission['notes_text']
                
                if other_notes:
                    other_sentences = _simple_sentence_tokenize(other_notes)
                    if other_sentences:
                        other_context_id = _generate_stable_id(f"{subject_id}_{other_hadm_id}_notes")
                        additional_positives.append({
                            "id": other_context_id,
                            "metadata": f"{subject_id}-{other_hadm_id}-notes",
                            "sentences": other_sentences,
                            "distance": 0.5
                        })
            
            # Balanced number of negatives
            num_positives = 1 + len(additional_positives)
            num_negatives = num_positives
            
            # Sample negative contexts from DIFFERENT patients
            # CRITICAL: Use deterministic per-example seed for reproducible negative sampling
            # Derive seed from global seed + example identifier
            example_seed = seed + _generate_stable_id(f"{subject_id}_{hadm_id}_neg_sample") % (2**31)
            example_rng = random.Random(example_seed)
            
            negatives = []
            other_patient_ids = [pid for pid in all_patient_ids if pid != subject_id]
            
            if other_patient_ids:
                sampled_patient_ids = example_rng.sample(
                    other_patient_ids, 
                    min(num_negatives, len(other_patient_ids))
                )
                
                for neg_patient_id in sampled_patient_ids:
                    neg_notes_list = all_patient_notes.get(neg_patient_id, [])
                    if neg_notes_list:
                        # Use same deterministic RNG for choosing among notes
                        neg_hadm_id, neg_notes_text = example_rng.choice(neg_notes_list)
                        neg_sentences = _simple_sentence_tokenize(neg_notes_text)
                        if neg_sentences:
                            neg_context_id = _generate_stable_id(f"{neg_patient_id}_{neg_hadm_id}_notes")
                            negatives.append({
                                "id": neg_context_id,
                                "metadata": f"{neg_patient_id}-{neg_hadm_id}-notes",
                                "sentences": neg_sentences,
                                "distance": 10.0
                            })
            
            # Primary positive context
            primary_context_id = _generate_stable_id(f"{subject_id}_{hadm_id}_notes")
            primary_positive = {
                "id": primary_context_id,
                "metadata": f"{subject_id}-{hadm_id}-notes",
                "sentences": primary_sentences,
                "distance": 0.0
            }
            
            # Process BOTH tables for this admission
            for table_type in ['diagnosis', 'medication']:
                table_key = f'{table_type}_table'
                df = admission[table_key]
                
                if df is None or len(df) == 0:
                    continue
                
                anchor_id = _generate_stable_id(f"{subject_id}_{hadm_id}_{table_type}")
                anchor_metadata = f"{subject_id}-{hadm_id}-{table_type}"
                
                headers = df.columns.tolist()
                visible_headers = [h for h in headers if h not in hidden_cols]
                
                anchor_rows = []
                for idx, (_, row) in enumerate(df.iterrows(), start=1):
                    row_dict = row.to_dict()
                    formatted_row = format_mimic_table_row(row_dict, headers, hidden_cols)
                    formatted_row['row_idx'] = idx
                    anchor_rows.append(formatted_row)
                
                if not anchor_rows:
                    continue
                
                example = {
                    "anchor_id": anchor_id,
                    "anchor_metadata": anchor_metadata,
                    "anchor_headers": visible_headers,
                    "anchor_rows": anchor_rows,
                    "primary_positive": primary_positive,
                    "additional_positives": additional_positives,
                    "negatives": negatives,
                    "threshold": 5.0
                }
                
                examples.append(example)
    
    print(f"Generated {len(examples)} examples total")
    return examples

def detect_sections(sentences: List[str]) -> List[Dict[str, Any]]:
    """Detect clinical note sections from sentence list."""
    import re
    
    if not sentences:
        return []
    
    sections = []
    current_section = {
        "section_idx": 0,
        "section_name": "Preamble",
        "section_type": "administrative",
        "start_sentence_idx": 0
    }
    
    for idx, sentence in enumerate(sentences):
        sentence_stripped = sentence.strip()
        
        for pattern, name, section_type in SECTION_PATTERNS:
            if re.search(pattern, sentence_stripped, re.IGNORECASE):
                if idx > current_section["start_sentence_idx"]:
                    current_section["end_sentence_idx"] = idx - 1
                    current_section["sentence_count"] = idx - current_section["start_sentence_idx"]
                    sections.append(current_section.copy())
                
                current_section = {
                    "section_idx": len(sections),
                    "section_name": name,
                    "section_type": section_type,
                    "start_sentence_idx": idx
                }
                break
    
    current_section["end_sentence_idx"] = len(sentences) - 1
    current_section["sentence_count"] = len(sentences) - current_section["start_sentence_idx"]
    if current_section["sentence_count"] > 0:
        sections.append(current_section)
    
    return sections

def get_section_for_sentence(sentence_idx: int, sections: List[Dict]) -> Tuple[int, str]:
    """Get section index and name for a given sentence index."""
    for section in sections:
        if section["start_sentence_idx"] <= sentence_idx <= section.get("end_sentence_idx", float('inf')):
            return section["section_idx"], section["section_name"]
    return 0, "Unknown"

def build_indexed_sentences(sentences: List[str], sections: List[Dict]) -> Dict[str, Dict]:
    """Build indexed sentences dictionary with section metadata."""
    sentences_dict = OrderedDict()
    char_position = 0
    
    for idx, sentence_text in enumerate(sentences):
        section_idx, section_name = get_section_for_sentence(idx, sections)
        sentences_dict[str(idx)] = {
            "text": sentence_text,
            "section_idx": section_idx,
            "section_name": section_name,
            "char_start": char_position,
            "char_end": char_position + len(sentence_text)
        }
        char_position += len(sentence_text) + 1
    
    return sentences_dict

def transform_document(doc_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a document to enhanced format with section detection."""
    sentences_list = doc_data.get("sentences", [])
    
    if not sentences_list:
        return {
            "id": doc_data.get("id"),
            "metadata": doc_data.get("metadata", ""),
            "distance": doc_data.get("distance", 0.0),
            "total_sentences": 0,
            "total_sections": 0,
            "sections": [],
            "sentences": {}
        }
    
    sections = detect_sections(sentences_list)
    sentences_indexed = build_indexed_sentences(sentences_list, sections)
    
    return {
        "id": doc_data.get("id"),
        "metadata": doc_data.get("metadata", ""),
        "distance": doc_data.get("distance", 0.0),
        "total_sentences": len(sentences_list),
        "total_sections": len(sections),
        "sections": sections,
        "sentences": sentences_indexed
    }

def transform_example_to_v2(example: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a single example to enhanced format v2."""
    anchor_metadata = example.get("anchor_metadata", "")
    parts = anchor_metadata.split("-")
    patient_id = parts[0] if len(parts) > 0 else ""
    admission_id = parts[1] if len(parts) > 1 else ""
    table_type = parts[2] if len(parts) > 2 else ""
    
    # Build table structure
    tables = {}
    table_key = "diagnosis" if "diagnosis" in table_type.lower() else "medication"
    tables[table_key] = {
        "headers": example.get("anchor_headers", []),
        "rows": example.get("anchor_rows", [])
    }
    
    # Transform documents
    primary_doc = transform_document(example.get("primary_positive", {}))
    
    additional_positives = []
    for add_pos in example.get("additional_positives", []):
        additional_positives.append(transform_document(add_pos))
    
    negatives = []
    for neg in example.get("negatives", []):
        negatives.append(transform_document(neg))
    
    return {
        "schema_version": "2.0",
        "data_format": "loki_mimic_v2",
        "example_id": f"{patient_id}-{admission_id}-{table_type}",
        "patient_id": patient_id,
        "admission_id": admission_id,
        "anchor_id": example.get("anchor_id"),
        "anchor_metadata": anchor_metadata,
        "tables": tables,
        "primary_positive": primary_doc,
        "additional_positives": additional_positives,
        "negatives": negatives,
        "threshold": example.get("threshold", 5.0)
    }

def generate_annotations_file(test_data: List[Dict], output_path: str, verbose: bool = True) -> Dict[str, Any]:
    """Generate blank annotations file from enhanced test data."""
    if verbose:
        print(f"\n[ANNOTATIONS] Generating annotations file...")
    
    annotations_file = {
        "schema_version": "2.0",
        "annotation_type": "loki_mimic_ground_truth",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "last_modified": datetime.now().strftime("%Y-%m-%d"),
        
        "indexing_convention": {
            "row_idx": "1-based (matches data file)",
            "sentence_idx": "0-based (Python convention)"
        },
        
        "relationship_types": [
            "TREATS",
            "ADVERSE_EFFECT",
            "CONTRAINDICATED",
            "DISCONTINUED"
        ],
        
        "statistics": {
            "total_examples": len(test_data),
            "annotated": 0,
            "pending": len(test_data),
            "total_relationships": 0,
            "by_type": {
                "TREATS": 0,
                "ADVERSE_EFFECT": 0,
                "CONTRAINDICATED": 0,
                "DISCONTINUED": 0
            },
            "multi_relationship_cases": 0
        },
        
        "annotations": []
    }
    
    for example in test_data:
        anchor_id = example.get("anchor_id")
        anchor_metadata = example.get("anchor_metadata", "")
        
        table_type = "diagnosis" if "diagnosis" in anchor_metadata.lower() else "medication"
        
        tables = example.get("tables", {})
        table_data = tables.get(table_type, {})
        rows = table_data.get("rows", [])
        
        primary_doc = example.get("primary_positive", {})
        num_sentences = primary_doc.get("total_sentences", 0)
        
        blank_annotation = {
            "anchor_id": anchor_id,
            "anchor_metadata": anchor_metadata,
            "table_type": table_type,
            "reference_info": {
                "num_rows": len(rows),
                "num_sentences": num_sentences,
                "row_indices": [r.get("row_idx") for r in rows if isinstance(r, dict)]
            },
            "status": "pending",
            "annotator": None,
            "reviewer": None,
            "timestamp": None,
            "row_grounding": {
                "diagnosis": {},
                "medication": {}
            },
            "relationships": [],
            "multi_relationship_flags": [],
            "quality_notes": None
        }
        
        annotations_file["annotations"].append(blank_annotation)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotations_file, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"   [OK] Generated {len(test_data)} blank annotations")
        print(f"   [FILE] Output: {output_path}")
    
    return {"total_examples": len(test_data), "output_path": str(output_path)}

# --- End of injected dependencies ---

def collect_admissions_from_split(split_dir: Path, hidden_cols: Set[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect all admissions from a split folder (train/val/test).
    """
    if hidden_cols is None:
        hidden_cols = {'subject_id', 'hadm_id'}
    
    patient_admissions = {}
    
    if not split_dir.exists():
        print(f"Warning: Split directory does not exist: {split_dir}")
        return patient_admissions
    
    patient_folders = sorted([f for f in split_dir.iterdir() if f.is_dir()])
    
    print(f"  Scanning {len(patient_folders)} patient folders in {split_dir.name}/...")
    
    for patient_folder in tqdm(patient_folders, desc=f"  Loading {split_dir.name}"):
        patient_id = patient_folder.name
        patient_admissions[patient_id] = []
        
        admission_folders = sorted([f for f in patient_folder.iterdir() if f.is_dir()])
        
        has_valid_admission = False
        for admission_folder in admission_folders:
            admission_data = load_mimic_admission_data(admission_folder, hidden_cols)
            
            # Check if this admission has notes
            if admission_data.get('notes_text'):
                patient_admissions[patient_id].append(admission_data)
                has_valid_admission = True
        
        # Remove patients without valid admissions
        if not has_valid_admission:
            del patient_admissions[patient_id]
    
    return patient_admissions


def process_split(
    split_dir: Path,
    split_name: str,
    hidden_cols: Set[str],
    seed: int,
    skip_v2: bool = False
) -> List[Dict[str, Any]]:
    """
    Process a single split (train/val/test) and return examples.
    """
    print(f"\n[Processing {split_name.upper()}]")
    
    # Collect admissions
    patient_admissions = collect_admissions_from_split(split_dir, hidden_cols)
    
    if not patient_admissions:
        print(f"  Warning: No valid admissions found in {split_dir}")
        return []
    
    print(f"  Found {len(patient_admissions)} patients with valid admissions")
    
    # Convert to protrix format
    examples = convert_mimic_to_protrix_format(
        patient_admissions,
        hidden_cols=hidden_cols,
        seed=seed
    )
    
    # Transform to v2 if needed
    if not skip_v2:
        print(f"  Transforming {len(examples)} examples to v2 format...")
        examples = [transform_example_to_v2(ex) for ex in tqdm(examples, desc="  v2 transform")]
    
    return examples


def preprocess_split_mimic(
    split_dir: str,
    output_dir: str,
    hidden_cols: Set[str] = None,
    seed: int = 42,
    skip_v2: bool = False,
    generate_annotations: bool = True,
    test_only: bool = False
):
    """
    Process pre-split MIMIC folders into v2 format.
    
    Args:
        split_dir: Path to mimic_split/ folder containing train/val/test subfolders
        output_dir: Directory to save output files
        hidden_cols: Set of column names to hide (default: subject_id, hadm_id)
        seed: Random seed for reproducibility
        skip_v2: If True, only generate protrix format (no v2 transformation)
        generate_annotations: If True, generate blank annotations for test set
        test_only: If True, only process the test split
    """
    if hidden_cols is None:
        hidden_cols = {'subject_id', 'hadm_id'}
    
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MIMIC Pre-Split Preprocessing Pipeline")
    print("=" * 60)
    print(f"Input: {split_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'Test only' if test_only else 'All splits'}")
    print(f"Output format: {'protrix' if skip_v2 else 'v2 (enhanced)'}")
    print("=" * 60)
    
    # Determine which splits to process
    splits_to_process = ['test'] if test_only else ['train', 'val', 'test']
    
    results = {}
    
    for split_name in splits_to_process:
        split_path = split_dir / split_name
        
        if not split_path.exists():
            print(f"\nWarning: {split_name}/ folder not found, skipping...")
            continue
        
        examples = process_split(
            split_path,
            split_name,
            hidden_cols,
            seed,
            skip_v2
        )
        
        results[split_name] = examples
        
        # Save to file
        output_file = output_dir / f"{split_name}_row_level_v2.json"
        print(f"  Saving {len(examples)} examples to {output_file.name}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2)
    
    # Generate annotations for test set
    if generate_annotations and 'test' in results and results['test']:
        print("\n[Generating Annotation Templates]")
        annotations_dir = output_dir / "annotations"
        annotations_dir.mkdir(exist_ok=True)
        
        generate_annotations_file(
            results['test'],
            annotations_dir / "test_annotations.json"
        )
    
    # Create processing summary
    summary = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "configuration": {
            "split_dir": str(split_dir),
            "output_dir": str(output_dir),
            "seed": seed,
            "skip_v2": skip_v2,
            "test_only": test_only
        },
        "statistics": {
            split_name: len(examples) for split_name, examples in results.items()
        }
    }
    
    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split_name, examples in results.items():
        print(f"  {split_name}: {len(examples)} examples")
    print(f"\nOutput saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Process pre-split MIMIC folders into v2 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all splits
    python preprocess_split_mimic.py
    
    # Process only test set (for annotation)
    python preprocess_split_mimic.py --test_only
    
    # Custom paths
    python preprocess_split_mimic.py --split_dir ./mimic_split --output_dir ./mimic_data
"""
    )
    
    parser.add_argument(
        "--split_dir", type=str, default="./mimic_split",
        help="Path to split folder containing train/val/test subfolders (default: ./mimic_split)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./mimic_data",
        help="Directory to save output files (default: ./mimic_data)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--skip_v2", action="store_true",
        help="Skip v2 transformation (only generate protrix format)"
    )
    parser.add_argument(
        "--no_annotations", action="store_true",
        help="Skip generating annotation templates"
    )
    parser.add_argument(
        "--test_only", action="store_true",
        help="Only process the test split (for annotation workflow)"
    )
    
    args = parser.parse_args()
    
    preprocess_split_mimic(
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        skip_v2=args.skip_v2,
        generate_annotations=not args.no_annotations,
        test_only=args.test_only
    )


if __name__ == "__main__":
    main()
