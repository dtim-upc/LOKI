#!/usr/bin/env python3
"""
Data Quality Check for Merged Annotations
Comprehensive validation script for training data
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
import argparse
from typing import Dict, Optional


def load_merged_annotations(file_path):
    """Load merged annotations file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_sentence_bounds(data_files) -> Optional[Dict[str, int]]:
    """Build admission_id -> max_valid_sentence_index from one or more row-level v2 files.
    Returns None if no files exist (bounds checking will be skipped).
    """
    bounds: Dict[str, int] = {}
    loaded = 0
    for path_str in data_files:
        p = Path(path_str)
        if not p.exists():
            continue
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for ex in data:
            adm = str(ex.get('admission_id', ''))
            sents = ex.get('primary_positive', {}).get('sentences', {})
            if adm and sents:
                max_key = max(int(k) for k in sents.keys())
                if adm not in bounds or max_key > bounds[adm]:
                    bounds[adm] = max_key
        loaded += 1
    return bounds if loaded > 0 else None


def check_data_quality(merged_data, sentence_bounds: Optional[Dict[str, int]] = None):
    """Perform comprehensive data quality checks."""

    issues = []
    warnings = []
    stats = defaultdict(int)

    stats['total_files'] = len(merged_data)
    oob_check_available = sentence_bounds is not None
    stats['sentence_bounds_available'] = int(oob_check_available)
    
    mention_types_seen = Counter()
    relationship_types_seen = Counter()
    evidence_scopes_seen = Counter()
    confidence_levels_seen = Counter()
    
    for file_id, annotation in merged_data.items():
        stats['total_annotations'] += 1
        
        # 1. Check required fields
        required_fields = ['patient_id', 'admission_id', 'row_grounding', 'relationships']
        for field in required_fields:
            if field not in annotation:
                issues.append(f"[{file_id}] Missing required field: {field}")
        
        # Determine max valid sentence index for this annotation (if available)
        max_sent_idx: Optional[int] = None
        if oob_check_available:
            adm_id_for_bounds = annotation.get('admission_id', file_id)
            max_sent_idx = sentence_bounds.get(str(adm_id_for_bounds))

        # 2. Check row_grounding structure
        if 'row_grounding' in annotation:
            rg = annotation['row_grounding']
            if 'diagnosis' not in rg or 'medication' not in rg:
                issues.append(f"[{file_id}] row_grounding missing diagnosis or medication")
            
            # Check mention types in diagnosis
            if 'diagnosis' in rg:
                for row_id, row_data in rg['diagnosis'].items():
                    if 'mention_types' not in row_data:
                        issues.append(f"[{file_id}] Diagnosis row {row_id} missing mention_types")
                    else:
                        for mt in row_data['mention_types']:
                            mention_types_seen[f"diagnosis:{mt}"] += 1
                            # Check for invalid mention types
                            valid_types = ['explicit', 'abbreviated', 'brand_name', 'synonym', 'context', 'section']
                            if mt not in valid_types:
                                warnings.append(f"[{file_id}] Diagnosis row {row_id}: unusual mention_type '{mt}'")

                    # Check for empty sentences (ungrounded row)
                    sentences = row_data.get('sentences', [])
                    if not sentences:
                        warnings.append(f"[{file_id}] Diagnosis row {row_id} has empty sentences (no textual grounding)")
                    elif 'mention_types' in row_data and len(sentences) != len(row_data['mention_types']):
                        warnings.append(f"[{file_id}] Diagnosis row {row_id}: sentences/mention_types length mismatch "
                                        f"({len(sentences)} sentences vs {len(row_data['mention_types'])} types)")

                    # OOB sentence index check
                    if max_sent_idx is not None:
                        for s in row_data.get('sentences', []):
                            if int(s) > max_sent_idx:
                                issues.append(f"[{file_id}] Diagnosis row {row_id}: sentence index {s} "
                                              f"exceeds document max ({max_sent_idx}) — run repair_oob_sentence_indices.py")
                                stats['oob_sentence_refs'] += 1

                    if '_sources' not in row_data:
                        warnings.append(f"[{file_id}] Diagnosis row {row_id} missing provenance (_sources)")
                    
                stats['diagnosis_rows'] += 1
            
            # Check medication grounding
            if 'medication' in rg:
                for row_id, row_data in rg['medication'].items():
                    if 'mention_types' not in row_data:
                        issues.append(f"[{file_id}] Medication row {row_id} missing mention_types")
                    else:
                        for mt in row_data['mention_types']:
                            mention_types_seen[f"medication:{mt}"] += 1
                            valid_types = ['explicit', 'abbreviated', 'brand_name', 'synonym', 'context', 'section']
                            if mt not in valid_types:
                                warnings.append(f"[{file_id}] Medication row {row_id}: unusual mention_type '{mt}'")

                    # Check for empty sentences (ungrounded row)
                    sentences = row_data.get('sentences', [])
                    if not sentences:
                        warnings.append(f"[{file_id}] Medication row {row_id} has empty sentences (no textual grounding)")
                    elif 'mention_types' in row_data and len(sentences) != len(row_data['mention_types']):
                        warnings.append(f"[{file_id}] Medication row {row_id}: sentences/mention_types length mismatch "
                                        f"({len(sentences)} sentences vs {len(row_data['mention_types'])} types)")

                    # OOB sentence index check
                    if max_sent_idx is not None:
                        for s in row_data.get('sentences', []):
                            if int(s) > max_sent_idx:
                                issues.append(f"[{file_id}] Medication row {row_id}: sentence index {s} "
                                              f"exceeds document max ({max_sent_idx}) — run repair_oob_sentence_indices.py")
                                stats['oob_sentence_refs'] += 1

                    if '_sources' not in row_data:
                        warnings.append(f"[{file_id}] Medication row {row_id} missing provenance (_sources)")
                    
                stats['medication_rows'] += 1
        
        # Build sets of grounded row IDs for cross-field consistency checks
        grounded_med_rows = set()
        grounded_diag_rows = set()
        if 'row_grounding' in annotation:
            for k in annotation['row_grounding'].get('medication', {}):
                try:
                    grounded_med_rows.add(int(k))
                except (ValueError, TypeError):
                    pass
            for k in annotation['row_grounding'].get('diagnosis', {}):
                try:
                    grounded_diag_rows.add(int(k))
                except (ValueError, TypeError):
                    pass

        # 3. Check relationships
        if 'relationships' in annotation:
            for rel in annotation['relationships']:
                stats['total_relationships'] += 1
                
                # Check required relationship fields
                rel_fields = ['id', 'drug_row', 'diagnosis_row', 'relationship_type', 
                            'evidence_sentences', 'evidence_scope', 'confidence']
                for field in rel_fields:
                    if field not in rel:
                        issues.append(f"[{file_id}] Relationship {rel.get('id', 'UNKNOWN')} missing: {field}")
                
                # Track relationship types
                if 'relationship_type' in rel:
                    relationship_types_seen[rel['relationship_type']] += 1
                
                # Track and validate evidence scopes
                valid_scopes = ['explicit', 'section', 'document']
                if 'evidence_scope' in rel:
                    scope_val = rel['evidence_scope']
                    evidence_scopes_seen[scope_val] += 1
                    if scope_val not in valid_scopes:
                        warnings.append(f"[{file_id}] Relationship {rel.get('id', 'UNKNOWN')}: "
                                        f"invalid evidence_scope '{scope_val}' (expected one of {valid_scopes})")

                # Track confidence levels
                if 'confidence' in rel:
                    confidence_levels_seen[rel['confidence']] += 1

                # Cross-field consistency: verify referenced rows exist in row_grounding
                if 'drug_row' in rel and grounded_med_rows:
                    try:
                        if int(rel['drug_row']) not in grounded_med_rows:
                            issues.append(f"[{file_id}] Relationship {rel.get('id', 'UNKNOWN')}: "
                                          f"drug_row {rel['drug_row']} not found in row_grounding['medication']")
                    except (ValueError, TypeError):
                        pass
                if 'diagnosis_row' in rel and grounded_diag_rows:
                    try:
                        if int(rel['diagnosis_row']) not in grounded_diag_rows:
                            issues.append(f"[{file_id}] Relationship {rel.get('id', 'UNKNOWN')}: "
                                          f"diagnosis_row {rel['diagnosis_row']} not found in row_grounding['diagnosis']")
                    except (ValueError, TypeError):
                        pass

                # OOB check on evidence_sentences
                if max_sent_idx is not None:
                    for s in rel.get('evidence_sentences', []):
                        if int(s) > max_sent_idx:
                            issues.append(f"[{file_id}] Relationship {rel.get('id', 'UNKNOWN')}: "
                                          f"evidence_sentence {s} exceeds document max ({max_sent_idx}) "
                                          f"— run repair_oob_sentence_indices.py")
                            stats['oob_sentence_refs'] += 1

                # Check provenance
                if '_provenance' not in rel:
                    warnings.append(f"[{file_id}] Relationship {rel.get('id')} missing provenance")
                else:
                    prov = rel['_provenance']
                    if 'vote_count' in prov:
                        stats[f"rel_votes_{prov['vote_count']}-way"] += 1
                    if 'agreement_level' in prov:
                        stats[f"rel_agreement_{prov['agreement_level']}"] += 1
                
                # Check for empty evidence
                if 'evidence_sentences' in rel and not rel['evidence_sentences']:
                    warnings.append(f"[{file_id}] Relationship {rel.get('id')} has no evidence sentences")
        
        # 4. Check merge metadata
        if '_merge_metadata' not in annotation:
            warnings.append(f"[{file_id}] Missing merge metadata")
        else:
            metadata = annotation['_merge_metadata']
            if metadata.get('n_annotators', 0) != 3:
                warnings.append(f"[{file_id}] Expected 3 annotators, found {metadata.get('n_annotators')}")
    
    return {
        'issues': issues,
        'warnings': warnings,
        'stats': dict(stats),
        'mention_types': dict(mention_types_seen),
        'relationship_types': dict(relationship_types_seen),
        'evidence_scopes': dict(evidence_scopes_seen),
        'confidence_levels': dict(confidence_levels_seen),
    }

def print_quality_report(results):
    """Print formatted quality check report."""
    
    print("\n" + "="*80)
    print("  DATA QUALITY REPORT - MERGED ANNOTATIONS")
    print("="*80 + "\n")
    
    print("[OVERVIEW]")
    print("-" * 40)
    stats = results['stats']
    print(f"  Total Files:               {stats.get('total_files', 0)}")
    print(f"  Total Relationships:       {stats.get('total_relationships', 0)}")
    print(f"  Diagnosis Rows:            {stats.get('diagnosis_rows', 0)}")
    print(f"  Medication Rows:           {stats.get('medication_rows', 0)}")
    print()
    
    print("[DATA QUALITY]")
    print("-" * 40)
    n_issues = len(results['issues'])
    n_warnings = len(results['warnings'])
    print(f"  Critical Issues:           {n_issues}")
    print(f"  Warnings:                  {n_warnings}")
    
    if n_issues == 0 and n_warnings == 0:
        print("  ✓ All quality checks PASSED!")
    print()
    
    if n_issues > 0:
        print("[CRITICAL ISSUES]")
        print("-" * 40)
        for issue in results['issues'][:10]:  # Show first 10
            print(f"  • {issue}")
        if n_issues > 10:
            print(f"  ... and {n_issues - 10} more issues")
        print()
    
    if n_warnings > 0:
        print("[WARNINGS]")
        print("-" * 40)
        for warning in results['warnings'][:10]:  # Show first 10
            print(f"  [!] {warning}")
        if n_warnings > 10:
            print(f"  ... and {n_warnings - 10} more warnings")
        print()
    
    print("[MENTION TYPES DISTRIBUTION]")
    print("-" * 40)
    for mt, count in sorted(results['mention_types'].items()):
        table, mtype = mt.split(':')
        print(f"  {table:15} {mtype:15} {count:>6}")
    print()
    
    print("[RELATIONSHIP TYPES]")
    print("-" * 40)
    for rt, count in results['relationship_types'].items():
        print(f"  {rt:25} {count:>6}")
    print()
    
    print("[EVIDENCE SCOPES]")
    print("-" * 40)
    for es, count in results['evidence_scopes'].items():
        print(f"  {es:25} {count:>6}")
    print()
    
    print("[CONFIDENCE LEVELS]")
    print("-" * 40)
    for conf, count in results['confidence_levels'].items():
        print(f"  {conf:25} {count:>6}")
    print()
    
    print("[AGREEMENT STATISTICS]")
    print("-" * 40)
    for key, value in stats.items():
        if key.startswith('rel_'):
            print(f"  {key:30} {value:>6}")
    print()

    if stats.get('sentence_bounds_available'):
        print("[SENTENCE BOUNDS CHECK]")
        print("-" * 40)
        oob = stats.get('oob_sentence_refs', 0)
        if oob == 0:
            print("  ✓ No out-of-bounds sentence indices detected")
        else:
            print(f"  OOB sentence index refs (CRITICAL): {oob}")
            print("  Run: python repair_oob_sentence_indices.py")
        print()
    else:
        print("[SENTENCE BOUNDS CHECK]")
        print("-" * 40)
        print("  (skipped — pass --data_files to enable)")
        print()
    
    print("="*80)
    if n_issues == 0:
        print("  [OK] DATA READY FOR MODEL TRAINING")
    else:
        print(f"  [FAIL] FIX {n_issues} CRITICAL ISSUES BEFORE TRAINING")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Check data quality of merged annotations.")
    parser.add_argument('--input_file', '-i', type=str,
                        default='Annotations/Voting/merged_annotations_all.json',
                        help='Path to the merged annotations JSON file to check')
    parser.add_argument('--data_files', '-d', type=str, nargs='*',
                        default=[
                            'mimic_data/test_row_level_v2.json',
                            'mimic_data/val_row_level_v2.json',
                            'mimic_data/train_row_level_v2.json',
                        ],
                        help='Row-level v2 data files used for sentence bounds checking '
                             '(optional; if omitted or files not found, OOB check is skipped)')

    args = parser.parse_args()
    file_path = Path(args.input_file)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return 1
    
    print(f"Loading merged annotations from: {file_path}")
    merged_data = load_merged_annotations(file_path)

    # Load sentence bounds for OOB checking if data files provided
    sentence_bounds = None
    if args.data_files:
        sentence_bounds = load_sentence_bounds(args.data_files)
        if sentence_bounds:
            print(f"Sentence bounds loaded for {len(sentence_bounds)} admissions (OOB check enabled)")
        else:
            print("Sentence bounds not loaded — OOB check will be skipped")

    print("Running quality checks...")
    results = check_data_quality(merged_data, sentence_bounds=sentence_bounds)
    
    print_quality_report(results)
    
    # Save detailed report
    output_path = file_path.parent / "data_quality_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed report saved to: {output_path}")

    # Copy final output to mimic_data/Annotated_Test.json if all checks pass
    if len(results['issues']) == 0:
        annotated_out = Path('mimic_data/Annotated_Test.json')
        annotated_out.parent.mkdir(parents=True, exist_ok=True)
        with open(annotated_out, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2)
        print(f"Quality check passed — annotations copied to: {annotated_out}")

    return 0 if len(results['issues']) == 0 else 1

if __name__ == '__main__':
    exit(main())
