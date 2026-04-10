#!/usr/bin/env python3
"""
Data Quality Check for Merged Annotations
Comprehensive validation script for training data
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
import argparse

def load_merged_annotations(file_path):
    """Load merged annotations file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_data_quality(merged_data):
    """Perform comprehensive data quality checks."""
    
    issues = []
    warnings = []
    stats = defaultdict(int)
    
    stats['total_files'] = len(merged_data)
    
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
                    
                    if '_sources' not in row_data:
                        warnings.append(f"[{file_id}] Medication row {row_id} missing provenance (_sources)")
                    
                stats['medication_rows'] += 1
        
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
                
                # Track evidence scopes
                if 'evidence_scope' in rel:
                    evidence_scopes_seen[rel['evidence_scope']] += 1
                
                # Track confidence levels
                if 'confidence' in rel:
                    confidence_levels_seen[rel['confidence']] += 1
                
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
    
    args = parser.parse_args()
    file_path = Path(args.input_file)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return 1
    
    print(f"Loading merged annotations from: {file_path}")
    merged_data = load_merged_annotations(file_path)
    
    print("Running quality checks...")
    results = check_data_quality(merged_data)
    
    print_quality_report(results)
    
    # Save detailed report
    output_path = file_path.parent / "data_quality_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed report saved to: {output_path}")
    
    return 0 if len(results['issues']) == 0 else 1

if __name__ == '__main__':
    exit(main())
